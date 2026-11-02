"""Guarded endpoint probes for the experimental sphere fitter.

These diagnostics use fitting observations only. A sign-changing radius cusp is
evidence about a one-coordinate section, never a joint convergence certificate.
No model, fitting data, or saved report is changed by these functions.
"""

import hashlib
import time

import numpy as np
import torch

from . import constrained_joint_fit as cj


def _identity(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_from_fit_state(state, *, configuration=None, candidate_ids=None):
    """Restore an exact exported state or a legacy saved terminal-start record.

    Legacy records require the original configuration including its box length.
    Their probability-level representation is converted back to logits; callers
    should compare the reconstructed NLL with the saved NLL before interpretation.
    Exported raw-coordinate states use the fitter's validating restoration API.
    """
    if "state_version" in state:
        return cj.restore_constrained_state(state)
    if configuration is None or "box" not in configuration:
        raise ValueError("legacy state requires its original configuration including box")
    if any(state.get(key) is None for key in ("centres", "radii", "constants", "shape")):
        raise ValueError("legacy state must contain a complete terminal model")
    shape = state["shape"]
    fixed_ell = configuration.get("fixed_ell")
    if fixed_ell is not None and state["constants"]["ell"] != fixed_ell:
        raise ValueError("legacy ell does not match fixed_ell")
    model = cj.ConstrainedDiffusionModel(
        state["centres"], state["radii"], **state["constants"], levels=shape["levels"],
        knots=shape["knots"], width=shape["width"], box=configuration["box"],
        min_gap=configuration.get("min_gap", 1e-5),
        radius_bounds=configuration.get("radius_bounds", (0.5, 20.)),
        fixed_ell=fixed_ell, candidate_ids=candidate_ids or state.get("candidate_ids"))
    low, high = model.layout.bounds(model.box, model.radius_bounds)
    vector = model.vector.detach().numpy()
    if not np.isfinite(vector).all() or np.any(vector < low) or np.any(vector > high):
        raise ValueError("legacy parameter coordinates are nonfinite or outside bounds")
    model.guard_geometry()
    return model


def _radius_crossings(distances, assignment, particle, radius, knots, sample_indices, count):
    """Finite positive radius crossings; exact ties retain every sample identity."""
    atoms = np.flatnonzero(assignment == particle)
    fractions = np.arange(1, knots + 1) / knots
    with np.errstate(over="ignore", invalid="ignore"):
        values = distances[atoms, particle, None] / fractions[None, :]
    valid = np.isfinite(values) & (values > 0)
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return []
    values = values[rows, columns]
    unique = np.unique(values)
    selected = sorted(unique, key=lambda value: (abs(value - radius), value))[:count]
    output = []
    for value in selected:
        tied = np.flatnonzero(values == value)
        other = unique[unique != value]
        output.append({"radius": float(value), "distance_from_endpoint": float(value - radius),
                       "nearest_other_crossing_distance": float(np.min(abs(other - value))) if len(other) else None,
                       "possible_nonsmooth_terms": (["interior_clamp", "constant_source_continuation"]
                                                    if np.any(columns[tied] + 1 == knots) else ["interior_knot"]),
                       "origins": [{"fitting_position": int(atoms[rows[t]]),
                                    "sample_index": int(sample_indices[atoms[rows[t]]]),
                                    "knot_index": int(columns[t] + 1),
                                    "u": float(fractions[columns[t]])} for t in tied]})
    return output


def _slope_evidence(points, offset):
    """Check same-side finite differences before describing a derivative jump."""
    if not all(point["valid"] for point in points):
        return {"verified": False, "reason": "one_or_more_probe_points_invalid",
                "coordinate_cusp_minimum": False}
    far_left, near_left, near_right, far_right = points
    width = offset / 2
    left_fd = (near_left["mean_nll"] - far_left["mean_nll"]) / width
    right_fd = (far_right["mean_nll"] - near_right["mean_nll"]) / width
    left = near_left["scaled_radius_derivative"]
    right = near_right["scaled_radius_derivative"]
    # A finite difference of nearly equal float64 losses has an unavoidable
    # roundoff scale. Observed within-side slope variation bounds truncation.
    roundoff = 32 * np.finfo(float).eps * max(1., *(abs(p["mean_nll"]) for p in points)) / width
    left_error = roundoff + abs(left - far_left["scaled_radius_derivative"])
    right_error = roundoff + abs(right - far_right["scaled_radius_derivative"])
    verified = abs(left_fd - left) <= left_error and abs(right_fd - right) <= right_error
    jump = right - left
    resolved = abs(jump) > 4 * (left_error + right_error)
    return {"verified": bool(verified), "left_derivative": left, "right_derivative": right,
            "left_finite_difference": left_fd, "right_finite_difference": right_fd,
            "left_error_allowance": left_error, "right_error_allowance": right_error,
            "derivative_jump": jump, "resolved_jump": bool(resolved),
            "coordinate_cusp_minimum": bool(verified and resolved and left < -left_error
                                             and right > right_error)}


def diagnose_radius_knots(model, x, y, *, top_n=4, max_crossings=2,
                          side_offset=1e-6, nearby_radius_window=1e-3, sample_indices=None):
    """Probe interior-knot crossings near the largest radius residuals.

    ``side_offset`` is in the fitter's scaled radius coordinate (radius / 5).
    ``nearby_radius_window`` is in physical length units and labels proximity,
    not convergence. ``sample_indices`` optionally identifies rows in the source
    dataset; fitting positions are always retained as well. The baseline uses
    fresh nearest-surface assignments. Each probe reports both fresh assignments
    and those frozen at the endpoint. All likelihoods use the original evaluator's
    geometry and sampled raw-probability guards; failed probes remain explicit.
    """
    started = time.perf_counter()
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1))):
        raise ValueError("finite nonempty fitting positions and labels in [0, 1] required")
    if any(not np.isscalar(v) or not np.isfinite(v) or v < 1 or int(v) != v
           for v in (top_n, max_crossings)):
        raise ValueError("top_n and max_crossings must be positive integers")
    if not np.isfinite(side_offset) or side_offset <= 0 or not np.isfinite(nearby_radius_window) or nearby_radius_window < 0:
        raise ValueError("positive finite side_offset and nonnegative nearby_radius_window required")
    indices = np.arange(len(x)) if sample_indices is None else np.asarray(sample_indices)
    if indices.shape != (len(x),) or indices.dtype.kind not in "iu" or len(np.unique(indices)) != len(indices):
        raise ValueError("sample_indices must be unique integers, one per fitting row")
    # Work on a restored copy: evaluator cache changes cannot alter the caller.
    working = cj.restore_constrained_state(model.export_state())
    vector = working.vector.detach().numpy().copy()
    tx, ty = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    assignment = working.assign(tx)
    fixed = cj._fixed_probes(working.box)
    evaluator = cj._Evaluation(working, tx, ty, assignment, 1, [], fixed)
    result = evaluator.evaluate(vector)
    constraints = cj.geometry_constraints(working.layout, box=working.box, min_gap=working.min_gap)
    low, high = working.layout.bounds(working.box, working.radius_bounds)
    matrix, slack = cj._linearized_constraints(vector, result, constraints, low, high)
    direction, _, certificate = cj._project_direction(-result[1], matrix, slack)
    k = working.layout.count
    radii = vector[3*k:4*k] * cj.GEOMETRY_SCALE
    selected = np.argsort(-np.abs(direction[3*k:4*k]), kind="stable")[:int(top_n)]
    with torch.no_grad():
        distances = torch.cdist(tx, working.centres()).numpy()
    output = {"schema": "radius-knot-diagnostics-v1", "model_version": working.model_version,
              "scope": "fitting observations; one-coordinate sections; not joint convergence",
              "joint_convergence_assessed": False,
              "model_configuration": working.export_state()["configuration"],
              "fitting_identity": _identity(x, y, indices), "parameter_identity": _identity(vector),
              "frozen_assignment_identity": _identity(assignment.numpy()),
              "fitting_atoms": len(x), "mean_nll": result[0],
              "full_projected_gradient_inf": float(np.max(abs(direction))),
              "projection_certificate": certificate, "baseline_guard": evaluator.trace[-1],
              "settings": {"top_n": int(top_n), "max_crossings": int(max_crossings),
                           "side_offset_scaled": side_offset, "nearby_radius_window": nearby_radius_window},
              "particles": []}
    cache = {}
    def probe(radius, particle, fresh):
        point = vector.copy()
        point[3*k+particle] = radius / cj.GEOMETRY_SCALE
        working.set_vector(point)
        owners = working.assign(tx) if fresh else assignment
        key = (particle, float(radius), _identity(owners.numpy()))
        if key in cache:
            return cache[key]
        evaluated = cj._Evaluation(working, tx, ty, owners, 1, [], fixed)
        record = {"radius": float(radius), "scaled_radius": float(radius / cj.GEOMETRY_SCALE),
                  "assignment_changes": int((owners != assignment).sum()), "valid": False,
                  "mean_nll": None, "scaled_radius_derivative": None}
        try:
            values = evaluated.evaluate(point, derivatives=False)
            gradient = torch.autograd.grad(evaluated.cached_loss, evaluated.cached_parameter)[0]
            slope = float(gradient[3*k+particle])
            valid = np.isfinite(slope) and min(values[2]) >= -cj.DOMAIN_TOLERANCE
            record.update(valid=bool(valid), mean_nll=values[0], scaled_radius_derivative=slope,
                          physical_radius_derivative=slope / cj.GEOMETRY_SCALE)
        except (cj.DomainError, RuntimeError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["guard"] = evaluated.trace[-1] if evaluated.trace else None
        cache[key] = record
        return record
    for particle in selected:
        crossings = _radius_crossings(distances, assignment.numpy(), particle, radii[particle],
                                      working.layout.knots, indices, int(max_crossings))
        item = {"particle_index": int(particle), "candidate_id": working.candidate_ids[particle],
                "endpoint_radius": float(radii[particle]),
                "raw_radius_gradient": float(result[1][3*k+particle]),
                "joint_projected_radius_component": float(direction[3*k+particle]), "crossings": []}
        for crossing in crossings:
            physical_offset = side_offset * cj.GEOMETRY_SCALE
            if crossing["nearest_other_crossing_distance"] is not None:
                physical_offset = min(physical_offset, crossing["nearest_other_crossing_distance"] / 4)
            offset = physical_offset / cj.GEOMETRY_SCALE
            locations = [crossing["radius"] + factor * physical_offset for factor in (-1., -.5, .5, 1.)]
            crossing.update(actual_offset_scaled=offset, near_endpoint=abs(crossing["distance_from_endpoint"]) <= nearby_radius_window,
                            assignment_modes={})
            for mode in ("frozen", "fresh"):
                points = [probe(radius, particle, mode == "fresh") for radius in locations]
                evidence = _slope_evidence(points, offset) if len(set(locations)) == 4 else {
                    "verified": False, "reason": "bracket_below_float_resolution", "coordinate_cusp_minimum": False}
                crossing["assignment_modes"][mode] = {"points": points, **evidence,
                    "nearby_coordinate_cusp": bool(crossing["near_endpoint"] and evidence["coordinate_cusp_minimum"])}
            item["crossings"].append(crossing)
        output["particles"].append(item)
    guards = [output["baseline_guard"], *(value["guard"] for value in cache.values())]
    output["diagnostic_evaluations"] = {
        "outside_fitting_budget": True, "total": 1 + len(cache),
        "likelihood_evaluated": sum(bool(guard and guard["likelihood_evaluated"]) for guard in guards),
        "probe_cache_hits": sum(8 * len(particle["crossings"]) for particle in output["particles"]) - len(cache)}
    output["seconds"] = time.perf_counter() - started
    return cj._json_safe(output)
