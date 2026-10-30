"""Compare detected particle geometries with the same joint physics fit.

This adapter deliberately does not alter ``joint_fit``'s model or optimiser. Every
detector receives the same deterministic starting rule, fitting budget and interior
profile. Starts are selected by training loss only; optional validation coordinates
are used solely for diagnostics, never to choose a start.

``reliable`` means that the finite-value, search-bound, separated-sphere and sampled
probability checks below passed. It is not an uncertainty estimate, proof of physical
identification, or evidence of agreement with the material's actual physics.
"""

import math
import time

import numpy as np
import torch

from geom_mesh_net.fields import joint_fit, physics


DEFAULT_STARTS = ((None, None), (1.0, 3.0), (6.0, 12.0))


def _json_safe(value):
    """Keep diagnostic failures serialisable with ``json.dumps(..., allow_nan=False)``."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def geometry_diagnostics(centres, radii):
    """Describe whether spheres meet the analytic solver's separatedness assumption."""
    centres, radii = np.asarray(centres, dtype=float), np.asarray(radii, dtype=float)
    finite = bool(np.isfinite(centres).all() and np.isfinite(radii).all())
    positive = bool(np.all(radii > 0))
    pairs, minimum = 0, None
    if finite and len(radii) > 1:
        i, j = np.triu_indices(len(radii), k=1)
        gaps = np.linalg.norm(centres[i] - centres[j], axis=1) - radii[i] - radii[j]
        pairs, minimum = int(np.sum(gaps < 0)), float(gaps.min())
    return {"finite": finite, "positive_radii": positive,
            "separated": bool(finite and positive and pairs == 0),
            "overlap_pairs": pairs, "minimum_surface_gap": minimum}


def _matrix_diagnostics(model, points, source):
    """Inspect the unclipped screened field only outside the fitted precipitates.

    ``joint_fit.predict(..., matrix_only=True)`` clips its answer, which would hide
    an invalid concentration. No true geometry or observed validation labels enter
    this check. Finite samples do not establish validity everywhere in the domain.
    """
    count = nonfinite = outside = 0
    low, high = math.inf, -math.inf
    with torch.no_grad():
        centres, radii = model.centres(), model.radii()
        for start in range(0, len(points), joint_fit.CHUNK):
            tx = torch.as_tensor(points[start:start + joint_fit.CHUNK], dtype=torch.float64)
            matrix = torch.all(torch.cdist(tx, centres) >= radii[None, :], dim=1)
            if not bool(matrix.any()):
                continue
            values = model.matrix_field(tx[matrix]).detach().numpy()
            finite = np.isfinite(values)
            count += len(values)
            nonfinite += int(np.sum(~finite))
            outside += int(np.sum(finite & ((values < 0) | (values > 1))))
            if finite.any():
                low, high = min(low, float(values[finite].min())), max(high, float(values[finite].max()))
    return {"source": source, "supplied_points": len(points), "matrix_points": count,
            "nonfinite": nonfinite, "outside_probability_range": outside,
            "minimum": low if np.isfinite(low) else None,
            "maximum": high if np.isfinite(high) else None,
            "passed": bool(count > 0 and nonfinite == 0 and outside == 0)}


def fit_particle_physics(centres, radii, x, y, *, width=0.3, knots=6,
                         iterations=150, rounds=3, volume=216000.0,
                         starts=DEFAULT_STARTS, validation_points=None,
                         bound_tolerance=1.01):
    """Return ``(model_or_None, JSON_safe_record)`` for one detected geometry.

    Positions are in the same physical units as ``width`` and ``volume ** (1/3)``.
    ``x, y`` contain fitting observations only. ``validation_points`` may contain
    held-out positions but no held-out labels; they do not affect optimisation or
    start selection. When omitted, probability diagnostics use fitting positions.

    All starts use the observed mean label as ``c_inf``, half that mean as ``c_eq``,
    a six-segment monotone interior by default, and freely fitted centres. A start
    ``(None, None)`` uses the detected mean radius and mean-field screening length
    for ``ell`` and ``xi``. No true geometry, physical constants or oracle enters.
    The caller controls Torch's thread count; this function changes no random seed
    or global threading setting and the fitting procedure has no random steps.

    Overlapping starting spheres are refused. If freely fitted spheres overlap,
    or a fit hits a bound or produces invalid sampled matrix concentrations, its
    model and diagnostics are retained with ``reliable=False``. The wrapper does
    not silently substitute a different start using validation information.
    """
    started = time.perf_counter()
    centres, radii = np.array(centres, dtype=float, copy=True), np.array(radii, dtype=float, copy=True)
    if centres.size == 0:
        centres = centres.reshape(0, 3)
    if centres.ndim != 2 or centres.shape[1] != 3 or radii.ndim != 1 or len(centres) != len(radii):
        raise ValueError("centres must have shape (K, 3) and radii shape (K,)")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x must have shape (N, 3) and y shape (N,)")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("fitting positions and labels must be finite, with labels in [0, 1]")
    check_points = x if validation_points is None else np.asarray(validation_points, dtype=float)
    if check_points.ndim != 2 or check_points.shape[1] != 3 or not np.isfinite(check_points).all():
        raise ValueError("validation_points must be a finite array of shape (N, 3)")
    if not np.isfinite(width) or width <= 0 or not np.isfinite(volume) or volume <= 0:
        raise ValueError("width and volume must be positive and finite")
    if int(knots) != knots or knots < 1 or int(iterations) != iterations or iterations < 1 or int(rounds) != rounds or rounds < 1:
        raise ValueError("knots, iterations and rounds must be positive integers")
    if not np.isfinite(bound_tolerance) or bound_tolerance <= 1:
        raise ValueError("bound_tolerance must be finite and greater than one")
    starts = tuple(tuple(start) for start in starts)
    if not starts or any(len(start) != 2 for start in starts):
        raise ValueError("starts must contain at least one (ell, xi) pair")
    for start in starts:
        if any(value is not None and (not np.isfinite(value) or value <= 0) for value in start):
            raise ValueError("ell and xi starts must be positive finite numbers or None")

    before = geometry_diagnostics(centres, radii)
    record = {"status": "fit_failed", "reliable": False, "flags": [],
              "particles": len(radii), "fitting_atoms": len(x), "constants": None,
              "centres": None, "radii": None, "shape": None, "train_loss": None,
              "geometry_before": before, "geometry_after": None,
              "at_bounds": [], "radius_at_bounds": [], "matrix_probability": None,
              "starts": [], "selected_start": None,
              "configuration": {"width": width, "knots": int(knots), "iterations": int(iterations),
                                "rounds": int(rounds), "volume": volume, "starts": starts,
                                "free_centres": True, "bound_tolerance": bound_tolerance,
                                "selection": "minimum finite training loss"}}

    def finish(model, status, flags):
        record.update(status=status, flags=flags, reliable=status == "ok",
                      seconds=round(time.perf_counter() - started, 3))
        return model, _json_safe(record)

    if not len(radii):
        return finish(None, "no_particles", ["no_particles"])
    if not len(x):
        return finish(None, "no_observations", ["no_observations"])
    if not before["separated"]:
        return finish(None, "invalid_geometry", ["invalid_starting_geometry"])
    lower_radius, upper_radius = joint_fit.RADIUS_BOUNDS
    if np.any((radii < lower_radius) | (radii > upper_radius)):
        return finish(None, "invalid_geometry", ["starting_radius_outside_search_bounds"])

    mean = float(np.clip(y.mean(), 1e-5, 1 - 1e-5))
    best = None
    for number, (ell, xi) in enumerate(starts):
        initial = {"c_eq": mean / 2, "ell": float(radii.mean()) if ell is None else float(ell),
                   "xi": physics.screening_length(radii, volume) if xi is None else float(xi), "c_inf": mean}
        clipped = []
        for name, value in initial.items():
            lo, hi = joint_fit.BOUNDS[name]
            bounded = float(np.clip(value, lo * 1.001, hi / 1.001))
            if bounded != value:
                clipped.append(name)
            initial[name] = bounded
        attempt = {"start": number, "initial_constants": initial,
                   "initial_constants_clipped": clipped, "train_loss": None, "status": "failed"}
        record["starts"].append(attempt)
        try:
            model = joint_fit.JointDiffusionModel(centres, radii, **initial, width=width,
                                                  knots=int(knots), free_centres=True)
            fitted = joint_fit.fit_joint(model, x, y, iterations=int(iterations), rounds=int(rounds))
            loss = float(fitted["train_loss"])
            finite = np.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
            attempt.update(train_loss=loss, status="fitted" if finite else "nonfinite")
            if finite and (best is None or loss < best[2]["train_loss"]):
                best = (number, model, fitted)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
    if best is None:
        return finish(None, "fit_failed", ["all_starts_failed"])

    number, model, fitted = best
    fitted_centres = model.centres().detach().numpy()
    fitted_radii = model.radii().detach().numpy()
    constants = model.constants()
    after = geometry_diagnostics(fitted_centres, fitted_radii)
    at_bounds = [name for name, value in constants.items()
                 if value <= joint_fit.BOUNDS[name][0] * bound_tolerance
                 or value >= joint_fit.BOUNDS[name][1] / bound_tolerance]
    radius_at_bounds = np.flatnonzero((fitted_radii <= lower_radius * bound_tolerance)
                                     | (fitted_radii >= upper_radius / bound_tolerance)).tolist()
    flags = []
    if not after["separated"]:
        flags.append("invalid_fitted_geometry")
    if at_bounds:
        flags.append("physical_constant_at_bound")
    if radius_at_bounds:
        flags.append("radius_at_bound")
    if np.all(y == y[0]):
        flags.append("constant_labels")
    source = "fitting_positions" if validation_points is None else "validation_positions"
    try:
        probability = _matrix_diagnostics(model, check_points, source)
        if not probability["passed"]:
            flags.append("no_matrix_check_points" if probability["matrix_points"] == 0
                         else "invalid_matrix_probability")
    except (RuntimeError, ValueError, FloatingPointError) as error:
        probability = {"source": source, "passed": False, "error": f"{type(error).__name__}: {error}"}
        flags.append("matrix_probability_check_failed")
    record.update(constants=constants, centres=fitted_centres, radii=fitted_radii,
                  shape=model.shape(), train_loss=float(fitted["train_loss"]),
                  round_losses=fitted.get("round_losses", []), selected_start=number,
                  geometry_after=after, at_bounds=at_bounds, radius_at_bounds=radius_at_bounds,
                  matrix_probability=probability)
    return finish(model, "invalid_fit" if flags else "ok", flags)
