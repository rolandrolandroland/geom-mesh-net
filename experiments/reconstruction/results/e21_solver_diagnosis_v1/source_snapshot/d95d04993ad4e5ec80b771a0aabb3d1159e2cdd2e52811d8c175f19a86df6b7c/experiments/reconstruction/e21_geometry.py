"""E21: one observed candidate catalogue, four geometry/physics procedures.

Run a bounded development fitter check first::

    python -m experiments.reconstruction.e21_geometry --preflight

Then run the four-arm smoke comparison (only after a successful preflight)::

    python -m experiments.reconstruction.e21_geometry

Each attempt is immutable. Failed first attempts stay in the denominator. The
reserved test cannot be generated before the full confirmation gate passes.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.reconstruction import e21_protocol as protocol
from experiments.reconstruction import e21_simulation as simulation
from experiments.reconstruction import stage2_field as s2
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.fields import boundary_refinement as boundary
from geom_mesh_net.fields import cluster_extraction as ce
from geom_mesh_net.fields import constrained_joint_fit as constrained
from geom_mesh_net.fields import joint_fit, particle_physics
from geom_mesh_net.neural import field_diagnostics as diagnostics


RESULTS = Path(__file__).parent / "results" / "e21_geometry_v3"
ARTIFACTS = Path(__file__).parent / "results" / "e21_artifacts" / "v3"


def strict_log_loss(y, q):
    """Validate raw values first; the log safeguard only treats exact endpoints."""
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    if y.shape != q.shape or not len(y):
        raise ValueError("nonempty matching label and probability arrays required")
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("invalid raw probability; no clipped score is supplied")
    safe = np.clip(q, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return float(-np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe)))


def freeze_matches(centres, true_centres, true_radii):
    """Truth associations are scored once and cannot change as fitted radii move."""
    truth, candidates = ce.match_precipitates(true_centres, true_radii, centres)
    return {"truth_indices": truth.tolist(), "candidate_indices": candidates.tolist(),
            "identity": protocol.array_hash(truth, candidates),
            "matched": len(truth), "candidate_count": len(centres), "truth_count": len(true_radii),
            "precision": len(truth) / len(centres) if len(centres) else 0.,
            "recall": len(truth) / len(true_radii) if len(true_radii) else 0.}


def gaussian_catalogue(x, y, spec, design):
    """Detection, CV and profiles depend only on outer fitting observations."""
    box = design["simulation"]["box"]
    fit = fb.fit_baselines(x, y.astype(bool), seed=spec["seeds"]["resampling"],
                           upper=box, adaptive_k=(4,), adaptive_c=(.5,))
    guests, atoms = ce.smoothed_counts(x, y.astype(bool), fit.bandwidth, upper=box, grid=.5)
    squared = ce.squared_kernel_counts(x, fit.bandwidth, upper=box, grid=.5)
    segmented = ce.segment(ce.guest_fraction(guests, atoms), atoms, grid=.5,
                           method="significance", squared=squared)
    catalog = boundary.CandidateCatalog(segmented.centres, segmented.radii,
                 provenance={"case_id": spec["case_id"], "detector": "Gaussian significance",
                             "bandwidth": fit.bandwidth, "observations": protocol.array_hash(x, y)})
    counts = fb.SmoothedCounts(x, y.astype(bool), bandwidths=(fit.bandwidth,), upper=box)
    return catalog, lambda points: counts.fixed(points, 0), asdict(fit)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _old_predictions(model, points):
    # Historical A retains its original clipped likelihood. This is documented,
    # and its metrics never imply that a legacy optimizer established convergence.
    return joint_fit.predict(model, points)


def legacy_raw_diagnostics(model, points, box):
    """Inspect A's actual blend before its historical clipping, without refitting."""
    geometry = constrained.geometry_diagnostics(_numpy(model.centres()), _numpy(model.radii()), box=box)
    if not geometry["feasible"]:
        return {"passed": False, "geometry": geometry, "reason": "invalid_geometry_no_physical_check"}
    grid = np.linspace(0, box, 5)
    fixed = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    directions = constrained.surface_directions()
    surface = _numpy(model.centres())[:, None, :] + _numpy(model.radii())[:, None, None] * directions[None]
    points = np.concatenate([points, fixed, surface.reshape(-1, 3)])
    full_values, matrix_values, outside_values = [], [], []
    with torch.no_grad():
        radii, centres = model.radii(), model.centres()
        for start in range(0, len(points), joint_fit.CHUNK):
            tx = torch.as_tensor(points[start:start + joint_fit.CHUNK], dtype=torch.float64)
            distance = torch.cdist(tx, centres)
            matrix = model.matrix_field(tx)
            assignment = torch.argmin(distance - radii[None], dim=1)
            nearest = distance.gather(1, assignment[:, None]).squeeze(1)
            radius = radii[assignment]
            interior = model.interior(torch.clamp(nearest / radius, max=1.))
            blend = torch.sigmoid((radius - nearest) / model.width())
            full_values.append(_numpy(matrix + (interior - matrix) * blend))
            matrix_values.append(_numpy(matrix))
            outside_values.append(_numpy(torch.all(distance >= radii[None], dim=1)))
    checked = constrained.raw_probability_diagnostics(np.concatenate(full_values),
                    np.concatenate(matrix_values), np.concatenate(outside_values))
    checked["geometry"] = geometry
    checked["historical_predictions_and_fitting_remain_clipped"] = True
    return checked


def score_model(model, record, arm, arrays, metadata, matching):
    radii = _numpy(model.radii())
    centres = _numpy(model.centres())
    true_indices = np.asarray(matching["truth_indices"], dtype=int)
    candidate_indices = np.asarray(matching["candidate_indices"], dtype=int)
    if not len(true_indices):
        raise ValueError("no original-catalogue truth matches; radius score undefined")
    radius_mare = float(np.mean(np.abs(radii[candidate_indices] / arrays["radii"][true_indices] - 1)))
    centre_errors = np.linalg.norm(centres[candidate_indices] - arrays["centres"][true_indices], axis=1)
    rematched_truth, rematched_candidates = ce.match_precipitates(arrays["centres"], arrays["radii"], centres)
    constants, truth = model.constants(), metadata["field"]
    held = arrays["heldback"]
    matrix = arrays["owner"][held] < 0
    predict = (lambda points: _old_predictions(model, points)) if arm == "A" else model.raw_predict
    q = _numpy(predict(arrays["x"][held]))
    full_loss = strict_log_loss(arrays["y"][held], q)
    matrix_loss = strict_log_loss(arrays["y"][held][matrix], q[matrix])
    removed_q = _numpy(predict(arrays["x"][arrays["removed"]]))
    removed_loss = strict_log_loss(arrays["y"][arrays["removed"]], removed_q)
    if arm == "A":
        feasible = constrained.geometry_diagnostics(_numpy(model.centres()), radii,
                           box=metadata["configuration"]["box"])["feasible"]
        record["raw_probability_diagnostics"] = legacy_raw_diagnostics(model,
                         arrays["x"][arrays["fitting"] | arrays["heldback"]], metadata["configuration"]["box"])
        valid = record["raw_probability_diagnostics"]["passed"]
        converged, contract = False, False
    else:
        feasible = record["geometry_feasible"]
        valid = record["sampled_probability_valid"]
        converged = record["converged"] and record["assignment_stable"]
        # Every objective trace must show domain validation before likelihood.
        evaluations = [e for start in record["starts"] for e in start.get("evaluations", [])]
        contract = any(e.get("likelihood_evaluated") is True for e in evaluations) and all(
            e.get("likelihood_evaluated") is False or
            (e.get("likelihood_evaluated") is True and e.get("geometry_feasible") is True
             and e.get("probability_valid") is True) for e in evaluations)
    metrics = {"feasible": bool(feasible), "probability_valid": bool(valid),
               "converged": bool(converged), "objective_contract_passed": bool(contract),
               "search_bound_contact": bool(record.get("at_bounds") or record.get("radius_at_bounds")),
               "ambiguous": bool(record.get("ambiguous", False)), "radius_mare": radius_mare,
               "ell_absolute_relative_error": abs(constants["ell"] / truth["ell"] - 1),
               "xi_absolute_relative_error": abs(constants["xi"] / truth["xi"] - 1),
               "full_heldback_log_loss": full_loss, "matrix_heldback_log_loss": matrix_loss,
               "full_removed_log_loss": removed_loss,
               "centre_mean_error_fixed_matches": float(centre_errors.mean()),
               "centre_maximum_error_fixed_matches": float(centre_errors.max()),
               "initial_catalogue_precision": matching["precision"],
               "initial_catalogue_recall": matching["recall"],
               "secondary_rematched_count": len(rematched_truth),
               "secondary_rematched_radius_mare": float(np.mean(np.abs(
                    radii[rematched_candidates] / arrays["radii"][rematched_truth] - 1)))
                    if len(rematched_truth) else None}
    return metrics


def validate_attempt_execution(row, arm, catalog, metadata, matching, spec, report):
    """An orphan is reusable only under the exact execution that produced it."""
    if (not isinstance(row.get("execution_fingerprint"), str)
            or row["execution_fingerprint"] != report["fingerprint"]):
        raise RuntimeError("attempt execution provenance is missing or changed; use a new artifact directory")
    if any(row.get(key) != spec[key] for key in ("case_id", "phase", "case_index")) or (
            row.get("arm") != arm or type(row.get("attempt")) is not int or row["attempt"] != 1):
        raise RuntimeError("orphaned attempt has a mismatched case, arm or attempt identity")
    if (row.get("data_identity") != metadata["data_identity"]
            or row.get("catalogue_identity") != catalog.fingerprint
            or row.get("matching_identity") != matching["identity"]):
        raise RuntimeError("orphaned attempt has mismatched data or catalogue")


def fit_arm(arm, geometry, catalog, arrays, metadata, matching, spec, design, report, output, artifacts):
    existing = [r for r in report["rows"] if r["case_id"] == spec["case_id"]
                and r["arm"] == arm and r["attempt"] == 1]
    if existing:
        validate_attempt_execution(existing[0], arm, catalog, metadata, matching, spec, report)
        return existing[0]
    try:
        folder = protocol.allocate_attempt_dir(artifacts / "fits", spec["case_id"], arm, 1)
    except FileExistsError:
        folder = artifacts / "fits" / protocol.digest(spec["case_id"])[:24] / arm / "attempt_001"
        result = folder / "result.json"
        row = json.loads((result if result.exists() else folder / "started.json").read_text())
        # Check before writing anything into a potentially foreign artifact tree.
        validate_attempt_execution(row, arm, catalog, metadata, matching, spec, report)
        if not result.exists():
            row.update(status="error", error="previous process interrupted; canonical attempt is not retried",
                       elapsed_seconds=None)
            protocol.write_report(result, row)
        protocol.record_attempt(report, row)
        protocol.write_report(output, report)
        return row
    started = time.perf_counter()
    row = {**{key: spec[key] for key in ("case_id", "phase", "case_index")}, "arm": arm,
           "attempt": 1, "data_identity": metadata["data_identity"],
           "execution_fingerprint": report["fingerprint"],
           "catalogue_identity": catalog.fingerprint, "matching_identity": matching["identity"],
           "status": "error", "error": "attempt started but not completed", "metrics": {}}
    protocol.write_report(folder / "started.json", row)
    try:
        radii, preparation = constrained.prepare_feasible_geometry(geometry.centres, geometry.radii,
                                    candidate_ids=catalog.candidate_ids, box=design["simulation"]["box"])
        row["preparation"] = preparation
        if radii is None:
            raise ValueError(f"identity-preserving preparation failed: {preparation['reason']}")
        config = design["physics"]
        options = dict(width=config["width"], knots=config["knots"], rounds=config["rounds"],
                       starts=config["starts"], validation_points=arrays["x"][arrays["heldback"]])
        x, y = arrays["x"][arrays["fitting"]], arrays["y"][arrays["fitting"]].astype(float)
        if arm == "A":
            model, record = particle_physics.fit_particle_physics(geometry.centres, radii, x, y,
                                iterations=config["maxiter"], volume=design["simulation"]["box"] ** 3, **options)
        else:
            model, record = constrained.fit_constrained(geometry.centres, radii, x, y,
                                candidate_ids=catalog.candidate_ids, box=design["simulation"]["box"],
                                maxiter=config["maxiter"], max_evaluations=config["max_evaluations"],
                                backend=config["backend"], progress_callback=lambda event: print(
                                    json.dumps({"case": spec["case_index"], "arm": arm, **protocol.json_safe(event)}),
                                    flush=True), **options)
        row["fit"] = record
        protocol.write_report(folder / "fit.json", record)
        if model is None:
            raise RuntimeError(f"{record['status']}: {record.get('flags', [])}")
        row["metrics"] = score_model(model, record, arm, arrays, metadata, matching)
        protocol.write_report(folder / "fit.json", record)
        row["status"] = "ok"
        row.pop("error")
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["elapsed_seconds"] = time.perf_counter() - started
    protocol.write_report(folder / "result.json", row)
    protocol.record_attempt(report, row)
    protocol.write_report(output, report)
    print(json.dumps({"case": spec["case_index"], "arm": arm, "status": row["status"],
                      "seconds": round(row["elapsed_seconds"], 2), "error": row.get("error")}), flush=True)
    return row


def numerically_ready(row):
    return row["status"] == "ok" and all(row["metrics"].get(name) is True for name in
                   ("feasible", "probability_valid", "converged", "objective_contract_passed"))


def record_unrun(report, indices, design, reason):
    """Keep every prescribed identity after a terminal dependency/preparation failure."""
    for index in indices:
        spec = protocol.case_spec(report["phase"], index, design)
        existing = [r for r in report["rows"] if r["case_id"] == spec["case_id"]]
        identity = {key: existing[0][key] if existing else "unavailable:" + protocol.digest(spec)
                    for key in ("data_identity", "catalogue_identity", "matching_identity")}
        for arm in protocol.ARMS:
            if not any(r["arm"] == arm and r["attempt"] == 1 for r in existing):
                protocol.record_attempt(report, {**spec, **identity, "arm": arm, "attempt": 1,
                    "execution_fingerprint": report["fingerprint"],
                    "status": "error", "error": reason, "execution": "not_run_dependency_failure",
                    "metrics": {}, "elapsed_seconds": None})


def preserve_sources(output_folder, provenance):
    snapshot = output_folder / "source_snapshot" / protocol.digest(provenance)
    snapshot.mkdir(parents=True, exist_ok=True)
    for relative, expected in provenance["sources"].items():
        destination = snapshot / relative
        content = (protocol.ROOT / relative).read_bytes()
        import hashlib
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError("source changed during manifest creation")
        if destination.exists():
            if destination.read_bytes() != content:
                raise RuntimeError("preserved source snapshot was altered")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
    protocol.write_report(snapshot / "provenance.json", provenance)


def case_execution_provenance(folder, spec, report, *, create=False):
    """Bind data, fits and neural caches before reading any preexisting case data.

    A new report must not give an old artifact tree new source/runtime provenance.
    Cases have phase-specific identities, so freezing before confirmation creates
    distinct case bindings without changing completed development artifacts.
    """
    expected = {"schema": "e21-case-execution-v1", "case_id": spec["case_id"],
                "execution_fingerprint": report["fingerprint"], "provenance": report["provenance"]}
    if protocol.digest(report["provenance"]) != report["fingerprint"]:
        raise RuntimeError("report execution provenance is inconsistent")
    path = folder / "execution_provenance.json"
    if create and not path.exists():
        protocol.write_report(path, expected)
    elif not path.exists() or json.loads(path.read_text()) != expected:
        raise RuntimeError("case execution provenance is missing or changed; use a new artifact directory")
    return expected


def run_case(index, phase, design, report, output, artifacts, preflight=False):
    spec = protocol.case_spec(phase, index, design)
    folder = artifacts / "cases" / protocol.digest(spec["case_id"])[:24]
    if folder.exists():
        case_execution_provenance(folder, spec, report)
        arrays, metadata = simulation.load_case(folder, spec, design)
    else:
        arrays, metadata = simulation.generate_case(spec, design)
        simulation.save_case(folder, arrays, metadata)
        case_execution_provenance(folder, spec, report, create=True)
    started = time.perf_counter()
    x, y = arrays["x"][arrays["fitting"]], arrays["y"][arrays["fitting"]].astype(float)
    catalog, gaussian, baseline = gaussian_catalogue(x, y, spec, design)
    matching = freeze_matches(catalog.centres, arrays["centres"], arrays["radii"])
    shared = {"catalogue": catalog.to_dict(), "baseline": baseline, "matching": matching,
              "data_metadata": metadata, "gaussian_preparation_seconds": time.perf_counter() - started}
    protocol.write_report(folder / "preparation.json", shared)
    print(json.dumps({"case": index, "candidates": len(catalog.radii),
                      "true_particles": len(arrays["radii"]), "fitting_atoms": len(x)}), flush=True)
    b = fit_arm("B", catalog.geometry(), catalog, arrays, metadata, matching, spec,
                design, report, output, artifacts)
    if preflight or not numerically_ready(b):
        return numerically_ready(b)
    fit_arm("A", catalog.geometry(), catalog, arrays, metadata, matching, spec,
            design, report, output, artifacts)
    samples = boundary.sample_candidate_profiles(catalog, box=design["simulation"]["box"])
    gaussian_proposals = boundary.propose_radii(catalog, samples, gaussian)
    c = boundary.apply_proposals_with_fallback(catalog, gaussian_proposals)
    c_row = fit_arm("C", c, catalog, arrays, metadata, matching, spec, design, report, output, artifacts)
    if not numerically_ready(c_row):
        return False
    fitting, field_design = simulation.neural_fitting_data(arrays, spec, design, s2.load_design())
    training_folder = folder / "fourier"
    training_started = time.perf_counter()
    if training_folder.exists():
        models, records = diagnostics.load_fourier_artifacts(fitting, field_design,
                         artifact_dir=training_folder, device=design["runtime"]["neural_device"])
    else:
        models, records = diagnostics.fit_fourier_with_artifacts(fitting, field_design,
                         artifact_dir=training_folder, device=design["runtime"]["neural_device"],
                         periodic_every=250 if phase == "development" else None,
                         compare_passes=phase == "development" and index in design["smoke_cases"])
    training_flags = diagnostics.training_failure_flags(records)
    protocol.write_report(folder / "training_failures.json", training_flags)
    if training_flags["primary"]:
        raise RuntimeError("nonfinite neural run retained in artifacts; comparison is failed")
    fourier = lambda points: s2.predict_procedure(models, points, design["runtime"]["neural_device"])
    neural_proposals = boundary.propose_radii(catalog, samples, fourier)
    d = boundary.apply_proposals_with_fallback(catalog, neural_proposals, fallback=c)
    shared.update(gaussian_proposals=asdict(gaussian_proposals), fourier_proposals=asdict(neural_proposals),
                  neural_training_seconds=time.perf_counter() - training_started)
    protocol.write_report(folder / "preparation.json", shared)
    d_row = fit_arm("D", d, catalog, arrays, metadata, matching, spec, design, report, output, artifacts)
    if phase == "development":
        prerefit = [diagnostics.load_checkpoint(r["prerefit_artifact"]["path"],
                    device=design["runtime"]["neural_device"],
                    expected_sha256=r["prerefit_artifact"]["sha256"])[0] for r in records]
        fields = {"gaussian": gaussian, "final_refit": fourier,
                  "selected_prerefit": lambda points: s2.predict_procedure(prerefit, points, design["runtime"]["neural_device"])}
        if not training_flags["diagnostic"] and all("diagnostic_refit" in r for r in records):
            alternate = [diagnostics.load_checkpoint(r["diagnostic_refit"]["artifacts"]["selected"]["path"],
                          device=design["runtime"]["neural_device"],
                          expected_sha256=r["diagnostic_refit"]["artifacts"]["selected"]["sha256"])[0] for r in records]
            fields["matched_pass_refit"] = lambda points: s2.predict_procedure(alternate, points, design["runtime"]["neural_device"])
        held = arrays["heldback"]
        scored = diagnostics.score_particle_signal(fields, arrays["x"][held], arrays["y"][held],
                         arrays["centres"], arrays["radii"], observed_coords=x, observed_labels=y)
        protocol.write_report(folder / "signal_diagnostics.json", scored)
    return numerically_ready(d_row)


def preflight_readiness(report, design):
    """Recompute all four canonical B outcomes, never the requested subset alone.

    Other prescribed arms/cases may still be absent during preflight, but every
    existing row must satisfy the full protocol's identity and schema checks.
    """
    protocol.validate_design(design)
    if report.get("phase") != "development":
        return {"ready": False, "expected_cases": len(design["smoke_cases"]),
                "present_cases": 0, "numerically_ready_cases": 0,
                "accounting_errors": ["preflight requires a development report"]}
    try:
        rows, errors, _ = protocol._accounting(report, design)
    except (ValueError, TypeError, KeyError) as error:
        rows, errors = [], [f"malformed development accounting: {error}"]
    errors = [error for error in errors if not (
        error.startswith("missing ") and error.endswith(" canonical case/arm records"))]
    expected = {protocol.case_spec("development", index, design)["case_id"]
                for index in design["smoke_cases"]}
    preflight = [row for row in rows if row.get("arm") == "B" and row.get("case_id") in expected]
    present = {row["case_id"] for row in preflight}
    numerical = sum(numerically_ready(row) for row in preflight) if not errors else 0
    return {"ready": not errors and present == expected and len(preflight) == len(expected)
                     and numerical == len(expected),
            "expected_cases": len(expected), "present_cases": len(present),
            "numerically_ready_cases": numerical, "accounting_errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=protocol.PHASES, default="development")
    parser.add_argument("--preflight", action="store_true", help="four development B fits; no neural training")
    parser.add_argument("--all-development", action="store_true")
    parser.add_argument("--development-cases", type=int, nargs="+", default=None,
                        help="run only these smoke-case indices during development preflight; never changes readiness scope")
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--design", type=Path, default=protocol.DESIGN_PATH)
    args = parser.parse_args(argv)
    args.output, args.artifacts = args.output.resolve(), args.artifacts.resolve()
    design = protocol.load_design(args.design)
    if args.development_cases is not None:
        if args.phase != "development" or not args.preflight or args.all_development:
            raise ValueError("--development-cases requires development --preflight without --all-development")
        if (not args.development_cases or len(set(args.development_cases)) != len(args.development_cases)
                or any(index not in design["smoke_cases"] for index in args.development_cases)):
            raise ValueError("--development-cases must contain unique indices from the four declared smoke cases")
    torch.set_num_threads(design["runtime"]["threads"])
    provenance = protocol.make_provenance(design, {"field_design": s2.load_design()})
    output = args.output / f"{args.phase}.json"
    report = protocol.read_compatible(output, provenance) if output.exists() else protocol.new_report(args.phase, design, provenance)
    if args.phase == "development":
        initial_readiness = preflight_readiness(report, design)
        if initial_readiness["accounting_errors"]:
            raise RuntimeError("invalid development report: " + "; ".join(initial_readiness["accounting_errors"]))
    if args.preflight and args.phase != "development":
        raise ValueError("preflight is development only")
    if args.phase != "development":
        if design["status"] != "frozen":
            raise RuntimeError("freeze the design after development before confirmation")
        development = json.loads((args.output / "development.json").read_text())
        if any(key not in development.get("provenance", {}) for key in ("design", "sources", "runtime", "inputs")):
            raise RuntimeError("incomplete development provenance")
        development_design = development["provenance"]["design"]
        expected_design = dict(design, status=development_design["status"])
        if expected_design != development_design:
            raise RuntimeError("freeze changes only status; changed settings require a new development version")
        old_sources, new_sources = development["provenance"]["sources"], provenance["sources"]
        design_source = str(protocol.DESIGN_PATH.relative_to(protocol.ROOT))
        if ({k: v for k, v in old_sources.items() if k != design_source}
                != {k: v for k, v in new_sources.items() if k != design_source}
                or development["provenance"]["runtime"] != provenance["runtime"]
                or development["provenance"]["inputs"] != provenance["inputs"]):
            raise RuntimeError("implementation or runtime changed since development")
        full_development = protocol.evaluate_gate(development, development_design)
        if (not protocol.smoke_gate(development, development_design)["passed"]
                or not full_development["complete"]
                or not all(numerically_ready(r) for r in protocol.canonical_rows(development) if r["arm"] != "A")):
            raise RuntimeError("development smoke is not numerically ready")
    if args.phase == "test":
        confirmation = protocol.read_compatible(args.output / "confirmation.json", provenance)
        protocol.require_test_permission(confirmation, design, provenance)
    indices = design["scope"][args.phase]
    if args.phase == "development" and not args.all_development:
        indices = design["smoke_cases"]
    if args.development_cases is not None:
        indices = args.development_cases
    if args.all_development and not protocol.smoke_gate(report, design)["passed"]:
        raise RuntimeError("four-arm smoke must pass before expanding development")
    if args.phase == "development" and not args.preflight:
        if not preflight_readiness(report, design)["ready"]:
            raise RuntimeError("all four canonical B preflight fits must be numerically ready before neural comparison")
    protocol.write_report(output, report)
    preserve_sources(args.output, provenance)
    ready = True
    for index in indices:
        try:
            case_ready = run_case(index, args.phase, design, report, output, args.artifacts, args.preflight)
        except (ValueError, RuntimeError, OSError, FloatingPointError, np.linalg.LinAlgError) as error:
            record_unrun(report, [index], design, f"case preparation failed: {type(error).__name__}: {error}")
            protocol.write_report(output, report)
            case_ready = False
        ready = ready and case_ready
        if not case_ready and not args.preflight:
            report["stopped"] = "numerical prerequisite failed; do not scale or open reserved outcomes"
            record_unrun(report, indices, design, report["stopped"])
            break
    report["preflight_readiness"] = preflight_readiness(report, design) if args.phase == "development" else None
    report["preflight_ready"] = (report["preflight_readiness"]["ready"]
                                 if report["preflight_readiness"] is not None else None)
    report["smoke_gate"] = protocol.smoke_gate(report, design) if args.phase == "development" else None
    report["gate"] = protocol.evaluate_gate(report, design)
    protocol.write_report(output, report)
    print(json.dumps({"preflight_ready": report["preflight_ready"], "smoke_gate": report["smoke_gate"],
                      "gate_passed": report["gate"]["passed"]}), flush=True)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
