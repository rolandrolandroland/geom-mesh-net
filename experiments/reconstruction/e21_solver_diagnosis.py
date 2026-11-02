"""Staged numerical diagnosis on E21 development case 0 and an exact-model control.

This runner cannot promote a method or access confirmation/test realizations.
Independent ablations and sequential warm starts have distinct identities/budgets.
Run ``python -m experiments.reconstruction.e21_solver_diagnosis`` from the root.
"""

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.reconstruction import e21_geometry as e21
from experiments.reconstruction import e21_protocol as protocol
from experiments.reconstruction import e21_simulation as simulation
from geom_mesh_net.fields import constrained_joint_fit as fitter
from geom_mesh_net.fields import solver_diagnostics
from geom_mesh_net.fields.boundary_refinement import CandidateCatalog


ROOT = protocol.ROOT
DESIGN_PATH = Path(__file__).with_name("e21_solver_diagnosis_design.json")
RESULTS = Path(__file__).parent / "results" / "e21_solver_diagnosis_v1"
ARTIFACTS = Path(__file__).parent / "results" / "e21_artifacts" / "diagnosis_v1"
STAGE_MASKS = {"chemistry": (False, False), "radii": (False, True),
               "centres": (True, False), "joint": (True, True),
               "warm_radii": (False, True), "warm_joint": (True, True)}
PARENTS = {"chemistry": None, "radii": None, "centres": None, "joint": None,
           "warm_radii": "chemistry", "warm_joint": "warm_radii"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design(path=DESIGN_PATH):
    design = json.loads(Path(path).read_text())
    return validate_design(design)


def validate_design(design):
    if design["observed"]["phase"] != "development" or design["observed"]["case_index"] != 0:
        raise ValueError("diagnosis is restricted to development case 0")
    if design.get("promotion") is not False or design.get("reserved_data_access") is not False:
        raise ValueError("diagnosis cannot promote or release reserved data")
    if [s["id"] for s in design["stages"]] != list(STAGE_MASKS):
        raise ValueError("the declared diagnostic stage inventory changed")
    for stage in design["stages"]:
        if ((stage["fit_centres"], stage["fit_radii"]) != STAGE_MASKS[stage["id"]]
                or stage["parent"] != PARENTS[stage["id"]]):
            raise ValueError("stage mask or lineage changed")
    if design["warm_scope"] != ["observed"]:
        raise ValueError("warm chains are observed-case diagnostics only")
    config = design["physics"]
    if config["backend"] != "feasible_lbfgs" or any(
            type(config[k]) is not int or config[k] < 1 for k in ("maxiter", "max_evaluations", "rounds")):
        raise ValueError("invalid diagnostic solver budget")
    if design["runtime"] != {"threads": 3, "device": "cpu", "dtype": "float64"}:
        raise ValueError("diagnostic runtime must remain CPU float64 with three threads")
    return design


def load_observed_case(design, root=ROOT):
    """Validate the pinned archive, then return only its fitting observations."""
    scope = design["observed"]
    if scope["phase"] != "development" or scope["case_index"] != 0:
        raise ValueError("diagnosis is restricted to development case 0")
    root = Path(root)
    report_path, folder = root / scope["report"], root / scope["case_folder"]
    if file_hash(report_path) != scope["report_sha256"]:
        raise RuntimeError("archived report identity changed")
    archived = json.loads(report_path.read_text())
    if archived["phase"] != "development":
        raise RuntimeError("archived report is not development")
    rows = [row for row in archived["rows"] if row["phase"] == "development"
            and row["case_index"] == 0 and row["arm"] == "B" and row["attempt"] == 1]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one canonical archived baseline")
    row = rows[0]
    archived_design = archived["provenance"]["design"]
    spec = protocol.case_spec("development", 0, archived_design)
    if row["case_id"] != spec["case_id"] or row["data_identity"] != scope["data_identity"]:
        raise RuntimeError("archived case/data identity changed")
    if file_hash(folder / "preparation.json") != scope["preparation_sha256"]:
        raise RuntimeError("archived catalogue preparation changed")
    preparation = json.loads((folder / "preparation.json").read_text())
    catalog = CandidateCatalog(**preparation["catalogue"])
    if catalog.fingerprint != scope["catalogue_identity"] or catalog.fingerprint != row["catalogue_identity"]:
        raise RuntimeError("archived catalogue identity changed")
    arrays, metadata = simulation.load_case(folder, spec, archived_design)
    if metadata["data_identity"] != scope["data_identity"]:
        raise RuntimeError("cached data identity differs from archive")
    if len(row["fit"]["starts"]) != len(scope["starts"]):
        raise RuntimeError("archived prescribed starts are incomplete")
    # Preparation was performed and recorded before the archived fit. Reuse it
    # exactly; rerunning a projection would change the frozen starting geometry.
    radii = np.asarray(row["fit"]["preparation"]["radii"], dtype=float)
    if row["fit"]["preparation"]["candidate_ids"] != list(catalog.candidate_ids):
        raise RuntimeError("prepared candidate identity changed")
    fit = arrays["fitting"]
    x, y = arrays["x"][fit].copy(), arrays["y"][fit].astype(float)
    return {"id": "observed", "x": x, "y": y, "box": archived_design["simulation"]["box"],
            "centres": catalog.centres.copy(), "radii": radii,
            "candidate_ids": list(catalog.candidate_ids), "starts": scope["starts"],
            "initial_constants": [s["initial_feasibility_projection"]["after"] for s in row["fit"]["starts"]],
            "initial_levels": None, "legacy_starts": row["fit"]["starts"],
            "legacy_configuration": row["fit"]["configuration"],
            "identity": {"data_identity": metadata["data_identity"], "catalogue_identity": catalog.fingerprint,
                         "fitting_identity": protocol.array_hash(x, y),
                         "prepared_geometry_identity": protocol.array_hash(catalog.centres, radii),
                         "report_sha256": scope["report_sha256"], "preparation_sha256": scope["preparation_sha256"]}}


def make_control(design):
    """An expected-likelihood numerical control, not an inferential experiment."""
    control, physics = design["control"], design["physics"]
    centres, radii = np.asarray(control["centres"]), np.asarray(control["radii"])
    ids = [f"control-{i}" for i in range(len(radii))]
    truth = fitter.ConstrainedDiffusionModel(centres, radii, **control["truth"],
                  levels=control["levels"], box=control["box"], width=physics["width"],
                  knots=physics["knots"], min_gap=physics["min_gap"],
                  radius_bounds=physics["radius_bounds"], candidate_ids=ids)
    truth.guard_geometry()
    x = np.random.default_rng(control["seed"]).uniform(0, control["box"], (control["count"], 3))
    y = truth.raw_predict(x)
    if not np.isfinite(y).all() or np.any((y <= 0) | (y >= 1)):
        raise RuntimeError("invalid expected-likelihood control probabilities")
    return {"id": "control", "x": x, "y": y, "box": control["box"],
            "centres": centres, "radii": radii, "candidate_ids": ids,
            "starts": [[control["initial"]["ell"], control["initial"]["xi"]]],
            "initial_constants": [control["initial"]], "initial_levels": control["initial_levels"],
            "truth_entropy": float(-np.mean(y * np.log(y) + (1 - y) * np.log1p(-y))),
            "identity": {"configuration": deepcopy(control), "fitting_identity": protocol.array_hash(x, y),
                         "prepared_geometry_identity": protocol.array_hash(centres, radii)}}


def initial_states(case, design):
    physics = design["physics"]
    states = []
    for constants in case["initial_constants"]:
        model = fitter.ConstrainedDiffusionModel(case["centres"], case["radii"], **constants,
                      levels=case["initial_levels"], candidate_ids=case["candidate_ids"], box=case["box"],
                      width=physics["width"], knots=physics["knots"], min_gap=physics["min_gap"],
                      radius_bounds=physics["radius_bounds"])
        model.guard_geometry()
        states.append(model.export_state())
    return states


def make_provenance(design):
    # Reuse the existing environment inventory, while retaining this study's own
    # design and identity rather than claiming compatibility with an E21 run.
    environment = protocol.make_provenance(protocol.load_design())["runtime"]
    return {"schema": "e21-solver-diagnosis-provenance-v1", "design": deepcopy(design),
            "design_id": protocol.digest(design), "sources": protocol.source_hashes(), "runtime": environment}


def expected_keys(design, case_names=("control", "observed")):
    return [(name, stage["id"], i) for name in case_names for stage in design["stages"]
            if stage["parent"] is None or name in design["warm_scope"]
            for i in range(1 if name == "control" else len(design["observed"]["starts"]))]


def row_key(row):
    return row["case"], row["stage"], row["start"]


def artifact_reference(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def verify_cached_fit(row, folder):
    if (row.get("fit_artifact") is not None or row.get("last_accepted_state") is not None) and (
            not isinstance(row.get("fit_sha256"), str) or len(row["fit_sha256"]) != 64):
        raise RuntimeError("cached diagnostic fit checksum is missing")
    if "fit_sha256" in row:
        path = Path(folder) / "fit.json"
        if file_hash(path) != row["fit_sha256"]:
            raise RuntimeError("cached diagnostic fit artifact changed")
        fit = json.loads(path.read_text())
        if (len(fit.get("starts", [])) != 1 or row.get("last_accepted_state")
                != fit["starts"][0].get("last_accepted_state")):
            raise RuntimeError("cached terminal state differs from the verified fit artifact")


def load_report(path, provenance):
    fingerprint = protocol.digest(provenance)
    if not Path(path).exists():
        return {"schema": "e21-solver-diagnosis-report-v1", "fingerprint": fingerprint,
                "provenance": provenance, "rows": [], "promotion": False, "inference_validated": False}
    report = json.loads(Path(path).read_text())
    if report.get("provenance") != provenance or report.get("fingerprint") != fingerprint:
        raise RuntimeError("diagnostic provenance changed; use a new version and output directory")
    keys = [row_key(row) for row in report["rows"]]
    if len(keys) != len(set(keys)) or any(key not in expected_keys(provenance["design"]) for key in keys):
        raise RuntimeError("malformed or duplicate diagnostic attempt identity")
    return report


def resolve_start(stage, start, cold_states, rows, case_id):
    """No best-start selection: each chain inherits its own parent's full state."""
    if stage["parent"] is None:
        return cold_states[start], None
    matches = [r for r in rows if row_key(r) == (case_id, stage["parent"], start)]
    if len(matches) != 1:
        return None, None
    parent = matches[0]
    return parent.get("last_accepted_state"), parent


def run_stage(case, stage, start, state, parent, design, report, artifacts):
    key = (case["id"], stage["id"], start)
    common = {"case": key[0], "stage": key[1], "start": start, "attempt": 1,
              "execution_fingerprint": report["fingerprint"], "data_identity": case["identity"],
              "initial_state_identity": protocol.digest(state),
              "parent": None if parent is None else {"key": list(row_key(parent)),
                  "state_identity": protocol.digest(parent.get("last_accepted_state")),
                  "converged": parent.get("converged", False)},
              "fit_centres": stage["fit_centres"], "fit_radii": stage["fit_radii"],
              "promotion": False, "inference_validated": False}
    existing = [r for r in report["rows"] if row_key(r) == key]
    folder = Path(artifacts) / case["id"] / stage["id"] / f"start_{start:02d}"
    if existing:
        if len(existing) != 1 or any(existing[0].get(k) != v for k, v in common.items()):
            raise RuntimeError("cached diagnostic row has incompatible identity or state")
        verify_cached_fit(existing[0], folder)
        return existing[0]
    if folder.exists():
        result = folder / "result.json"
        previous = json.loads((result if result.exists() else folder / "started.json").read_text())
        if any(previous.get(k) != v for k, v in common.items()):
            raise RuntimeError("orphan diagnostic artifact has incompatible provenance")
        verify_cached_fit(previous, folder)
        if not result.exists():
            previous.update(status="interrupted", converged=False, last_accepted_state=None,
                            error="Canonical attempt interrupted; never silently retried")
            protocol.write_report(result, previous)
        return previous
    folder.mkdir(parents=True, exist_ok=False)
    row = dict(common, status="started", converged=False, last_accepted_state=None)
    protocol.write_report(folder / "started.json", row)
    if state is None:
        row.update(status="dependency_failure", error="parent has no accepted state", fit_seconds=0.0,
                   cumulative_fit_seconds=(parent or {}).get("cumulative_fit_seconds", 0.0), evaluations=0,
                   cumulative_evaluations=(parent or {}).get("cumulative_evaluations", 0))
        protocol.write_report(folder / "result.json", row)
        return row
    physics = design["physics"]
    started = time.perf_counter()
    try:
        before = fitter.restore_constrained_state(state)
        def progress(event):
            compact = {k: v for k, v in event.items() if k in
                       ("event", "round", "iteration", "mean_nll", "projected_gradient",
                        "optimality", "status", "converged", "continuation_decision")}
            print(json.dumps({"case": case["id"], "stage": stage["id"], "start": start, **compact}), flush=True)
        _, fit = fitter.fit_constrained(before.centres().numpy(), before.radii().numpy(), case["x"], case["y"],
                  candidate_ids=case["candidate_ids"], box=case["box"], **physics,
                  starts=[case["starts"][start]], initial_states=[state], prepare_geometry=False,
                  fit_centres=stage["fit_centres"], fit_radii=stage["fit_radii"], diagnose_blocks=True,
                  progress_callback=progress)
        fit_seconds = time.perf_counter() - started
        protocol.write_report(folder / "fit.json", fit)
        attempt = fit["starts"][0]
        row.update(status=attempt["status"], converged=bool(attempt["converged"]),
                   fit_seconds=fit_seconds, fit_artifact=artifact_reference(folder / "fit.json"),
                   fit_sha256=file_hash(folder / "fit.json"),
                   evaluations=len(attempt["evaluations"]),
                   unsafe_trials_rejected=sum(not e["likelihood_evaluated"] and not (
                       e["geometry_feasible"] and e["probability_valid"]) for e in attempt["evaluations"]),
                   invalid_likelihoods=sum(e["likelihood_evaluated"] and not (
                       e["geometry_feasible"] and e["probability_valid"]) for e in attempt["evaluations"]),
                   last_accepted_state=attempt.get("last_accepted_state"),
                   last_accepted_metadata=attempt.get("last_accepted_metadata"),
                   termination_reason=attempt.get("termination_reason"), error=attempt.get("error"),
                   train_loss=attempt.get("train_loss"), summed_nll=attempt.get("summed_nll"),
                   objective_assignment_context=attempt.get("objective_assignment_context"),
                   diagnostics=attempt.get("diagnostics"), block_stationarity=attempt.get("block_stationarity"),
                   assignment_stable=attempt.get("assignment_stable"),
                   rounds=[{k:v for k,v in round_.items() if k != "iteration_history"} for round_ in attempt["rounds"]])
        row["cumulative_evaluations"] = row["evaluations"] + (parent or {}).get("cumulative_evaluations", 0)
        row["cumulative_fit_seconds"] = fit_seconds + (parent or {}).get("cumulative_fit_seconds", 0.0)
        if row["last_accepted_state"] is not None:
            after = fitter.restore_constrained_state(row["last_accepted_state"])
            row["fixed_centres_unchanged"] = (stage["fit_centres"] or np.array_equal(before.centres().numpy(), after.centres().numpy()))
            row["fixed_radii_unchanged"] = (stage["fit_radii"] or np.array_equal(before.radii().numpy(), after.radii().numpy()))
            if not row["fixed_centres_unchanged"] or not row["fixed_radii_unchanged"]:
                raise RuntimeError("frozen coordinates changed")
            if case["id"] == "control":
                row["expected_nll_gap"] = row["train_loss"] - case["truth_entropy"] if row["train_loss"] is not None else None
                row["prediction_rmse"] = float(np.sqrt(np.mean((after.raw_predict(case["x"]) - case["y"]) ** 2)))
            elif start == 0 and (stage["fit_radii"] or stage["fit_centres"]):
                diagnostic_started = time.perf_counter()
                knots = solver_diagnostics.diagnose_radius_knots(after, case["x"], case["y"], **design["endpoint_knots"])
                protocol.write_report(folder / "knots.json", knots)
                row.update(knot_diagnostics=knots, diagnostic_seconds=time.perf_counter() - diagnostic_started)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        row.update(status="diagnostic_error", converged=False, error=f"{type(error).__name__}: {error}",
                   fit_seconds=row.get("fit_seconds", time.perf_counter() - started))
    protocol.write_report(folder / "result.json", row)
    print(json.dumps({"event": "stage_completed", "case": key[0], "stage": key[1], "start": start,
                      "status": row["status"], "converged": row["converged"], "seconds": row["fit_seconds"]}), flush=True)
    return row


def run_diagnosis(design, output, artifacts, case_names=("control", "observed")):
    validate_design(design)
    if any(name not in ("control", "observed") for name in case_names) or len(set(case_names)) != len(case_names):
        raise ValueError("only the control and observed development case 0 are allowed")
    output, artifacts = Path(output).resolve(), Path(artifacts).resolve()
    provenance = make_provenance(design)
    report_path = output / "diagnosis.json"
    report = load_report(report_path, provenance)
    output.mkdir(parents=True, exist_ok=True)
    e21.preserve_sources(output, provenance)
    protocol.write_report(report_path, report)
    for name in case_names:
        case = make_control(design) if name == "control" else load_observed_case(design)
        states = initial_states(case, design)
        protocol.write_report(output / f"{name}_initial_states.json", {"identity": case["identity"], "states": states})
        if name == "observed" and not (output / "archived_v3_knots.json").exists():
            legacy = solver_diagnostics.model_from_fit_state(case["legacy_starts"][0],
                          configuration=case["legacy_configuration"], candidate_ids=case["candidate_ids"])
            knots = solver_diagnostics.diagnose_radius_knots(legacy, case["x"], case["y"], **design["endpoint_knots"])
            protocol.write_report(output / "archived_v3_knots.json", knots)
        for stage in design["stages"]:
            if stage["parent"] is not None and name not in design["warm_scope"]:
                continue
            for start in range(len(states)):
                state, parent = resolve_start(stage, start, states, report["rows"], name)
                row = run_stage(case, stage, start, state, parent, design, report, artifacts)
                if not any(row_key(old) == row_key(row) for old in report["rows"]):
                    report["rows"].append(row)
                protocol.write_report(report_path, report)
    keys = {row_key(row) for row in report["rows"]}
    report["requested_scope_complete"] = set(expected_keys(design, case_names)) <= keys
    report["all_diagnostic_stages_complete"] = keys == set(expected_keys(design))
    report["promotion"] = False
    report["inference_validated"] = False
    protocol.write_report(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "control", "observed"), default="all")
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    design = load_design(args.design)
    torch.set_num_threads(design["runtime"]["threads"])
    cases = ("control", "observed") if args.case == "all" else (args.case,)
    report = run_diagnosis(design, args.output, args.artifacts, cases)
    print(json.dumps({"complete": report["all_diagnostic_stages_complete"], "promotion": False,
                      "rows": len(report["rows"])}), flush=True)
    # Nonconvergence is the measured diagnostic outcome; infrastructure or
    # incomplete/interrupted executions must still return a failing exit code.
    return int(any(r["status"] in ("started", "interrupted", "diagnostic_error") for r in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
