#!/usr/bin/env python3
"""Summarize completed frozen diagnostic JSON traces without loading the fitter.

Standard library only; no fitting, gradients, simulation, or source mutations.
Rerunning replaces trace_summary.json with a fresh, input-hashed snapshot.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), digest(data)


def finite(values):
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def distribution(values):
    values = finite(values)
    return {"count": len(values), "minimum": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None,
            "sum": sum(values) if values else None}


def state_geometry(state):
    if state is None:
        return None
    count = state["configuration"]["count"]
    box = state["configuration"]["box"]
    min_gap = state["configuration"]["min_gap"]
    vector = state["vector"]
    centres = [[5 * vector[3 * i + axis] for axis in range(3)] for i in range(count)]
    radii = [5 * value for value in vector[3 * count:4 * count]]
    contacts, clearances = [], []
    for i, (centre, radius) in enumerate(zip(centres, radii)):
        for axis, coordinate in enumerate(centre):
            for side, clearance in (("lower", coordinate - radius), ("upper", box - coordinate - radius)):
                clearances.append(clearance)
                if clearance <= 1e-8:
                    contacts.append({"candidate_id": state["candidate_ids"][i], "axis": axis,
                                     "side": side, "clearance": clearance})
    gaps = [math.dist(centres[i], centres[j]) - radii[i] - radii[j]
            for i in range(count) for j in range(i + 1, count)]
    return {"minimum_box_clearance": min(clearances),
            "minimum_surface_gap": min(gaps) if gaps else None,
            "box_contacts_within_1e_minus_8": contacts,
            "pair_contacts_within_1e_minus_8_of_required_gap": sum(gap <= min_gap + 1e-8 for gap in gaps),
            "radius_range": [min(radii), max(radii)]}


def knot_summary(path):
    if not path.exists():
        return None
    report, sha = read_json(path)
    crossings = []
    for particle in report.get("particles", []):
        for crossing in particle.get("crossings", []):
            modes = {}
            for name, mode in crossing.get("assignment_modes", {}).items():
                modes[name] = {key: mode.get(key) for key in
                               ("verified", "resolved_jump", "coordinate_cusp_minimum", "nearby_coordinate_cusp",
                                "left_derivative", "right_derivative", "derivative_jump")}
            crossings.append({"candidate_id": particle["candidate_id"],
                              "distance_from_endpoint": crossing["distance_from_endpoint"],
                              "near_endpoint": crossing["near_endpoint"], "assignment_modes": modes})
    return {"sha256": sha, "scope": report.get("scope"), "settings": report.get("settings"),
            "mean_nll_actual_assignments": report.get("mean_nll"),
            "full_projected_gradient_inf": report.get("full_projected_gradient_inf"),
            "joint_convergence_assessed": report.get("joint_convergence_assessed"),
            "diagnostic_evaluations": report.get("diagnostic_evaluations"),
            "crossings": crossings,
            "caveat": "These are radius-coordinate sections even for centre-only fits; they do not establish a centre-coordinate cusp or joint stationarity."}


def summarize(row, root):
    fit_path = root / row["fit_artifact"]
    fit, fit_sha = read_json(fit_path)
    if fit_sha != row["fit_sha256"]:
        raise ValueError(f"fit SHA mismatch: {fit_path}")
    attempt = fit["starts"][0]
    evaluations = attempt["evaluations"]
    rounds = attempt["rounds"]
    history = [iteration for round_ in rounds for iteration in round_.get("iteration_history", [])]
    accepted_history = [item for item in history if item.get("accepted")]
    metadata = attempt.get("last_accepted_metadata") or {}
    completed_steps = sum(round_["iterations"] for round_ in rounds)
    partial_steps = (metadata.get("iteration", 0) or 0) if metadata.get("origin") != "actual_assignment_refresh" else 0
    accepted_steps = completed_steps + partial_steps
    refused = [e for e in evaluations if not e["likelihood_evaluated"]]
    geometry_refused = [e for e in evaluations if not e["geometry_feasible"]]
    probability_refused = [e for e in evaluations if e["geometry_feasible"] and not e["probability_valid"]]
    box_refused = [e for e in geometry_refused if e["minimum_box_clearance"] < -1e-10]
    pair_refused = [e for e in geometry_refused if e["minimum_surface_gap"] is not None
                    and e["minimum_surface_gap"] < fit["configuration"]["min_gap"] - 1e-10]
    runs, run = [], 0
    for evaluation in evaluations:
        if not evaluation["likelihood_evaluated"]:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    round_summaries = []
    for round_ in rounds:
        items = round_.get("iteration_history", [])
        first = items[0]["mean_nll"] if items else None
        last = items[-1]["mean_nll"] if items else None
        round_summaries.append({
            **{key: round_.get(key) for key in
               ("round", "iterations", "termination", "continuation_decision", "optimality",
                "optimizer_optimality", "reassigned_stationarity", "reassigned_complementarity",
                "assignment_changes", "assignment_objective_change", "assignment_gradient_change",
                "assignment_stable", "train_loss", "summed_nll", "block_stationarity")},
            "initial_actual_assignment_mean_nll": first,
            "terminal_frozen_assignment_mean_nll": last,
            "frozen_assignment_loss_reduction": None if first is None or last is None else first - last,
            "actual_assignment_loss_reduction": None if first is None else first - round_["train_loss"]})
    reductions = [item for item in accepted_history if item.get("step_size", 1) < 1]
    step_blocks = {}
    for block in ("centres", "radii", "chemistry", "interior_profile"):
        step_blocks[block] = {
            "accepted_step_infinity": distribution(item.get("accepted_step_blocks", {}).get(block, {}).get("infinity") for item in accepted_history),
            "gradient_change_infinity": distribution(item.get("gradient_change_blocks", {}).get(block, {}).get("infinity") for item in accepted_history)}
    first_loss = next((e["mean_nll"] for e in evaluations if e["likelihood_evaluated"]), None)
    return {
        "case": row["case"], "stage": row["stage"], "start": row["start"],
        "fit_artifact": row["fit_artifact"], "fit_sha256": fit_sha,
        "status": attempt["status"], "converged": attempt["converged"],
        "termination_reason": attempt.get("termination_reason"), "error": attempt.get("error"),
        "seconds": fit["seconds"], "evaluation_count": len(evaluations),
        "likelihood_evaluations": sum(e["likelihood_evaluated"] for e in evaluations),
        "accepted_steps_reconstructed": accepted_steps,
        "accepted_steps_completed_rounds": completed_steps, "accepted_steps_partial_round": partial_steps,
        "accepted_steps_with_saved_iteration_metrics": len(accepted_history),
        "iteration_metrics_complete": len(accepted_history) == accepted_steps,
        "history_caveat": "An exception inside a round retains its accepted checkpoint and evaluation trace, but not that unfinished round's iteration history. Partial accepted-step counts use the checkpoint iteration index.",
        "unsafe_refusals": {"total_before_likelihood": len(refused), "geometry": len(geometry_refused),
                            "box": len(box_refused), "pair_gap": len(pair_refused), "raw_probability": len(probability_refused),
                            "errors": dict(Counter(e.get("error", "unspecified") for e in refused)),
                            "invalid_likelihoods": sum(e["likelihood_evaluated"] and not (e["geometry_feasible"] and e["probability_valid"]) for e in evaluations),
                            "box_clearance_on_box_refusals": distribution(e["minimum_box_clearance"] for e in box_refused),
                            "gap_on_pair_refusals": distribution(e["minimum_surface_gap"] for e in pair_refused),
                            "box_refusals_beyond_physical_magnitude": {str(threshold): sum(e["minimum_box_clearance"] < -threshold for e in box_refused)
                                                                     for threshold in (1e-10, 2e-10, 1e-9, 1e-8)},
                            "consecutive_refusal_run_counts": dict(Counter(runs))},
        "completed_round_step_metrics": {
            "full_steps": sum(item.get("step_size") == 1 for item in accepted_history),
            "reduced_steps": len(reductions), "step_sizes": distribution(item.get("step_size") for item in accepted_history),
            "backtracks": distribution(item.get("backtracks") for item in accepted_history),
            "direction_counts": dict(Counter(item.get("direction", "unspecified") for item in accepted_history)),
            "trust_cap_applied": sum(item.get("trust_cap_divisor", 1) > 1 for item in accepted_history),
            "trust_cap_divisors": distribution(item.get("trust_cap_divisor") for item in accepted_history),
            "desired_infinity_norms": distribution(item.get("desired_infinity_norm") for item in accepted_history),
            "secant_memory_updates": sum(bool(item.get("memory_updated")) for item in accepted_history),
            "secant_curvature": distribution(item.get("secant_curvature") for item in accepted_history),
            "projection_primal_violation": distribution(item.get("direction_projection", {}).get("primal_violation") for item in accepted_history),
            "block_steps_and_gradient_changes": step_blocks},
        "first_evaluated_mean_nll": first_loss, "terminal_reported_mean_nll": attempt.get("train_loss"),
        "terminal_objective_assignment_context": attempt.get("objective_assignment_context"),
        "reported_loss_reduction": None if first_loss is None or attempt.get("train_loss") is None else first_loss - attempt["train_loss"],
        "last_accepted_metadata": metadata, "endpoint_geometry": state_geometry(attempt.get("last_accepted_state")),
        "bound_contacts_reported_by_fitter": attempt.get("at_bounds"),
        "endpoint_block_stationarity": attempt.get("block_stationarity"),
        "rounds": round_summaries, "radius_knot_diagnostics": knot_summary(fit_path.parent / "knots.json")}


def main():
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=script.parent / "diagnosis.json")
    parser.add_argument("--output", type=Path, default=script.parent / "trace_summary.json")
    args = parser.parse_args()
    root = script.parents[4]
    report, report_sha = read_json(args.report)
    rows = [summarize(row, root) for row in report["rows"] if row.get("fit_artifact")]
    missing = [{key: row.get(key) for key in ("case", "stage", "start", "status", "error")}
               for row in report["rows"] if not row.get("fit_artifact")]
    groups = {}
    for row in rows:
        key = f'{row["case"]}/{row["stage"]}'
        group = groups.setdefault(key, {"attempts": 0, "converged": 0, "evaluations": 0,
                                       "likelihoods": 0, "geometry_refusals": 0, "probability_refusals": 0,
                                       "invalid_likelihoods": 0, "statuses": Counter()})
        group["attempts"] += 1
        group["converged"] += row["converged"]
        group["evaluations"] += row["evaluation_count"]
        group["likelihoods"] += row["likelihood_evaluations"]
        group["geometry_refusals"] += row["unsafe_refusals"]["geometry"]
        group["probability_refusals"] += row["unsafe_refusals"]["raw_probability"]
        group["invalid_likelihoods"] += row["unsafe_refusals"]["invalid_likelihoods"]
        group["statuses"][row["status"]] += 1
    summary = {"schema": "e21-solver-trace-summary-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Cheap descriptive analysis of completed canonical diagnostic JSON traces; no new fitting or model evaluation.",
               "readiness": False, "promotion": False, "inference_validated": False,
               "all_diagnostic_stages_complete": bool(report.get("all_diagnostic_stages_complete", False)),
               "expected_attempts": 22, "completed_report_rows": len(report["rows"]),
               "inputs": {"report": str(args.report.resolve()), "report_sha256": report_sha,
                          "execution_fingerprint": report["fingerprint"],
                          "script": str(script), "script_sha256": digest(script.read_bytes())},
               "tolerance_mechanism": {
                   "physical_domain_tolerance": 1e-10, "normalized_projection_primal_tolerance": 1e-9,
                   "centre_only_containment_row_norm": 5.0, "centre_only_pair_row_norm": 5 * math.sqrt(2),
                   "joint_containment_row_norm": 5 * math.sqrt(2), "joint_pair_row_norm": 10.0,
                   "description": "Row normalization means the QP may certify a tiny violation larger than the physical-domain tolerance. Clipping admissible negative current slack to zero hides the available inward-restoration requirement. Backtracking remains fail-closed but may consume the budget at a boundary.",
                   "caveat": "QP tolerance is an allowed bound, not proof every refused trial reaches that bound; read actual certificates and physical trace clearances."},
               "groups": groups, "attempts": rows, "rows_without_fit_artifacts": missing,
               "caveats": ["Results from the same reused development case are descriptive diagnostics, not independent confirmation.",
                           "Terminal frozen-assignment losses are labelled and are not asserted to be the current actual-assignment objective.",
                           "Radius-knot sections cannot establish centre-coordinate cusps or joint nonsmooth stationarity.",
                           "Missing unfinished-round history prevents complete step-cap/secant/backtrack attribution for budget exceptions."]}
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(args.output), "rows": len(rows), "complete": summary["all_diagnostic_stages_complete"],
                      "report_sha256": report_sha, "script_sha256": summary["inputs"]["script_sha256"], "groups": groups}, sort_keys=True))


if __name__ == "__main__":
    main()
