"""Verify the completed diagnostic archive without rerunning any fit."""

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    script = Path(__file__).resolve()
    folder, root = script.parent, script.parents[4]
    report_path = folder / "diagnosis.json"
    report = json.loads(report_path.read_text())
    assert report["all_diagnostic_stages_complete"] and report["requested_scope_complete"]
    assert report["promotion"] is False and report["inference_validated"] is False
    assert report["fingerprint"] == digest(report["provenance"])
    stages = ("chemistry", "radii", "centres", "joint", "warm_radii", "warm_joint")
    expected = {(case, stage, start) for case in ("control", "observed")
                for stage in (stages[:4] if case == "control" else stages)
                for start in range(1 if case == "control" else 3)}
    rows = {(r["case"], r["stage"], r["start"]): r for r in report["rows"]}
    assert len(report["rows"]) == len(rows) == 22 and set(rows) == expected
    snapshots = list((folder / "source_snapshot").iterdir())
    assert len(snapshots) == 1
    for name, sha in report["provenance"]["sources"].items():
        assert file_hash(root / name) == sha, f"Working source changed: {name}"
        assert file_hash(snapshots[0] / name) == sha, f"Archived source changed: {name}"
    cold = {case: json.loads((folder / f"{case}_initial_states.json").read_text())["states"]
            for case in ("control", "observed")}
    observed = report["provenance"]["design"]["observed"]
    old_path = root / observed["report"]
    assert file_hash(old_path) == observed["report_sha256"]
    old = json.loads(old_path.read_text())["rows"][0]["fit"]["starts"]
    checks = []
    for key, row in rows.items():
        assert row["status"] in ("converged", "not_converged", "evaluation_budget")
        assert row["invalid_likelihoods"] == 0
        path = root / row["fit_artifact"]
        assert file_hash(path) == row["fit_sha256"], key
        fit = json.loads(path.read_text())
        assert len(fit["starts"]) == 1
        attempt = fit["starts"][0]
        state = row["last_accepted_state"]
        assert state is not None and state == attempt["last_accepted_state"]
        assert row["status"] == attempt["status"] and row["converged"] == fit["converged"]
        if not row["converged"]:
            assert fit["selected_start"] is None and fit["selected_state"] is None
        parent_stage = {"warm_radii": "chemistry", "warm_joint": "warm_radii"}.get(key[1])
        initial = cold[key[0]][key[2]]
        if parent_stage:
            parent_key = (key[0], parent_stage, key[2])
            parent = rows[parent_key]
            initial = parent["last_accepted_state"]
            assert row["parent"]["key"] == list(parent_key)
            assert row["parent"]["state_identity"] == digest(initial)
            assert row["parent"]["converged"] == parent["converged"]
            assert row["cumulative_evaluations"] == row["evaluations"] + parent["cumulative_evaluations"]
        else:
            assert row["parent"] is None and row["cumulative_evaluations"] == row["evaluations"]
        assert row["initial_state_identity"] == digest(initial)
        assert attempt["initial_state"] == initial == attempt["optimization_initial_state"]
        count = initial["configuration"]["count"]
        if not row["fit_centres"]:
            assert initial["vector"][:3 * count] == state["vector"][:3 * count]
        if not row["fit_radii"]:
            assert initial["vector"][3 * count:4 * count] == state["vector"][3 * count:4 * count]
        identical_to_v3 = None
        if key[:2] == ("observed", "joint"):
            identical_to_v3 = all(attempt[name] == old[key[2]][name] for name in
                                  ("centres", "radii", "constants", "shape", "train_loss",
                                   "summed_nll", "status", "assignment_stable"))
            assert identical_to_v3
        checks.append({"key": list(key), "fit_sha256": row["fit_sha256"],
                       "state_and_lineage_verified": True, "frozen_coordinates_verified": True,
                       "joint_endpoint_identical_to_archived_v3": identical_to_v3})
    summary = {"schema": "e21-diagnostic-archive-verification-v1", "passed": True,
               "scope": "Artifact integrity and experiment accounting only; no scientific-readiness claim",
               "promotion": False, "inference_validated": False, "attempts_verified": len(checks),
               "source_files_verified_in_working_tree_and_snapshot": len(report["provenance"]["sources"]),
               "report_sha256": file_hash(report_path), "script_sha256": file_hash(script),
               "checks": checks}
    (folder / "archive_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}))


if __name__ == "__main__":
    main()
