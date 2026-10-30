"""Gate accounting and resume behavior, without fitting models or generating data."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction import stage5_physics_fit as stage


RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "reconstruction" / "results"


def passing_cells(split="development"):
    return [
        {
            "kind": kind, "efficiency": eta, "pattern": index,
            "matrix_excess": {"physics": 0.01, "network": 0.02},
            "physics": {"normalised_error": {"ell": 0.5, "xi": 0.5}},
            "acceptance": {"accepted": kind == "diffusion"},
        }
        for kind, eta, index in sorted(stage.expected_cells(split))
    ]


def assert_blocked(result):
    assert not result["complete"]
    assert not result["passed"]
    assert not result["physics_claim_permitted"]
    assert result["conditions"] == {}


def test_exact_complete_scope_can_pass():
    result = stage.gate(passing_cells(), "development")
    assert result["complete"]
    assert result["passed"]
    assert result["physics_claim_permitted"]
    assert result["cells"] == {"correct": 12, "misspecified": 24}


def test_duplicate_favourable_case_cannot_turn_saved_failure_into_pass():
    cells = json.loads((RESULTS / "stage5_physics_fit_test.json").read_text())["cells"]
    favourable = next(c for c in cells if (c["kind"], c["efficiency"], c["pattern"]) == ("diffusion", 0.37, 51))
    duplicated = cells + [copy.deepcopy(favourable) for _ in range(21)]
    # This duplication previously moved the failed condition below its threshold;
    # the other two conditions already passed and the favourable row preserves them.
    correct = [c for c in duplicated if c["kind"] == "diffusion"]
    assert all(np.median([c["physics"]["normalised_error"][k] for c in correct]) < stage.GATE["normalised_error"]
               for k in ("ell", "xi"))
    result = stage.gate(duplicated, "test")
    assert_blocked(result)
    assert result["verdict"] == "invalid"
    assert result["duplicate_cells"] == 21
    assert result["duplicates"][0]["cell"] == ["diffusion", 0.37, 51]
    assert len(result["duplicates"][0]["record_indices"]) == 22
    assert result["missing_cells"] == 0


@pytest.mark.parametrize("kind", stage.KINDS[1:])
def test_every_prescribed_alternative_field_control_is_required(kind):
    cells = [c for c in passing_cells() if c["kind"] != kind]
    result = stage.gate(cells, "development")
    assert_blocked(result)
    assert result["missing_cells"] == 12
    assert all(key[0] == kind for key in result["missing"])


@pytest.mark.parametrize("identity_change", [{"pattern": 999}, {"efficiency": 0.5}, {"kind": "unknown"}])
def test_unexpected_favourable_identity_cannot_enter_the_gate(identity_change):
    cells = passing_cells()
    extra = copy.deepcopy(cells[0])
    extra.update(identity_change)
    result = stage.gate(cells + [extra], "development")
    assert_blocked(result)
    assert result["unexpected_cells"] == 1
    assert result["unexpected"][0]["cell"] == [extra["kind"], extra["efficiency"], extra["pattern"]]
    assert result["missing_cells"] == 0


def test_failed_case_remains_explicit_and_cannot_be_excluded():
    cells = passing_cells()
    key = {k: cells[0][k] for k in ("kind", "efficiency", "pattern")}
    cells[0] = {**key, "error": "RuntimeError: fit failed"}
    result = stage.gate(cells, "development")
    assert_blocked(result)
    assert result["errored_cells"] == 1
    assert result["errors"][0]["error"] == "RuntimeError: fit failed"
    assert result["missing_cells"] == 1


def test_successful_duplicate_does_not_hide_an_error_for_the_same_case():
    cells = passing_cells()
    failure = {k: cells[0][k] for k in ("kind", "efficiency", "pattern")}
    failure["error"] = "first attempt failed"
    result = stage.gate(cells + [failure], "development")
    assert_blocked(result)
    assert result["errored_cells"] == 1
    assert result["duplicate_cells"] == 1


@pytest.mark.parametrize("replacement", [
    None,
    {},
    {"kind": ["diffusion"], "efficiency": 0.1, "pattern": 0},
    {"kind": "diffusion", "efficiency": "0.1", "pattern": 0},
    {"kind": "diffusion", "efficiency": np.nan, "pattern": 0},
    {"kind": "diffusion", "efficiency": 0.1, "pattern": False},
])
def test_malformed_case_identities_fail_closed(replacement):
    cells = passing_cells()
    cells[0] = replacement
    result = stage.gate(cells, "development")
    assert_blocked(result)
    assert result["malformed_cells"] == 1
    assert result["malformed"][0]["record_index"] == 0


@pytest.mark.parametrize("change", [
    {"acceptance": {"accepted": "false"}},
    {"acceptance": {}},
    {"matrix_excess": {"physics": np.nan, "network": 0.02}},
    {"matrix_excess": {"physics": 0.01}},
    {"physics": {"normalised_error": {"ell": -1, "xi": 0.5}}},
    {"physics": {"normalised_error": {"ell": 0.5, "xi": np.inf}}},
])
def test_malformed_gate_metrics_fail_closed(change):
    cells = passing_cells()
    cells[0].update(change)
    result = stage.gate(cells, "development")
    assert_blocked(result)
    assert result["malformed_cells"] == 1


@pytest.mark.parametrize("physics", [
    {},
    {"constants": {"c_eq": 0.02, "ell": 2.0, "xi": 5.0, "c_inf": 0.1}},
    {"constants": None},
])
def test_missing_errors_cannot_hide_an_accepted_or_unspecified_fit(physics):
    cells = passing_cells()
    cells[0]["physics"] = physics
    result = stage.gate(cells, "development")
    assert_blocked(result)
    assert result["malformed_cells"] == 1
    assert "explicit constants=None and accepted=False" in result["malformed"][0]["reason"]


def test_completed_cells_without_fits_are_not_dropped_from_parameter_error():
    cells = passing_cells()
    for cell in cells:
        if cell["kind"] == "diffusion":
            cell["physics"] = {"constants": None}
            cell["acceptance"]["accepted"] = False
    result = stage.gate(cells, "development")
    assert result["complete"]
    assert not result["passed"]
    assert result["cells"]["correct"] == 12
    assert result["conditions"]["ii_median_normalised_error"]["measured"] == {"ell": np.inf, "xi": np.inf}


@pytest.mark.parametrize("split,suffix", [
    ("development", ""), ("test", ""), ("development", "_first_run"), ("test", "_first_run"),
])
def test_saved_reports_keep_their_failed_verdicts_and_statistics(split, suffix):
    report = json.loads((RESULTS / f"stage5_physics_fit_{split}{suffix}.json").read_text())
    result = stage.gate(report["cells"], split)
    assert result["complete"]
    assert result["verdict"] == "failed"
    assert not result["passed"]
    assert result["conditions"] == report["gate"]["conditions"]
    assert result["rejected_by_kind"] == report["gate"]["rejected_by_kind"]
    assert result["physics_claim_permitted"] == report["gate"]["physics_claim_permitted"]


def test_runner_keeps_failures_on_resume_and_recomputes_a_tampered_gate(monkeypatch, tmp_path, capsys):
    output = tmp_path / "failed_attempt.json"
    attempts = []

    class Pool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, worker, tasks):
            for task in tasks:
                attempts.append(task)
                index, eta, kind, *_ = task
                yield {"pattern": index, "efficiency": eta, "kind": kind, "error": "recorded failure"}

    monkeypatch.setattr(stage, "SCOPE", {"development": {0.37: (0,)}})
    monkeypatch.setattr(stage.multiprocessing, "get_context", lambda method: SimpleNamespace(Pool=Pool))
    monkeypatch.setattr("sys.argv", ["stage5", "--split", "development", "--device", "cpu",
                                    "--workers", "1", "--output", str(output)])
    stage.main()
    original = json.loads(output.read_text())
    assert len(attempts) == 3
    assert original["gate"]["errored_cells"] == 3
    original["gate"] = {"passed": True, "physics_claim_permitted": True}
    output.write_text(json.dumps(original))
    stage.main()
    resumed = json.loads(output.read_text())
    assert len(attempts) == 3  # Failed attempts are not silently retried or discarded.
    assert resumed["cells"] == original["cells"]
    assert resumed["gate"]["errored_cells"] == 3
    assert_blocked(resumed["gate"])
    assert "fresh --output" in capsys.readouterr().out


@pytest.mark.parametrize("kinds", ["diffusion,diffusion", "diffusion,unknown"])
def test_runner_rejects_duplicate_or_unknown_requested_kinds(monkeypatch, kinds):
    monkeypatch.setattr("sys.argv", ["stage5", "--split", "development", "--kinds", kinds])
    with pytest.raises(SystemExit, match="2"):
        stage.main()


def test_runner_rejects_invalid_saved_records_without_rewriting_them(monkeypatch, tmp_path):
    output = tmp_path / "duplicate.json"
    cells = passing_cells()
    payload = json.dumps({"cells": cells + [cells[0]], "gate": {"passed": True}})
    output.write_text(payload)
    monkeypatch.setattr("sys.argv", ["stage5", "--split", "development", "--output", str(output)])
    with pytest.raises(SystemExit, match='Diagnostics:.*"duplicates"'):
        stage.main()
    assert output.read_text() == payload
