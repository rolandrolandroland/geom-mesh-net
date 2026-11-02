"""Selecting numerical diagnostics must not reduce the four-case readiness scope."""

from copy import deepcopy
import json

import pytest

from experiments.reconstruction import e21_geometry as runner
from experiments.reconstruction import e21_protocol as protocol


@pytest.fixture
def design():
    return protocol.load_design()


def provenance(design):
    return {"design": deepcopy(design), "design_id": protocol.digest(design),
            "sources": {}, "runtime": deepcopy(design["runtime"]), "inputs": {}}


def row(design, index, *, arm="B", attempt=1, converged=True):
    spec = protocol.case_spec("development", index, design)
    return {**spec, "arm": arm, "attempt": attempt, "status": "ok", "elapsed_seconds": 1.,
            "data_identity": "data:" + spec["case_id"],
            "catalogue_identity": "catalogue:" + spec["case_id"],
            "metrics": {"feasible": True, "probability_valid": True, "converged": converged,
                        "objective_contract_passed": True, "search_bound_contact": False,
                        "ambiguous": False, "radius_mare": .1, "ell_absolute_relative_error": .1,
                        "xi_absolute_relative_error": .1, "full_heldback_log_loss": .2,
                        "matrix_heldback_log_loss": .2}}


def report(design, indices):
    result = protocol.new_report("development", design, provenance(design))
    for index in indices:
        protocol.record_attempt(result, row(design, index))
    return result


def configure_cli(monkeypatch, tmp_path, design):
    monkeypatch.setattr(protocol, "load_design", lambda path=None: deepcopy(design))
    monkeypatch.setattr(protocol, "make_provenance", lambda *args, **kwargs: provenance(design))
    monkeypatch.setattr(runner, "preserve_sources", lambda *args, **kwargs: None)
    called = []
    def run_case(index, phase, settings, result, output, artifacts, preflight=False):
        assert phase == "development"
        called.append((index, preflight))
        if not any(item["arm"] == "B" and item["case_index"] == index for item in result["rows"]):
            protocol.record_attempt(result, row(settings, index))
        return True
    monkeypatch.setattr(runner, "run_case", run_case)
    common = ["--output", str(tmp_path / "reports"), "--artifacts", str(tmp_path / "artifacts")]
    return called, common


def test_successful_subset_stays_unready_until_all_four_canonical_cases_exist(design, monkeypatch, tmp_path):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    assert runner.main(common + ["--preflight", "--development-cases", "2"]) == 0
    saved = json.loads((tmp_path / "reports" / "development.json").read_text())
    assert called == [(2, True)]
    assert saved["preflight_ready"] is False
    assert saved["preflight_readiness"]["present_cases"] == 1
    assert saved["preflight_readiness"]["expected_cases"] == 4
    assert not saved["gate"]["passed"]
    assert runner.main(common + ["--preflight", "--development-cases", "0", "1", "3"]) == 0
    saved = json.loads((tmp_path / "reports" / "development.json").read_text())
    assert called == [(2, True), (0, True), (1, True), (3, True)]
    assert saved["preflight_ready"] is True
    assert saved["preflight_readiness"]["present_cases"] == 4
    assert not saved["smoke_gate"]["passed"]  # B readiness is not four-arm readiness.


@pytest.mark.parametrize("arguments", [
    ["--development-cases", "0"],
    ["--preflight", "--development-cases", "0", "0"],
    ["--preflight", "--development-cases", "4"],
    ["--preflight", "--development-cases", "-1"],
    ["--preflight", "--development-cases", "0", "--all-development"],
    ["--preflight", "--development-cases", "0", "--phase", "confirmation"],
    ["--preflight", "--development-cases", "0", "--phase", "test"],
])
def test_subset_arguments_cannot_change_phase_or_scope(design, monkeypatch, tmp_path, arguments):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    with pytest.raises(ValueError):
        runner.main(common + arguments)
    assert called == []
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("values", [[], ["not-an-integer"]])
def test_subset_parser_requires_nonempty_integer_list(design, monkeypatch, tmp_path, values):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    with pytest.raises(SystemExit) as error:
        runner.main(common + ["--preflight", "--development-cases", *values])
    assert error.value.code == 2
    assert called == []


def test_default_preflight_still_requests_every_smoke_case(design, monkeypatch, tmp_path):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    assert runner.main(common + ["--preflight"]) == 0
    assert called == [(index, True) for index in design["smoke_cases"]]
    saved = json.loads((tmp_path / "reports" / "development.json").read_text())
    assert saved["preflight_ready"] is True


def test_normal_smoke_recomputes_exact_coverage_instead_of_trusting_saved_readiness(design, monkeypatch, tmp_path):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    partial = report(design, design["smoke_cases"][:3])
    partial["preflight_ready"] = True
    protocol.write_report(tmp_path / "reports" / "development.json", partial)
    with pytest.raises(RuntimeError, match="all four canonical B"):
        runner.main(common)
    assert called == []


def test_normal_smoke_and_preflight_use_the_same_readiness_helper(design, monkeypatch, tmp_path):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    complete = report(design, design["smoke_cases"])
    complete["preflight_ready"] = False
    protocol.write_report(tmp_path / "reports" / "development.json", complete)
    assert runner.preflight_readiness(complete, design)["ready"] is True
    assert runner.main(common) == 0
    assert called == [(index, False) for index in design["smoke_cases"]]
    saved = json.loads((tmp_path / "reports" / "development.json").read_text())
    assert saved["preflight_ready"] is True


@pytest.mark.parametrize("defect", ["duplicate", "malformed", "wrong_case_id", "wrong_index",
                                   "wrong_phase", "invalid_arm", "invalid_attempt", "missing_metric",
                                   "malformed_other_arm", "mismatched_shared_catalogue"])
def test_any_malformed_existing_row_blocks_readiness_and_execution(design, monkeypatch, tmp_path, defect):
    called, common = configure_cli(monkeypatch, tmp_path, design)
    result = report(design, design["smoke_cases"])
    target = result["rows"][0]
    if defect == "duplicate":
        result["rows"].append(deepcopy(target))
    elif defect == "malformed":
        result["rows"].append(None)
    elif defect == "wrong_case_id":
        target["case_id"] = "unexpected"
    elif defect == "wrong_index":
        target["case_index"] = 3
    elif defect == "wrong_phase":
        target["phase"] = "test"
    elif defect == "invalid_arm":
        target["arm"] = "other"
    elif defect == "invalid_attempt":
        target["attempt"] = True
    elif defect == "missing_metric":
        del target["metrics"]["xi_absolute_relative_error"]
    elif defect == "malformed_other_arm":
        additional = row(design, 5, arm="A")
        additional["metrics"]["converged"] = "yes"
        result["rows"].append(additional)
    elif defect == "mismatched_shared_catalogue":
        additional = row(design, 0, arm="A")
        additional["catalogue_identity"] = "different-catalogue"
        result["rows"].append(additional)
    decision = runner.preflight_readiness(result, design)
    assert not decision["ready"] and decision["accounting_errors"]
    protocol.write_report(tmp_path / "reports" / "development.json", result)
    with pytest.raises(RuntimeError, match="invalid development report"):
        runner.main(common + ["--preflight", "--development-cases", "1"])
    assert called == []


def test_successful_retry_cannot_replace_a_failed_primary_B_case(design):
    result = report(design, design["smoke_cases"])
    result["rows"][0].update(status="error", error="original numerical failure")
    protocol.record_attempt(result, row(design, 0, attempt=2))
    decision = runner.preflight_readiness(result, design)
    assert decision["present_cases"] == 4
    assert decision["numerically_ready_cases"] == 3
    assert not decision["ready"]


def test_valid_later_development_records_do_not_replace_a_missing_smoke_case(design):
    result = report(design, [0, 1, 2, 4])
    decision = runner.preflight_readiness(result, design)
    assert not decision["ready"] and decision["present_cases"] == 3


def test_valid_additional_records_do_not_invalidate_completed_preflight(design):
    result = report(design, [*design["smoke_cases"], 4])
    protocol.record_attempt(result, row(design, 0, arm="C"))
    assert runner.preflight_readiness(result, design)["ready"]
