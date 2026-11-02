"""E21 decisions fail closed without reading data or running a physical fit."""

from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.reconstruction import e21_protocol as protocol


@pytest.fixture
def design():
    return protocol.load_design()


def provenance(design):
    return {"design": deepcopy(design), "design_id": protocol.digest(design),
            "sources": {"example.py": "source-hash"}, "runtime": {"fit_device": "cpu"}}


def row(design, phase, index, arm):
    spec = protocol.case_spec(phase, index, design)
    metrics = {"feasible": True, "probability_valid": True, "converged": True,
               "search_bound_contact": False, "ambiguous": False,
               "objective_contract_passed": True, "radius_mare": 0.2,
               "ell_absolute_relative_error": 0.3, "xi_absolute_relative_error": 0.4,
               "full_heldback_log_loss": 0.25, "matrix_heldback_log_loss": 0.2}
    if arm == "D":
        metrics.update(radius_mare=0.1, ell_absolute_relative_error=0.1,
                       full_heldback_log_loss=0.2)
    return {**spec, "arm": arm, "attempt": 1, "status": "ok", "metrics": metrics,
            "data_identity": f"data-{spec['case_id']}",
            "catalogue_identity": f"catalogue-{spec['case_id']}",
            "matching_identity": f"match-{spec['case_id']}", "elapsed_seconds": 1.0}


def report(design, phase="confirmation", indices=None):
    result = protocol.new_report(phase, design, provenance(design))
    for index in design["scope"][phase] if indices is None else indices:
        for arm in protocol.ARMS:
            protocol.record_attempt(result, row(design, phase, index, arm))
    return result


def test_complete_physical_success_and_separate_secondary_metrics(design):
    result = report(design)
    decision = protocol.evaluate_gate(result, design)
    assert decision["complete"] and decision["passed"]
    assert decision["retained_baseline"] == "D"
    assert decision["summary"]["patterns"] == 24
    assert decision["summary"]["radius_wins"] == 24
    assert "median_xi_error_change" in decision["summary"]
    assert "median_matrix_loss_change" in decision["summary"]
    assert protocol.require_test_permission(result, design, provenance(design))["passed"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "wrong_phase",
                                   "wrong_index", "invalid_arm", "boolean_attempt",
                                   "malformed", "bad_status", "missing_metric",
                                   "nan", "boolean_error", "negative_error"])
def test_accounting_cannot_be_rescued_by_saved_passed_flag(design, defect):
    result = report(design)
    target = result["rows"][3]
    if defect == "missing":
        result["rows"].pop()
    elif defect == "duplicate":
        result["rows"].append(deepcopy(target))
    elif defect == "unexpected":
        target["case_id"] = "favourable-extra-case"
    elif defect == "wrong_phase":
        target["phase"] = "test"
    elif defect == "wrong_index":
        target["case_index"] = 99
    elif defect == "invalid_arm":
        target["arm"] = "other"
    elif defect == "boolean_attempt":
        target["attempt"] = True
    elif defect == "malformed":
        result["rows"][3] = []
    elif defect == "bad_status":
        target["status"] = "running"
    elif defect == "missing_metric":
        del target["metrics"]["xi_absolute_relative_error"]
    elif defect == "nan":
        target["metrics"]["radius_mare"] = np.nan
    elif defect == "boolean_error":
        target["metrics"]["ell_absolute_relative_error"] = False
    elif defect == "negative_error":
        target["metrics"]["ell_absolute_relative_error"] = -0.1
    result["decision"] = {"passed": True}
    decision = protocol.evaluate_gate(result, design)
    assert not decision["complete"] and not decision["passed"]
    with pytest.raises(RuntimeError):
        protocol.require_test_permission(result, design)


def test_failed_canonical_attempt_is_complete_failure_and_retry_cannot_rescue(design):
    result = report(design)
    successful_retry = deepcopy(result["rows"][3])
    successful_retry["attempt"] = 2
    result["rows"][3].update(status="error", error="solver rejected the physical domain")
    result["rows"][3].pop("metrics")
    protocol.record_attempt(result, successful_retry)
    decision = protocol.evaluate_gate(result, design)
    assert decision["complete"] and not decision["passed"]
    assert decision["failed_records"] == 1
    assert decision["canonical_records"] == 96
    assert decision["retained_baseline"] == "B"


def test_historical_failure_is_retained_without_misrepresenting_corrected_comparison(design):
    result = report(design)
    result["rows"][0].update(status="error", error="historical overlap")
    result["rows"][0].pop("metrics")
    decision = protocol.evaluate_gate(result, design)
    assert decision["passed"] and decision["failed_records"] == 1


@pytest.mark.parametrize("metric", ["feasible", "probability_valid", "converged",
                                    "objective_contract_passed"])
def test_every_C_D_case_must_meet_each_numerical_condition(design, metric):
    result = report(design)
    result["rows"][2]["metrics"][metric] = False
    decision = protocol.evaluate_gate(result, design)
    assert decision["complete"] and not decision["passed"]
    assert not decision["geometry_useful"]


def test_geometry_only_success_keeps_reserved_test_closed(design):
    result = report(design)
    for item in result["rows"]:
        if item["arm"] == "D":
            item["metrics"]["ell_absolute_relative_error"] = 0.3
    decision = protocol.evaluate_gate(result, design)
    assert decision["geometry_useful"] and not decision["physical_inference_useful"]
    assert decision["retained_baseline"] == "B"
    with pytest.raises(RuntimeError, match="test remains unopened"):
        protocol.require_test_permission(result, design)


@pytest.mark.parametrize("metric,value", [("search_bound_contact", True), ("ambiguous", True),
                                         ("full_heldback_log_loss", 0.3)])
def test_each_no_harm_condition_is_required(design, metric, value):
    result = report(design)
    for item in result["rows"]:
        if item["arm"] == "D":
            item["metrics"][metric] = value
    assert not protocol.evaluate_gate(result, design)["passed"]


def test_wins_are_strict_and_all_24_patterns_remain_in_denominator(design):
    result = report(design)
    for item in result["rows"]:
        if item["arm"] == "D" and item["case_index"] >= 17:
            item["metrics"]["radius_mare"] = 0.2
    decision = protocol.evaluate_gate(result, design)
    assert decision["summary"]["radius_wins"] == 17
    assert not decision["passed"]
    next(item for item in result["rows"] if item["arm"] == "D" and item["case_index"] == 17)["metrics"]["radius_mare"] = 0.1
    assert protocol.evaluate_gate(result, design)["passed"]


@pytest.mark.parametrize("identity", ["data_identity", "catalogue_identity", "matching_identity"])
def test_shared_catalogue_split_and_matching_cannot_change_across_arms(design, identity):
    result = report(design)
    result["rows"][3][identity] += "-changed"
    assert not protocol.evaluate_gate(result, design)["complete"]


def test_development_is_never_promotion_and_smoke_requires_cost_and_numerical_success(design):
    full = report(design, "development")
    assert not protocol.evaluate_gate(full, design)["passed"]
    result = report(design, "development", design["smoke_cases"])
    decision = protocol.smoke_gate(result, design)
    assert decision["passed"] and not decision["promotion"]
    assert decision["runtime"]["sum_seconds"] == 16
    result["rows"][1]["metrics"]["converged"] = False
    assert not protocol.smoke_gate(result, design)["passed"]
    result["rows"][1]["metrics"]["converged"] = True
    result["rows"][0].pop("elapsed_seconds")
    assert not protocol.smoke_gate(result, design)["passed"]


def test_smoke_does_not_drop_an_unexpected_record(design):
    result = report(design, "development", design["smoke_cases"])
    unexpected = deepcopy(result["rows"][0])
    unexpected["case_index"] = 999
    unexpected["case_id"] = "unexpected"
    result["rows"].append(unexpected)
    assert not protocol.smoke_gate(result, design)["passed"]


def test_case_identity_streams_are_deterministic_disjoint_and_design_changes_refuse_resume(design, tmp_path):
    manifest = [protocol.case_spec(phase, index, design)
                for phase in protocol.PHASES for index in design["scope"][phase]]
    assert len({spec["case_id"] for spec in manifest}) == 60
    assert len({seed for spec in manifest for seed in spec["seeds"].values()}) == 480
    assert manifest[0] == protocol.case_spec("development", 0, design)
    changed = deepcopy(design)
    changed["physics"]["width"] = 0.4
    result = report(design)
    assert not protocol.evaluate_gate(result, changed)["passed"]
    path = tmp_path / "report.json"
    protocol.write_report(path, result)
    assert protocol.read_compatible(path, provenance(design)) == result
    with pytest.raises(RuntimeError, match="changed"):
        protocol.read_compatible(path, provenance(changed))
    altered = provenance(design)
    altered["runtime"]["fit_device"] = "mps"
    with pytest.raises(RuntimeError):
        protocol.require_test_permission(result, design, altered)


def test_source_manifest_includes_new_helper_and_metadata_not_output_reports(design, tmp_path):
    source = tmp_path / "geom_mesh_net" / "fields"
    source.mkdir(parents=True)
    (source / "helper.py").write_text("VALUE = 1\n")
    directory = tmp_path / "experiments" / "reconstruction" / "results"
    directory.mkdir(parents=True)
    (directory / "report.json").write_text("{}")
    first = protocol.make_provenance(design, {"truth_parameters": {"ell": 0.2}}, root=tmp_path)
    (directory / "report.json").write_text('{"progress": 2}')
    assert first == protocol.make_provenance(design, {"truth_parameters": {"ell": 0.2}}, root=tmp_path)
    (source / "new_helper.py").write_text("VALUE = 2\n")
    assert first != protocol.make_provenance(design, {"truth_parameters": {"ell": 0.2}}, root=tmp_path)
    assert first != protocol.make_provenance(design, {"truth_parameters": {"ell": 0.3}}, root=tmp_path)


def test_attempts_are_immutable_and_artifact_directories_cannot_be_overwritten(design, tmp_path):
    result = report(design)
    with pytest.raises(ValueError, match="already recorded"):
        protocol.record_attempt(result, result["rows"][0])
    case_id = result["rows"][0]["case_id"]
    folder = protocol.allocate_attempt_dir(tmp_path, case_id, "A", 1)
    assert folder.is_dir()
    with pytest.raises(FileExistsError):
        protocol.allocate_attempt_dir(tmp_path, case_id, "A", 1)
    assert protocol.allocate_attempt_dir(tmp_path, case_id, "A", 2) != folder
    path = tmp_path / "report.json"
    protocol.write_report(path, result)
    result["rows"][0]["metrics"]["radius_mare"] = 0.01
    with pytest.raises(RuntimeError, match="immutable"):
        protocol.write_report(path, result)


def test_changed_seed_metadata_cannot_pass_under_an_old_case_identity(design):
    result = report(design)
    result["rows"][3]["seeds"]["labels"] += 1
    assert not protocol.evaluate_gate(result, design)["passed"]


@pytest.mark.parametrize("defect", ["duplicate_case", "missing_arm", "births", "device", "weaker_gate"])
def test_invalid_design_contract_is_rejected(design, defect):
    if defect == "duplicate_case":
        design["scope"]["confirmation"][-1] = design["scope"]["confirmation"][0]
    elif defect == "missing_arm":
        del design["arms"]["C"]
    elif defect == "births":
        design["physics"]["births"] = True
    elif defect == "device":
        design["runtime"]["fit_device"] = "mps"
    elif defect == "weaker_gate":
        design["promotion"]["require_all_C_D_feasible_converged"] = False
    with pytest.raises(ValueError):
        protocol.validate_design(design)


def test_atomic_report_refuses_nonfinite_values_without_corrupting_existing_file(design, tmp_path):
    path = tmp_path / "report.json"
    original = report(design)
    protocol.write_report(path, original)
    bad = deepcopy(original)
    bad["extra"] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        protocol.write_report(path, bad)
    assert json.loads(path.read_text()) == original
    assert not list(tmp_path.glob("*.tmp"))
