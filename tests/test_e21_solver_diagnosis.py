"""Diagnostic protocol checks without running a physical fitting experiment."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.reconstruction import e21_protocol as protocol
from experiments.reconstruction import e21_solver_diagnosis as runner
from geom_mesh_net.fields.boundary_refinement import CandidateCatalog


@pytest.fixture
def design():
    return runner.load_design()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def provenance(design):
    return {"schema": "test-diagnosis-provenance", "design": deepcopy(design),
            "design_id": protocol.digest(design), "sources": {"solver.py": "frozen"},
            "runtime": deepcopy(design["runtime"])}


def small_case(design, name="observed"):
    count = 3 if name == "observed" else 1
    constants = [{"c_eq": .03 + i * .001, "ell": 2., "xi": 5., "c_inf": .1}
                 for i in range(count)]
    return {"id": name, "x": np.array([[2., 2., 2.], [8., 8., 8.], [10., 10., 10.]]),
            "y": np.array([.1, .7, .2]), "box": 32.,
            "centres": np.array([[8., 8., 8.], [20., 20., 20.]]),
            "radii": np.array([2., 3.]), "candidate_ids": ["first", "second"],
            "starts": deepcopy(design["observed"]["starts"][:count]),
            "initial_constants": constants, "initial_levels": None,
            "truth_entropy": .2, "identity": {"fitting_identity": name + "-data"},
            "legacy_starts": [{}], "legacy_configuration": {}}


@pytest.fixture
def archive(design, tmp_path, monkeypatch):
    """A tiny pinned development archive; simulation verification remains a seam."""
    old_design = protocol.load_design()
    spec = protocol.case_spec("development", 0, old_design)
    catalog = CandidateCatalog([[8., 8., 8.], [20., 20., 20.]], [2., 3.])
    config = deepcopy(design)
    folder = tmp_path / "case"
    prep_path, report_path = folder / "preparation.json", tmp_path / "development.json"
    save_json(prep_path, {"catalogue": catalog.to_dict()})
    starts = [{"initial_feasibility_projection": {"after": constants}}
              for constants in small_case(design)["initial_constants"]]
    row = {"phase": "development", "case_index": 0, "arm": "B", "attempt": 1,
           "case_id": spec["case_id"], "data_identity": "pinned-data",
           "catalogue_identity": catalog.fingerprint,
           "fit": {"starts": starts, "configuration": {},
                   "preparation": {"radii": [1.9, 2.9], "candidate_ids": list(catalog.candidate_ids)}}}
    saved = {"phase": "development", "rows": [row], "provenance": {"design": old_design}}
    save_json(report_path, saved)
    config["observed"].update(report="development.json", case_folder="case",
                              report_sha256=runner.file_hash(report_path),
                              preparation_sha256=runner.file_hash(prep_path),
                              data_identity="pinned-data", catalogue_identity=catalog.fingerprint)
    arrays = {"x": np.arange(15., dtype=float).reshape(5, 3),
              "y": np.array([0., 987., 1., 654., 0.]),
              "fitting": np.array([True, False, True, False, True])}
    calls = []
    def checked_load(path, requested_spec, requested_design):
        calls.append((Path(path), requested_spec, requested_design))
        assert Path(path) == folder and requested_spec == spec and requested_design == old_design
        return arrays, {"data_identity": "pinned-data"}
    monkeypatch.setattr(runner.simulation, "load_case", checked_load)
    return config, arrays, calls, saved, report_path, prep_path


@pytest.mark.parametrize("phase,index", [("confirmation", 0), ("test", 0), ("development", 1)])
def test_non_development_zero_scope_is_refused_before_archive_reads(design, monkeypatch, phase, index):
    design["observed"].update(phase=phase, case_index=index)
    monkeypatch.setattr(runner, "file_hash", lambda *_: pytest.fail("forbidden archive was opened"))
    with pytest.raises(ValueError, match="development case 0"):
        runner.load_observed_case(design)


@pytest.mark.parametrize("field", ["promotion", "reserved_data_access"])
def test_design_cannot_enable_promotion_or_reserved_data(design, tmp_path, field):
    design[field] = True
    path = tmp_path / "design.json"
    save_json(path, design)
    with pytest.raises(ValueError, match="cannot promote"):
        runner.load_design(path)


def test_observed_loader_verifies_archive_and_returns_only_fitting_rows(archive, tmp_path):
    design, arrays, calls, _, _, _ = archive
    case = runner.load_observed_case(design, tmp_path)
    assert len(calls) == 1
    np.testing.assert_array_equal(case["x"], arrays["x"][arrays["fitting"]])
    np.testing.assert_array_equal(case["y"], [0., 1., 0.])
    np.testing.assert_array_equal(case["radii"], [1.9, 2.9])
    assert case["identity"]["fitting_identity"] == protocol.array_hash(case["x"], case["y"])
    case["x"][0, 0] = -1
    assert arrays["x"][0, 0] == 0


@pytest.mark.parametrize("target", ["report", "preparation", "catalogue", "data"])
def test_pinned_archive_changes_fail_before_observation_loading(archive, tmp_path, target):
    design, _, calls, saved, report_path, prep_path = archive
    if target == "report":
        report_path.write_text(report_path.read_text() + " ")
    elif target == "preparation":
        prep_path.write_text(prep_path.read_text() + " ")
    elif target == "catalogue":
        design["observed"]["catalogue_identity"] = "foreign-catalogue"
    else:
        saved["rows"][0]["data_identity"] = "foreign-data"
        save_json(report_path, saved)
        design["observed"]["report_sha256"] = runner.file_hash(report_path)
    with pytest.raises(RuntimeError, match="identity|preparation"):
        runner.load_observed_case(design, tmp_path)
    assert calls == []


def test_verified_simulation_metadata_must_match_pinned_data(archive, tmp_path, monkeypatch):
    design, arrays, _, _, _, _ = archive
    monkeypatch.setattr(runner.simulation, "load_case", lambda *_: (arrays, {"data_identity": "foreign"}))
    with pytest.raises(RuntimeError, match="data identity"):
        runner.load_observed_case(design, tmp_path)


def test_control_is_deterministic_exact_probability_target_with_unequal_radii(design):
    small = deepcopy(design)
    small["control"]["count"] = 25
    first, second = runner.make_control(small), runner.make_control(small)
    np.testing.assert_array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(first["y"], second["y"])
    assert len(np.unique(first["radii"])) == 3
    assert np.all((first["y"] > 0) & (first["y"] < 1))
    truth = runner.fitter.ConstrainedDiffusionModel(first["centres"], first["radii"],
             **small["control"]["truth"], levels=small["control"]["levels"], box=first["box"],
             candidate_ids=first["candidate_ids"], width=small["physics"]["width"],
             knots=small["physics"]["knots"], min_gap=small["physics"]["min_gap"],
             radius_bounds=small["physics"]["radius_bounds"])
    np.testing.assert_array_equal(first["y"], truth.raw_predict(first["x"]))
    assert first["identity"] == second["identity"]


@pytest.mark.parametrize("mutation", ["provenance", "fingerprint", "duplicate", "reserved"])
def test_cached_report_rejects_changed_provenance_or_attempt_identity(design, tmp_path, mutation):
    prov = provenance(design)
    path = tmp_path / "report.json"
    report = runner.load_report(path, prov)
    if mutation == "provenance":
        report["provenance"] = dict(prov, sources={"solver.py": "changed"})
    elif mutation == "fingerprint":
        report["fingerprint"] = "changed"
    elif mutation == "duplicate":
        report["rows"] = [{"case": "observed", "stage": "chemistry", "start": 0}] * 2
    else:
        report["rows"] = [{"case": "test", "stage": "chemistry", "start": 0}]
    save_json(path, report)
    with pytest.raises(RuntimeError, match="provenance|identity"):
        runner.load_report(path, prov)


def test_resolve_start_preserves_full_state_and_own_nonconverged_parent(design):
    case = small_case(design)
    cold = runner.initial_states(case, design)
    parent_state = deepcopy(cold[1])
    parent_state["vector"][-1] += .037
    parents = [{"case": "observed", "stage": "chemistry", "start": i,
                "converged": False, "last_accepted_state": deepcopy(cold[i])} for i in range(3)]
    parents[1]["last_accepted_state"] = parent_state
    stage = next(s for s in design["stages"] if s["id"] == "warm_radii")
    state, parent = runner.resolve_start(stage, 1, cold, parents, "observed")
    assert state == parent_state and parent is parents[1]
    assert state != cold[1] and parent["converged"] is False
    parents[1]["last_accepted_state"] = None
    assert runner.resolve_start(stage, 1, cold, parents, "observed") == (None, parents[1])
    for independent in design["stages"][:4]:
        state, parent = runner.resolve_start(independent, 1, cold, parents, "observed")
        assert state == cold[1] and parent is None


def fake_fit(calls):
    def fit(centres, radii, x, y, **kwargs):
        calls.append(deepcopy(kwargs))
        state = deepcopy(kwargs["initial_states"][0])
        state["vector"][-1] += .001 * len(calls)
        state["vector"][4 * len(radii)] += .0001
        attempt = {"status": "not_converged", "converged": False,
                   "last_accepted_state": state, "last_accepted_metadata": {"accepted": True},
                   "evaluations": [{"likelihood_evaluated": True, "geometry_feasible": True,
                                    "probability_valid": True}], "rounds": [],
                   "train_loss": .5, "summed_nll": .5 * len(y), "assignment_stable": False,
                   "termination_reason": "round_budget_exhausted"}
        return None, {"starts": [attempt], "converged": False, "status": "no_converged_start"}
    return fit


def test_all_independent_and_warm_runs_preserve_lineage_without_promotion(design, tmp_path, monkeypatch):
    observed, control = small_case(design), small_case(design, "control")
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "make_provenance", lambda _: provenance(design))
    monkeypatch.setattr(runner.e21, "preserve_sources", lambda *_: None)
    monkeypatch.setattr(runner, "make_control", lambda _: control)
    monkeypatch.setattr(runner, "load_observed_case", lambda _: observed)
    monkeypatch.setattr(runner.simulation, "load_case", lambda *_: pytest.fail("no extra dataset reads allowed"))
    monkeypatch.setattr(runner.simulation, "generate_case", lambda *_: pytest.fail("no dataset generation allowed"))
    monkeypatch.setattr(runner.fitter, "fit_constrained", fake_fit(calls))
    monkeypatch.setattr(runner.solver_diagnostics, "model_from_fit_state", lambda *_, **__: object())
    monkeypatch.setattr(runner.solver_diagnostics, "diagnose_radius_knots", lambda *_, **__: {"joint_convergence_assessed": False})
    report = runner.run_diagnosis(design, tmp_path / "out", tmp_path / "artifacts")
    assert len(report["rows"]) == len(calls) == 22
    assert report["requested_scope_complete"] and report["all_diagnostic_stages_complete"]
    assert report["promotion"] is report["inference_validated"] is False
    assert all(not row["converged"] and not row["promotion"] for row in report["rows"])
    by_key = {runner.row_key(row): row for row in report["rows"]}
    observed_calls = calls[4:]
    for start in range(3):
        independent = [observed_calls[stage * 3 + start] for stage in range(4)]
        assert all(call["initial_states"] == independent[0]["initial_states"] for call in independent)
        assert [(call["fit_centres"], call["fit_radii"]) for call in independent] == [
            (False, False), (False, True), (True, False), (True, True)]
        for stage, parent_stage, offset in [("warm_radii", "chemistry", 12), ("warm_joint", "warm_radii", 15)]:
            parent = by_key[("observed", parent_stage, start)]
            row = by_key[("observed", stage, start)]
            assert observed_calls[offset + start]["initial_states"] == [parent["last_accepted_state"]]
            assert row["parent"]["key"] == ["observed", parent_stage, start]
            assert row["parent"]["converged"] is False
            assert row["cumulative_evaluations"] == row["evaluations"] + parent["cumulative_evaluations"]
    for call in calls:
        assert call["maxiter"] == 150 and call["max_evaluations"] == 600 and call["rounds"] == 3
        assert call["gtol"] == 1e-5 and call["prepare_geometry"] is False
        assert "validation_points" not in call


def test_control_only_execution_never_opens_observed_or_reserved_data(design, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "make_provenance", lambda _: provenance(design))
    monkeypatch.setattr(runner.e21, "preserve_sources", lambda *_: None)
    monkeypatch.setattr(runner, "make_control", lambda _: small_case(design, "control"))
    monkeypatch.setattr(runner, "load_observed_case", lambda *_: pytest.fail("observed data opened in control-only mode"))
    monkeypatch.setattr(runner.fitter, "fit_constrained", fake_fit([]))
    report = runner.run_diagnosis(design, tmp_path / "out", tmp_path / "artifacts", ("control",))
    assert len(report["rows"]) == 4 and report["requested_scope_complete"]
    assert not report["all_diagnostic_stages_complete"] and not report["promotion"]


@pytest.mark.parametrize("name", ["confirmation", "test", "observed-1"])
def test_unknown_case_is_rejected_before_provenance_or_data_access(design, tmp_path, monkeypatch, name):
    monkeypatch.setattr(runner, "make_provenance", lambda *_: pytest.fail("scope guard ran too late"))
    with pytest.raises(ValueError, match="only the control"):
        runner.run_diagnosis(design, tmp_path / "out", tmp_path / "artifacts", (name,))


def test_failed_parent_becomes_retained_dependency_failure(design, tmp_path, monkeypatch):
    case = small_case(design)
    report = runner.load_report(tmp_path / "report.json", provenance(design))
    stage = next(s for s in design["stages"] if s["id"] == "warm_radii")
    parent = {"case": "observed", "stage": "chemistry", "start": 0,
              "last_accepted_state": None, "converged": False,
              "cumulative_evaluations": 17, "cumulative_fit_seconds": 2.5}
    monkeypatch.setattr(runner.fitter, "fit_constrained", lambda *_, **__: pytest.fail("missing parent must not be replaced"))
    row = runner.run_stage(case, stage, 0, None, parent, design, report, tmp_path / "artifacts")
    assert row["status"] == "dependency_failure" and row["last_accepted_state"] is None
    assert row["cumulative_evaluations"] == 17 and row["cumulative_fit_seconds"] == 2.5
    assert row["promotion"] is row["converged"] is False
    assert (tmp_path / "artifacts" / "observed" / "warm_radii" / "start_00" / "result.json").exists()


def test_invalid_initial_state_gets_terminal_record(design, tmp_path):
    case = small_case(design)
    report = runner.load_report(tmp_path / "report.json", provenance(design))
    row = runner.run_stage(case, design["stages"][0], 0, {"state_version": "invalid"},
                           None, design, report, tmp_path / "artifacts")
    assert row["status"] == "diagnostic_error" and row["converged"] is False
    assert row["last_accepted_state"] is None
    assert (tmp_path / "artifacts" / "observed" / "chemistry" / "start_00" / "result.json").exists()


@pytest.mark.parametrize("mutation", ["execution", "state", "data", "changed_fit", "missing_fit_hash", "terminal_state"])
def test_cached_stage_refuses_foreign_or_unverifiable_results(design, tmp_path, monkeypatch, mutation):
    case = small_case(design)
    state = runner.initial_states(case, design)[0]
    report = runner.load_report(tmp_path / "report.json", provenance(design))
    artifacts = tmp_path / "artifacts"
    stage = design["stages"][0]
    monkeypatch.setattr(runner.fitter, "fit_constrained", fake_fit([]))
    row = runner.run_stage(case, stage, 0, state, None, design, report, artifacts)
    assert row["status"] == "not_converged"
    report["rows"].append(row)
    if mutation == "execution":
        row["execution_fingerprint"] = "foreign-source"
    elif mutation == "state":
        row["initial_state_identity"] = "different-initial-state"
    elif mutation == "data":
        row["data_identity"] = {"fitting_identity": "different-data"}
    elif mutation == "changed_fit":
        fit_path = artifacts / "observed" / "chemistry" / "start_00" / "fit.json"
        fit_path.write_text(fit_path.read_text() + " ")
    elif mutation == "missing_fit_hash":
        row.pop("fit_sha256")
    else:
        row["last_accepted_state"]["vector"][-1] += .125
    monkeypatch.setattr(runner.fitter, "fit_constrained", lambda *_, **__: pytest.fail("incompatible cache was retried"))
    with pytest.raises(RuntimeError, match="identity|state|artifact|checksum|provenance"):
        runner.run_stage(case, stage, 0, state, None, design, report, artifacts)


def test_same_source_orphan_is_recovered_without_reexecution(design, tmp_path, monkeypatch):
    case = small_case(design)
    state = runner.initial_states(case, design)[0]
    report = runner.load_report(tmp_path / "report.json", provenance(design))
    artifacts = tmp_path / "artifacts"
    stage = design["stages"][0]
    monkeypatch.setattr(runner.fitter, "fit_constrained", fake_fit([]))
    original = runner.run_stage(case, stage, 0, state, None, design, report, artifacts)
    assert report["rows"] == []
    monkeypatch.setattr(runner.fitter, "fit_constrained", lambda *_, **__: pytest.fail("compatible orphan was executed again"))
    recovered = runner.run_stage(case, stage, 0, state, None, design, report, artifacts)
    assert recovered == original
    foreign = deepcopy(report)
    foreign["fingerprint"] = "foreign-source"
    with pytest.raises(RuntimeError, match="provenance"):
        runner.run_stage(case, stage, 0, state, None, design, foreign, artifacts)


def test_interrupted_attempt_is_terminal_and_never_restarted(design, tmp_path, monkeypatch):
    case = small_case(design)
    state = runner.initial_states(case, design)[0]
    report = runner.load_report(tmp_path / "report.json", provenance(design))
    artifacts = tmp_path / "artifacts"
    stage = design["stages"][0]
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated process interruption")
    monkeypatch.setattr(runner.fitter, "fit_constrained", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(case, stage, 0, state, None, design, report, artifacts)
    folder = artifacts / "observed" / "chemistry" / "start_00"
    assert (folder / "started.json").exists() and not (folder / "result.json").exists()
    monkeypatch.setattr(runner.fitter, "fit_constrained", lambda *_, **__: pytest.fail("interrupted attempt was restarted"))
    recovered = runner.run_stage(case, stage, 0, state, None, design, report, artifacts)
    assert recovered["status"] == "interrupted"
    assert recovered["last_accepted_state"] is None and recovered["converged"] is False
    assert (folder / "result.json").exists()
