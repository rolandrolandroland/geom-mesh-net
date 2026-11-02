"""A fresh report cannot relabel old cached data or fits with new execution provenance."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction import e21_geometry as runner
from experiments.reconstruction import e21_protocol as protocol


@pytest.fixture
def design():
    return protocol.load_design()


def report(design, phase="development", source="original"):
    provenance = {"design": deepcopy(design), "design_id": protocol.digest(design),
                  "sources": {"fitter.py": source}, "runtime": deepcopy(design["runtime"]), "inputs": {}}
    return protocol.new_report(phase, design, provenance)


def case_folder(artifacts, spec):
    return artifacts / "cases" / protocol.digest(spec["case_id"])[:24]


@pytest.mark.parametrize("binding", ["changed_source", "missing"])
def test_new_report_refuses_old_case_before_data_or_neural_cache_reads(design, tmp_path, monkeypatch, binding):
    spec = protocol.case_spec("development", 0, design)
    artifacts = tmp_path / "artifacts"
    folder = case_folder(artifacts, spec)
    folder.mkdir(parents=True)
    (folder / "fourier").mkdir()
    old = report(design)
    if binding != "missing":
        runner.case_execution_provenance(folder, spec, old, create=True)
    current = report(design, source="changed_optimizer_source")
    def forbidden(*args, **kwargs):
        pytest.fail("a foreign case was read before execution provenance was checked")
    monkeypatch.setattr(runner.simulation, "load_case", forbidden)
    monkeypatch.setattr(runner.simulation, "generate_case", forbidden)
    monkeypatch.setattr(runner.diagnostics, "load_fourier_artifacts", forbidden)
    monkeypatch.setattr(runner.diagnostics, "fit_fourier_with_artifacts", forbidden)
    with pytest.raises(RuntimeError, match="case execution provenance"):
        runner.run_case(0, "development", design, current, tmp_path / "report.json", artifacts)
    assert current["rows"] == []
    if binding != "missing":
        assert json.loads((folder / "execution_provenance.json").read_text())["execution_fingerprint"] == old["fingerprint"]


def test_case_binding_is_immutable_but_new_frozen_confirmation_case_is_allowed(design, tmp_path):
    spec = protocol.case_spec("development", 0, design)
    original = report(design)
    folder = case_folder(tmp_path, spec)
    folder.mkdir(parents=True)
    expected = runner.case_execution_provenance(folder, spec, original, create=True)
    assert runner.case_execution_provenance(folder, spec, original) == expected
    with pytest.raises(RuntimeError, match="case execution provenance"):
        runner.case_execution_provenance(folder, spec, report(design, source="new"), create=True)
    frozen = deepcopy(design)
    frozen["status"] = "frozen"
    confirmation = report(frozen, "confirmation")
    new_spec = protocol.case_spec("confirmation", 0, frozen)
    new_folder = case_folder(tmp_path, new_spec)
    assert new_folder != folder
    new_folder.mkdir(parents=True)
    runner.case_execution_provenance(new_folder, new_spec, confirmation, create=True)
    assert runner.case_execution_provenance(folder, spec, original) == expected


@pytest.mark.parametrize("fingerprint", [None, "old-execution-fingerprint"])
def test_orphan_fits_need_execution_identity_even_when_data_and_catalogue_match(design, tmp_path, fingerprint):
    spec = protocol.case_spec("development", 0, design)
    current = report(design, source="current")
    artifacts = tmp_path / "artifacts"
    folder = protocol.allocate_attempt_dir(artifacts / "fits", spec["case_id"], "B", 1)
    started = {**spec, "arm": "B", "attempt": 1, "status": "error", "error": "started",
               "data_identity": "same-data", "catalogue_identity": "same-catalogue",
               "matching_identity": "same-matches", "metrics": {}}
    if fingerprint is not None:
        started["execution_fingerprint"] = fingerprint
    protocol.write_report(folder / "started.json", started)
    with pytest.raises(RuntimeError, match="attempt execution provenance"):
        runner.fit_arm("B", None, SimpleNamespace(fingerprint="same-catalogue"), {},
                       {"data_identity": "same-data"}, {"identity": "same-matches"}, spec, design,
                       current, tmp_path / "report.json", artifacts)
    assert current["rows"] == []
    assert not (folder / "result.json").exists()
    assert json.loads((folder / "started.json").read_text()) == started


def test_completed_same_source_orphan_is_recovered_without_refitting(design, tmp_path, monkeypatch):
    spec = protocol.case_spec("development", 0, design)
    current = report(design)
    artifacts = tmp_path / "artifacts"
    folder = protocol.allocate_attempt_dir(artifacts / "fits", spec["case_id"], "B", 1)
    completed = {**spec, "arm": "B", "attempt": 1, "status": "ok", "metrics": {},
                 "execution_fingerprint": current["fingerprint"], "data_identity": "data",
                 "catalogue_identity": "catalogue", "matching_identity": "matches"}
    protocol.write_report(folder / "result.json", completed)
    def forbidden(*args, **kwargs):
        pytest.fail("completed compatible orphan must not be fitted again")
    monkeypatch.setattr(runner.constrained, "fit_constrained", forbidden)
    recovered = runner.fit_arm("B", None, SimpleNamespace(fingerprint="catalogue"), {},
                               {"data_identity": "data"}, {"identity": "matches"}, spec, design,
                               current, tmp_path / "report.json", artifacts)
    assert recovered == completed and current["rows"] == [completed]


def test_new_case_is_saved_before_binding_and_binding_precedes_fitting(design, tmp_path, monkeypatch):
    spec = protocol.case_spec("development", 0, design)
    current = report(design)
    artifacts = tmp_path / "artifacts"
    folder = case_folder(artifacts, spec)
    arrays = {"x": np.array([[1., 2., 3.], [2., 3., 4.]]), "y": np.array([0, 1]),
              "fitting": np.array([True, True]), "centres": np.array([[5., 5., 5.]]), "radii": np.array([2.])}
    metadata = {"data_identity": "data"}
    monkeypatch.setattr(runner.simulation, "generate_case", lambda *args: (arrays, metadata))
    catalog = SimpleNamespace(centres=arrays["centres"], radii=arrays["radii"],
                              to_dict=lambda: {}, geometry=lambda: None)
    monkeypatch.setattr(runner, "gaussian_catalogue", lambda *args: (catalog, None, {}))
    monkeypatch.setattr(runner, "freeze_matches", lambda *args: {"identity": "matches"})
    def fit(*args, **kwargs):
        assert (folder / "data.npz").exists() and (folder / "metadata.json").exists()
        binding = json.loads((folder / "execution_provenance.json").read_text())
        assert binding["execution_fingerprint"] == current["fingerprint"]
        return {"status": "ok", "metrics": {name: True for name in
                ("feasible", "probability_valid", "converged", "objective_contract_passed")}}
    monkeypatch.setattr(runner, "fit_arm", fit)
    assert runner.run_case(0, "development", design, current, tmp_path / "report.json", artifacts, preflight=True)
    runner.case_execution_provenance(folder, spec, current)
