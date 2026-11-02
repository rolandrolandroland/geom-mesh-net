"""Runner safety and scientific scoring checks without training or large fits."""

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


def provenance(design):
    return {"design": deepcopy(design), "design_id": protocol.digest(design),
            "sources": {}, "runtime": deepcopy(design["runtime"]), "inputs": {}}


def healthy_row(design, phase, index, arm):
    spec = protocol.case_spec(phase, index, design)
    return {**spec, "arm": arm, "attempt": 1, "status": "ok", "elapsed_seconds": 1.,
            "data_identity": "data:" + spec["case_id"],
            "catalogue_identity": "catalogue:" + spec["case_id"],
            "metrics": {"feasible": True, "probability_valid": True, "converged": True,
                        "objective_contract_passed": True, "search_bound_contact": False,
                        "ambiguous": False, "radius_mare": .1, "ell_absolute_relative_error": .1,
                        "xi_absolute_relative_error": .1, "full_heldback_log_loss": .2,
                        "matrix_heldback_log_loss": .2}}


def test_legacy_diagnostics_expose_invalid_blend_hidden_by_clipping():
    from geom_mesh_net.fields import joint_fit
    centre = np.array([[5., 5., 5.]])
    model = joint_fit.JointDiffusionModel(centre, np.array([2.]), c_eq=.05, ell=2.,
                                         xi=3., c_inf=.1, width=.3, knots=6)
    clipped = joint_fit.predict(model, centre)
    assert np.all((clipped >= 0) & (clipped <= 1))
    checked = runner.legacy_raw_diagnostics(model, centre, 10.)
    assert checked["passed"] is False
    assert checked["full"]["maximum"] > 1
    assert checked["historical_predictions_and_fitting_remain_clipped"] is True


@pytest.mark.parametrize("probabilities", [[-.01, .5], [.5, 1.01], [np.nan, .5],
                                          [np.inf, .5], []])
def test_strict_log_loss_refuses_invalid_raw_probabilities(probabilities):
    with pytest.raises(ValueError):
        runner.strict_log_loss(np.zeros(len(probabilities)), np.asarray(probabilities))


def test_strict_log_loss_matches_bernoulli_definition_and_rejects_wrong_shape():
    assert runner.strict_log_loss(np.array([0., 1.]), np.array([.2, .8])) == pytest.approx(-np.log(.8))
    with pytest.raises(ValueError):
        runner.strict_log_loss(np.zeros(2), np.array([.5]))


def test_fitted_particle_reassignment_cannot_improve_fixed_primary_radius_score():
    true_centres = np.array([[5., 5., 5.], [15., 15., 15.]])
    true_radii = np.array([2., 4.])
    matching = runner.freeze_matches(true_centres.copy(), true_centres, true_radii)
    # The fitted particles have exchanged positions and radii. Rematching would
    # give zero radius error, but the predeclared association must remain fixed.
    model = SimpleNamespace(radii=lambda: np.array([4., 2.]),
                            centres=lambda: true_centres[::-1],
                            constants=lambda: {"ell": 2., "xi": 3.},
                            raw_predict=lambda points: np.full(len(points), .2))
    arrays = {"radii": true_radii, "centres": true_centres, "x": np.array([[1., 1., 1.], [2., 2., 2.],
                                                 [3., 3., 3.], [4., 4., 4.]]),
              "y": np.array([0, 1, 0, 0]), "heldback": np.array([True, True, False, False]),
              "removed": np.array([False, False, True, True]), "owner": np.array([-1, 0, -1, -1])}
    record = {"geometry_feasible": True, "sampled_probability_valid": True, "converged": True,
              "assignment_stable": True, "starts": [{"evaluations": [{"likelihood_evaluated": True,
                  "geometry_feasible": True, "probability_valid": True}]}]}
    metadata = {"field": {"ell": 2., "xi": 3.}}
    metrics = runner.score_model(model, record, "D", arrays, metadata, matching)
    assert metrics["radius_mare"] == pytest.approx(.75)
    assert matching["candidate_indices"] == [0, 1]


def test_failed_confirmation_cannot_generate_or_open_test_data(design, monkeypatch, tmp_path):
    design["status"] = "frozen"
    prov = provenance(design)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    development = protocol.new_report("development", design, prov)
    for index in design["scope"]["development"]:
        for arm in protocol.ARMS:
            protocol.record_attempt(development, healthy_row(design, "development", index, arm))
    protocol.write_report(tmp_path / "development.json", development)
    confirmation = protocol.new_report("confirmation", design, prov)
    confirmation["gate"] = {"passed": True}  # A saved flag is not authorization.
    protocol.write_report(tmp_path / "confirmation.json", confirmation)
    monkeypatch.setattr(protocol, "make_provenance", lambda *args, **kwargs: prov)
    def forbidden(*args, **kwargs):
        pytest.fail("reserved test was generated or opened before confirmation passed")
    monkeypatch.setattr(simulation := runner.simulation, "generate_case", forbidden)
    monkeypatch.setattr(simulation, "load_case", forbidden)
    with pytest.raises(RuntimeError):
        runner.main(["--phase", "test", "--design", str(design_path),
                     "--output", str(tmp_path), "--artifacts", str(tmp_path / "artifacts")])
    assert not (tmp_path / "artifacts").exists()


def test_interrupted_attempt_remains_primary_failure_and_is_not_refitted(design, monkeypatch, tmp_path):
    spec = protocol.case_spec("development", 0, design)
    catalog = SimpleNamespace(fingerprint="catalogue-hash")
    metadata, matching = {"data_identity": "data-hash"}, {"identity": "matching-hash"}
    artifacts = tmp_path / "artifacts"
    folder = protocol.allocate_attempt_dir(artifacts / "fits", spec["case_id"], "B", 1)
    started = {**spec, "arm": "B", "attempt": 1, "status": "error", "error": "started",
               "execution_fingerprint": protocol.digest(provenance(design)),
               "metrics": {}, "data_identity": metadata["data_identity"],
               "catalogue_identity": catalog.fingerprint, "matching_identity": matching["identity"]}
    protocol.write_report(folder / "started.json", started)
    report = protocol.new_report("development", design, provenance(design))
    def forbidden(*args, **kwargs):
        pytest.fail("an interrupted canonical attempt must not be retried")
    monkeypatch.setattr(runner.constrained, "fit_constrained", forbidden)
    row = runner.fit_arm("B", None, catalog, {}, metadata, matching, spec, design,
                         report, tmp_path / "report.json", artifacts)
    assert row["status"] == "error" and "interrupted" in row["error"]
    assert row["elapsed_seconds"] is None
    assert len(report["rows"]) == 1
    assert (folder / "started.json").exists() and (folder / "result.json").exists()
    repeated = runner.fit_arm("B", None, catalog, {}, metadata, matching, spec, design,
                              report, tmp_path / "report.json", artifacts)
    assert repeated == row and len(report["rows"]) == 1


def test_orphaned_attempt_with_changed_catalogue_cannot_be_adopted(design, tmp_path):
    spec = protocol.case_spec("development", 0, design)
    artifacts = tmp_path / "artifacts"
    folder = protocol.allocate_attempt_dir(artifacts / "fits", spec["case_id"], "B", 1)
    started = {**spec, "arm": "B", "attempt": 1, "status": "error", "error": "started",
               "execution_fingerprint": protocol.digest(provenance(design)),
               "metrics": {}, "data_identity": "data", "catalogue_identity": "old",
               "matching_identity": "match"}
    protocol.write_report(folder / "started.json", started)
    report = protocol.new_report("development", design, provenance(design))
    with pytest.raises(RuntimeError, match="mismatched"):
        runner.fit_arm("B", None, SimpleNamespace(fingerprint="new"), {},
                       {"data_identity": "data"}, {"identity": "match"}, spec, design,
                       report, tmp_path / "report.json", artifacts)
    assert report["rows"] == []
