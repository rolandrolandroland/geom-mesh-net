"""Protocol checks for the particle pilot, without large data or neural training."""

import copy
import inspect
import json

import numpy as np
import pytest
import torch

from experiments.reconstruction import stage4_particles as stage
from geom_mesh_net.neural import implicit


@pytest.fixture
def design():
    return {
        "efficiency": 0.37,
        "held_back_fraction": 0.2,
        "outer_split_entropy": 20261015,
        "scope": {"calibration": [0, 1], "confirmation": [4, 5], "test": [74, 75]},
        "field_thresholds": [0.2, 0.4],
        "field_peak_distances": [2.0],
        "field_min_volume": 14.0,
        "point_eps": [1.0, 2.0],
        "point_min_cluster_size": [5],
        "dbscan_min_samples": [3],
        "selection": "mean F1, mean precision, mean relative radius MAE, candidate order",
        "promotion": {
            "f1_win_fraction": 0.75,
            "minimum_median_precision_difference": -0.02,
            "maximum_median_matrix_loss_difference": 0.0,
        },
    }


@pytest.fixture
def neural_design():
    return {"split": {"entropy": 20261001, "validation_fraction": 0.2, "streams": {"split": 0}}}


def calibration_cells(design):
    return [
        {
            "pattern": pattern,
            "candidates": {
                method: [
                    {"setting": setting, "metrics": {"f1": 0.6, "precision": 0.7, "recall": 0.6,
                                                       "relative_radius_mae": 0.2}}
                    for setting in stage.candidates(design, method)
                ]
                for method in stage.METHODS
            },
        }
        for pattern in design["scope"]["calibration"]
    ]


def selection(design):
    return {
        "settings": {method: stage.candidates(design, method)[0] for method in stage.METHODS},
        "reference": "dbscan",
        "physics_methods": ["legacy_b1", "fourier", "dbscan"],
    }


def evaluation_cells(design):
    cells = []
    for pattern in design["scope"]["confirmation"]:
        methods = {
            method: {"metrics": {"f1": 0.75, "precision": 0.9}}
            for method in stage.METHODS
        }
        methods["fourier"]["metrics"].update(f1=0.9, precision=0.9)
        physics = {
            method: {"reliable": True, "relative_error": {"ell": 0.3}, "matrix_excess": 0.02}
            for method in ("legacy_b1", "fourier", "dbscan")
        }
        physics["fourier"].update(relative_error={"ell": -0.1}, matrix_excess=0.01)
        cells.append({"pattern": pattern, "methods": methods, "physics": physics,
                      "common_matched_radius": {"difference": -0.05}})
    return cells


def test_splits_partition_the_data_and_have_stable_identity(design, neural_design):
    n, index = 10000, 7
    splits = stage.split_indices(n, index, design, neural_design)
    repeated = stage.split_indices(n, index, design, neural_design)
    for name in splits:
        np.testing.assert_array_equal(splits[name], repeated[name])
    observed = np.sort(np.concatenate([splits["fitting"], splits["held_back"]]))
    np.testing.assert_array_equal(np.sort(np.concatenate([observed, splits["removed"]])), np.arange(n))
    assert len(np.unique(np.concatenate([observed, splits["removed"]]))) == n
    np.testing.assert_array_equal(
        np.sort(np.concatenate([splits["inner_fit"], splits["inner_validation"]])), splits["fitting"]
    )
    assert not np.intersect1d(splits["inner_fit"], splits["inner_validation"]).size
    assert not np.intersect1d(splits["fitting"], splits["held_back"]).size
    assert all(len(values) for values in splits.values())
    assert stage.array_hash(*splits.values()) == stage.array_hash(*repeated.values())
    other = stage.split_indices(n, index + 1, design, neural_design)
    assert stage.array_hash(*splits.values()) != stage.array_hash(*other.values())


def test_splits_cannot_depend_on_labels_or_global_random_state(design, neural_design):
    assert set(inspect.signature(stage.split_indices).parameters) == {"n", "index", "design", "neural_design"}
    splits = stage.split_indices(1000, 3, design, neural_design)
    labels = np.zeros(1000)
    fitted_before = labels[splits["fitting"]].copy()
    labels[splits["held_back"]] = 1
    labels[splits["removed"]] = 1
    # A prior implementation used the acceptance labels in preprocessing. The
    # protocol exposes only fitting indices to preprocessing, with no label input.
    np.testing.assert_array_equal(labels[splits["fitting"]], fitted_before)
    state = np.random.get_state()
    try:
        np.random.seed(999)
        repeated = stage.split_indices(1000, 3, design, neural_design)
    finally:
        np.random.set_state(state)
    for name in splits:
        np.testing.assert_array_equal(splits[name], repeated[name])


def test_observation_loader_keeps_held_labels_out_of_every_fitting_input(
        tmp_path, monkeypatch, design, neural_design):
    """Changing every held label must not change fitting data or its constants."""
    index, n = 3, 1000
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0, 60, (n, 3))
    coords = {axis: xyz[:, i] for i, axis in enumerate("xyz")}
    centres = {axis: np.array([30.0]) for axis in "xyz"}
    radii = np.array([3.0])
    labels = np.where(rng.random(n) < 0.2, 2, 0)
    monkeypatch.setattr(stage.paths, "DIFFUSION_DIR", tmp_path)
    monkeypatch.setattr(stage.gd, "FROZEN_DIR", tmp_path)
    monkeypatch.setattr(stage, "field_design", lambda _: neural_design)

    def write_pattern():
        np.savez(tmp_path / f"clust_pattern_{index}.npz", coords=coords, labels=labels,
                 radii=radii, centers=centres)
        fingerprint = stage.gd.content_hash(coords, labels, radii, centres)
        (tmp_path / "dataset_checksums.json").write_text(json.dumps({str(index): fingerprint}))

    write_pattern()
    before, observations, before_checks = stage.load_observations(index, design)
    split = observations[2]
    withheld = np.concatenate([split["held_back"], split["removed"]])
    labels[withheld] = 2 - labels[withheld]
    write_pattern()
    after, _, after_checks = stage.load_observations(index, design)
    for name in ("fit_x", "fit_y", "validation_x", "validation_y", "observed_x", "observed_y"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert before.fit_rate == after.fit_rate
    assert before.observed_rate == after.observed_rate
    assert before_checks["split_hash"] == after_checks["split_hash"]
    assert before_checks["observed_input_hash"] == after_checks["observed_input_hash"]
    assert before_checks["dataset_hash"] != after_checks["dataset_hash"]


def test_field_checkpoint_preserves_predictions_and_encoding(tmp_path):
    architecture = {"fourier_features": 4, "hidden": [5, 3]}
    model = implicit.build_field("fourier", 17, base=np.arange(12).reshape(4, 3) / 13,
                                 sigma=2.75, hidden=(5, 3), box=13.0)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.5, 3.1, 7.0], [13.0, 13.0, 13.0]])
    expected = model(points).detach().clone()
    path = tmp_path / "field.pt"
    stage.save_field_checkpoint(path, model, {"architecture": architecture})
    restored = stage.load_field_checkpoint(path)
    assert not restored.training
    assert restored.hidden == (5, 3)
    torch.testing.assert_close(restored(points), expected, rtol=0, atol=0)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)


def test_field_transfer_scores_only_withheld_atoms_against_their_oracle(tmp_path):
    # Coincident fitting atoms give an analytically constant Gaussian fraction of
    # 3/4, while the saved Fourier model is initialized to a distinct constant 1/4.
    x = np.array([[2, 2, 2]] * 4 + [[2.1, 2, 2], [2, 2.1, 2],
                                  [2, 2, 2.1], [2.1, 2.1, 2], [2, 2.1, 2.1]], dtype=float)
    y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1], dtype=float)
    p_star = np.array([0.01, 0.99, 0.1, 0.9, 0.1, 0.8, 0.25, 0.6, 0.9])
    split = {"fitting": np.arange(4), "held_back": np.array([4, 5]), "removed": np.array([6, 7, 8])}
    data = stage.s2.FittingData(0, 0.37, x[:3].copy(), y[:3].copy(), x[3:4].copy(), y[3:4].copy(),
                               x[:4].copy(), y[:4].copy(), 1.0, 0.75, "synthetic-split")
    architecture = {"fourier_features": 2, "hidden": [3]}
    model = implicit.build_field("fourier", 13, base=np.ones((2, 3)), sigma=0.75,
                                 hidden=(3,), box=4.0).initialise_constant(0.25)
    stage.save_field_checkpoint(tmp_path / "field.pt", model, {"architecture": architecture})
    scoring_design = {"grid": 0.5, "box": 4.0, "device": "cpu"}
    result = stage.field_scores(data, (x, y, split), {"probability": p_star},
                                {"bandwidth": 0.75}, tmp_path, scoring_design)

    def bce(labels, probabilities):
        return -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))

    assert set(result) == {"held_back", "removed"}
    assert sum(entry["atoms"] for entry in result.values()) == len(x) - len(split["fitting"])
    for name in ("held_back", "removed"):
        ids = split[name]
        expected_oracle = bce(y[ids], p_star[ids])
        assert result[name]["atoms"] == len(ids)
        assert result[name]["oracle_loss"] == pytest.approx(expected_oracle)
        assert set(result[name]["methods"]) == {"constant", "b1", "fourier"}
        for method, q in (("constant", 0.75), ("b1", 0.75), ("fourier", 0.25)):
            expected_loss = bce(y[ids], q)
            score = result[name]["methods"][method]
            assert score["loss"] == pytest.approx(expected_loss, abs=1e-6)
            assert score["excess_loss"] == pytest.approx(expected_loss - expected_oracle, abs=1e-6)
            assert score["mean_squared_probability_error"] == pytest.approx(np.mean((q - p_star[ids]) ** 2), abs=1e-6)

    # Changing fitting entries in the scoring arrays cannot influence either
    # held-out score or the already fitted model/smoother inputs.
    changed_y, changed_truth = y.copy(), p_star.copy()
    changed_y[split["fitting"]] = 1 - changed_y[split["fitting"]]
    changed_truth[split["fitting"]] = np.nan
    repeated = stage.field_scores(data, (x, changed_y, split), {"probability": changed_truth},
                                  {"bandwidth": 0.75}, tmp_path, scoring_design)
    assert repeated == result


def test_field_transfer_moves_restored_model_to_requested_device(tmp_path, monkeypatch):
    """Exercise MPS placement without requiring a GPU in the test environment."""
    x = np.full((4, 3), 1.5)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    split = {"fitting": np.array([0, 1]), "held_back": np.array([2]), "removed": np.array([3])}
    data = stage.s2.FittingData(0, 0.37, x[:1], y[:1], x[1:2], y[1:2], x[:2], y[:2], 0.0, 0.5, "test")
    moves, predictions = [], []

    class RestoredField:
        device = "cpu"

        def to(self, device):
            moves.append(device)
            self.device = device
            return self

    restored = RestoredField()
    monkeypatch.setattr(stage, "load_field_checkpoint", lambda _: restored)

    def predict(models, points, device):
        assert models == [restored]
        assert restored.device == device == "mps"
        predictions.append(len(points))
        return np.full(len(points), 0.25)

    monkeypatch.setattr(stage.s2, "predict_procedure", predict)
    result = stage.field_scores(data, (x, y, split), {"probability": np.full(4, 0.5)},
                                {"bandwidth": 0.75}, tmp_path, {"grid": 0.5, "box": 3.0, "device": "mps"})
    assert moves == ["mps"]
    assert predictions == [1, 1]
    assert result["held_back"]["methods"]["fourier"]["loss"] == pytest.approx(-np.log(0.75))
    assert result["removed"]["methods"]["fourier"]["loss"] == pytest.approx(-np.log(0.25))


def test_settings_are_selected_globally_not_per_pattern(design):
    cells = calibration_cells(design)
    cells[0]["candidates"]["b1"][0]["metrics"]["f1"] = 0.95
    cells[1]["candidates"]["b1"][0]["metrics"]["f1"] = 0.05
    for cell in cells:
        cell["candidates"]["dbscan"][1]["metrics"]["f1"] = 0.8
        cell["candidates"]["fourier"][1]["metrics"]["f1"] = 0.99
    chosen = stage.select_settings(cells, design)
    assert chosen["settings"]["b1"] == stage.candidates(design, "b1")[1]
    assert chosen["reference"] == "dbscan"  # Fourier cannot select itself as its comparator.
    assert chosen["physics_methods"] == ["legacy_b1", "fourier", "dbscan"]
    assert chosen == stage.select_settings(list(reversed(cells)), design)


def test_selection_uses_precision_radius_and_candidate_order_for_ties(design):
    cells = calibration_cells(design)
    for cell in cells:
        cell["candidates"]["b1"][1]["metrics"].update(precision=0.9, relative_radius_mae=0.5)
        cell["candidates"]["fourier"][1]["metrics"]["relative_radius_mae"] = 0.1
    chosen = stage.select_settings(cells, design)
    assert chosen["settings"]["b1"] == stage.candidates(design, "b1")[1]
    assert chosen["settings"]["fourier"] == stage.candidates(design, "fourier")[1]
    assert chosen["settings"]["dbscan"] == stage.candidates(design, "dbscan")[0]
    assert chosen["reference"] == "b1"
    tied = stage.select_settings(calibration_cells(design), design)
    assert tied["reference"] == "legacy_b1"
    assert tied["physics_methods"] == ["legacy_b1", "fourier"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "failed"])
def test_selection_requires_exact_successful_calibration_scope(design, defect):
    cells = calibration_cells(design)
    if defect == "missing":
        cells.pop()
    elif defect == "duplicate":
        cells.append(copy.deepcopy(cells[0]))
    elif defect == "unexpected":
        cells[0]["pattern"] = 999
    else:
        cells[0]["error"] = "fit failed"
    with pytest.raises(ValueError):
        stage.select_settings(cells, design)


@pytest.mark.parametrize("defect", ["missing", "extra", "reordered", "failed"])
def test_selection_requires_complete_correctly_labelled_candidate_grid(design, defect):
    cells = calibration_cells(design)
    entries = cells[0]["candidates"]["fourier"]
    if defect == "missing":
        entries.pop()
    elif defect == "extra":
        entries.append(copy.deepcopy(entries[0]))
    elif defect == "reordered":
        entries.reverse()
    else:
        entries[0] = {"setting": entries[0]["setting"], "error": "non-finite training"}
    with pytest.raises(ValueError):
        stage.select_settings(cells, design)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "error", "method", "physics"])
def test_completeness_keeps_every_required_case_and_method(design, defect):
    cells = evaluation_cells(design)
    if defect == "missing":
        cells.pop()
    elif defect == "duplicate":
        cells.append(copy.deepcopy(cells[0]))
    elif defect == "unexpected":
        cells.append({"pattern": 999, "error": "unexpected"})
    elif defect == "error":
        cells[0]["error"] = "failure must remain in denominator"
    elif defect == "method":
        del cells[0]["methods"]["b1"]
    else:
        del cells[0]["physics"]["dbscan"]
    checked = stage.completeness(cells, design["scope"]["confirmation"], selection(design))
    assert not checked["complete"]
    assert not stage.promotion(cells, design, selection(design))["passed"]


@pytest.mark.parametrize("group", ["methods", "physics"])
def test_completeness_rejects_unexpected_methods(design, group):
    cells = evaluation_cells(design)
    cells[0][group]["unregistered_method"] = {}
    checked = stage.completeness(cells, design["scope"]["confirmation"], selection(design))
    assert not checked["complete"]


def test_promotion_requires_detection_geometry_physics_and_reliability(design):
    cells = evaluation_cells(design)
    chosen = selection(design)
    result = stage.promotion(cells, design, chosen)
    assert result["complete"] and result["passed"] and result["verdict"] == "passed"
    invalid = copy.deepcopy(cells)
    invalid[0]["physics"]["fourier"]["reliable"] = False
    result = stage.promotion(invalid, design, chosen)
    assert result["complete"] and not result["passed"]
    assert not result["conditions"]["all_fourier_fits_reliable"]["passed"]
    missing_physics = copy.deepcopy(cells)
    del missing_physics[0]["physics"]["fourier"]["relative_error"]
    assert not stage.promotion(missing_physics, design, chosen)["passed"]
    unmatched = copy.deepcopy(cells)
    unmatched[0]["common_matched_radius"]["difference"] = None
    assert not stage.promotion(unmatched, design, chosen)["passed"]


@pytest.mark.parametrize("defect", ["f1", "precision", "radius", "ell", "matrix"])
def test_promotion_does_not_trade_away_any_prespecified_condition(design, defect):
    cells = evaluation_cells(design)
    for cell in cells:
        if defect == "f1":
            cell["methods"]["fourier"]["metrics"]["f1"] = 0.75
        elif defect == "precision":
            cell["methods"]["fourier"]["metrics"]["precision"] = 0.7
        elif defect == "radius":
            cell["common_matched_radius"]["difference"] = 0.0
        elif defect == "ell":
            cell["physics"]["fourier"]["relative_error"]["ell"] = -0.4
        else:
            cell["physics"]["fourier"]["matrix_excess"] = 0.03
    assert not stage.promotion(cells, design, selection(design))["passed"]


def test_reserved_test_recomputes_decision_instead_of_trusting_saved_boolean(design):
    cells = evaluation_cells(design)
    chosen = selection(design)
    # A stale failure cannot forbid a complete successful report; a stale success
    # cannot unlock the test once the underlying evidence is missing or failed.
    stage.require_test_permission({"cells": cells, "decision": {"passed": False}}, design, chosen)
    with pytest.raises(RuntimeError, match="reserved test refused"):
        stage.require_test_permission({"cells": cells[:1], "decision": {"passed": True}}, design, chosen)
    cells[0]["physics"]["fourier"]["reliable"] = False
    with pytest.raises(RuntimeError, match="reserved test refused"):
        stage.require_test_permission({"cells": cells, "decision": {"passed": True}}, design, chosen)


def test_resume_requires_identical_provenance(tmp_path):
    path = tmp_path / "result.json"
    report = {"fingerprint": "original", "cells": [{"pattern": 0}]}
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    assert stage.read_compatible(path, "original") == report
    with pytest.raises(RuntimeError, match="provenance changed"):
        stage.read_compatible(path, "changed")
    assert path.read_bytes() == before


def test_common_radius_error_compares_the_same_particles():
    a = {"matched_true_indices": [0, 1], "relative_radius_error": [-0.1, 0.9]}
    b = {"matched_true_indices": [0, 2], "relative_radius_error": [0.2, 0.01]}
    result = stage.common_radius_errors(a, b)
    assert result["count"] == 1
    assert result["fourier"] == pytest.approx(0.1)
    assert result["reference"] == pytest.approx(0.2)
    assert result["difference"] == pytest.approx(-0.1)
    assert stage.common_radius_errors(a, {"matched_true_indices": [], "relative_radius_error": []})["difference"] is None
