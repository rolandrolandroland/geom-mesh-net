"""Instrumented Fourier fits preserve the existing learning/selection procedure."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.reconstruction import stage2_field as s2
from geom_mesh_net.neural import implicit
from geom_mesh_net.neural.field_diagnostics import (
    fit_field_with_artifacts, fit_fourier_with_artifacts, load_checkpoint, load_fourier_artifacts,
    score_particle_signal, training_failure_flags,
)


@pytest.fixture(autouse=True)
def limited_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    yield
    torch.set_num_threads(before)


def sample_data():
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 60, (64, 3))
    y = (x[:, 0] < 30).astype(float)
    return x[:48], y[:48], x[48:], y[48:]


def build():
    return implicit.build_field("fourier", 22, base=implicit.base_frequencies(4, 31), sigma=1,
                                hidden=(8,)).initialise_constant(.4)


@pytest.mark.parametrize("batch,updates,patience", [(16, 12, 100), (100, 12, 1), (16, 0, 10)])
def test_periodic_artifacts_leave_selected_weights_and_history_unchanged(tmp_path, batch, updates, patience):
    arrays = sample_data()
    original, instrumented = build(), build()
    options = dict(batch_size=batch, max_updates=updates, eval_every=2, patience=patience,
                   batch_seed=11, learning_rate=.01)
    expected = implicit.fit_field(original, *arrays, **options)
    actual = fit_field_with_artifacts(instrumented, *arrays, **options, artifact_dir=tmp_path,
                                      periodic_every=3, metadata={"test": "equivalence"})
    assert expected["history"] == actual["history"]
    for key in ("best_update", "best_validation_bce", "updates_run", "passes_run", "stopped", "mean_residual"):
        assert actual[key] == expected[key]
    for name, value in original.state_dict().items():
        assert torch.equal(value, instrumented.state_dict()[name])
    restored, metadata = load_checkpoint(actual["artifacts"]["selected"]["path"],
                                         expected_sha256=actual["artifacts"]["selected"]["sha256"])
    assert metadata["role"] == "selected"
    np.testing.assert_array_equal(implicit.predict(restored, arrays[0]), implicit.predict(original, arrays[0]))
    assert actual["artifacts"]["periodic"][0]["update"] == 0
    assert (tmp_path / "history.json").exists()
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(actual["artifacts"]["selected"]["path"], expected_sha256="changed")


def test_refit_adds_history_without_changing_weights_or_number_of_passes(tmp_path):
    x, y, _, _ = sample_data()
    original, instrumented = build(), build()
    options = dict(batch_size=17, max_updates=9, eval_every=2, batch_seed=7)
    expected = implicit.fit_field(original, x, y, **options)
    actual = fit_field_with_artifacts(instrumented, x, y, **options, artifact_dir=tmp_path,
                                      periodic_every=3)
    assert actual["best_update"] == expected["best_update"] == 9
    assert actual["best_passes"] == expected["best_passes"]
    assert [r["update"] for r in actual["history"]] == [0, 2, 4, 6, 8, 9]
    assert actual["history"][0] == expected["history"][0]
    assert actual["history"][-1] == expected["history"][-1]
    for name, value in original.state_dict().items():
        assert torch.equal(value, instrumented.state_dict()[name])
    assert actual["artifacts"]["selected"]["role"] == "final_refit"


def test_full_candidate_selection_and_alternate_refit_preserve_primary(tmp_path):
    x, y, vx, vy = sample_data()
    data = s2.FittingData(9999, .37, x, y, vx, vy, np.concatenate([x, vx]), np.concatenate([y, vy]),
                          float(y.mean()), float(np.concatenate([y, vy]).mean()), "test-split")
    design = s2.load_design()
    design["architecture"].update(fourier_features=4, hidden=[8])
    design["sigma_grid"] = [.75, 1.5]
    design["training"].update(max_updates=10, eval_every=2, patience=100, learning_rate=.01, batch_size=16)
    expected_models, expected_records = s2.fit_model(data, copy.deepcopy(design), "fourier", "cpu")
    models, records = fit_fourier_with_artifacts(data, design, artifact_dir=tmp_path,
                                                periodic_every=4, compare_passes=True)
    for name, value in expected_models[0].state_dict().items():
        assert torch.equal(value, models[0].state_dict()[name])
    entry = records[0]
    assert entry["selected"]["sigma"] == expected_records[0]["selected"]["sigma"]
    assert len(entry["candidates"]) == 2
    assert entry["refit"]["rule"] == "updates"
    assert entry["diagnostic_refit"]["rule"] == "passes"
    expected_passes = s2.refit_updates(entry["selected"], len(data.observed_x), 16, "passes")
    assert entry["diagnostic_refit"]["updates_run"] == expected_passes
    assert entry["diagnostic_refit"]["updates_run"] >= entry["refit"]["updates_run"]
    assert training_failure_flags(records) == {"primary": [], "diagnostic": []}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for relative, checksum in manifest["artifact_hashes"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == checksum
    for candidate in entry["candidates"]:
        assert candidate["history"][0]["update"] == 0
        assert candidate["artifacts"]["selected"]["sha256"]
    restored, _ = load_checkpoint(entry["prerefit_artifact"]["path"])
    assert float(restored.sigma) == entry["selected"]["sigma"]
    # Final is the primary same-update model, regardless of the diagnostic refit's fit loss.
    restored_final, _ = load_checkpoint(entry["final_artifact"]["path"])
    np.testing.assert_array_equal(implicit.predict(restored_final, x), implicit.predict(expected_models[0], x))
    cached, _ = load_fourier_artifacts(data, design, artifact_dir=tmp_path)
    np.testing.assert_array_equal(implicit.predict(cached[0], x), implicit.predict(models[0], x))
    changed_data = copy.deepcopy(data)
    changed_data.observed_y[0] = 1 - changed_data.observed_y[0]
    with pytest.raises(ValueError, match="observed inputs changed"):
        load_fourier_artifacts(changed_data, design, artifact_dir=tmp_path)
    # A corrupted unselected candidate must invalidate cache reuse too.
    candidate_path = tmp_path / "seed_0" / "candidate_00" / "terminal.pt"
    with candidate_path.open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="artifact changed"):
        load_fourier_artifacts(data, design, artifact_dir=tmp_path)


def test_artifacts_cannot_silently_overwrite_existing_evidence(tmp_path):
    x, y, vx, vy = sample_data()
    fit_field_with_artifacts(build(), x, y, vx, vy, artifact_dir=tmp_path, max_updates=0)
    with pytest.raises(FileExistsError, match="not empty"):
        fit_field_with_artifacts(build(), x, y, vx, vy, artifact_dir=tmp_path, max_updates=0)


def test_invalid_validation_pair_is_rejected_before_writing(tmp_path):
    x, y, vx, _ = sample_data()
    with pytest.raises(ValueError, match="supplied together"):
        fit_field_with_artifacts(build(), x, y, vx, None, artifact_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_particle_signal_compares_direct_fields_and_keeps_empty_particle_and_observed_counts():
    points = np.array([[0, 0, 0], [.1, 0, 0], [2, 2, 2], [4, 4, 4], [10, 10, 10]], dtype=float)
    labels = np.array([1, 1, 0, 0, 1])
    oracle = np.array([.8, .8, .1, .1, .1])
    calls = []
    def recorded(values):
        calls.append(values.copy())
        return oracle.copy()
    fields = {"gaussian": recorded, "selected_prerefit": lambda p: np.full(len(p), .3),
              "final_refit": lambda p: np.full(len(p), .1)}
    result = score_particle_signal(fields, points, labels, [[0, 0, 0], [20, 20, 20]], [1, 2],
                                   p_star=oracle, observed_coords=points[:3], observed_labels=labels[:3])
    np.testing.assert_array_equal(calls[0], points)
    small, empty = result["particles"]
    assert small["observed_atom_count"] == small["observed_guest_count"] == 2
    assert small["observed_guest_fraction"] == 1
    assert small["oracle_particle_matrix_contrast"] == pytest.approx(.7)
    assert small["fields"]["gaussian"]["oracle_mse"] == 0
    assert small["fields"]["selected_prerefit"]["probability_max"] == .3
    assert small["fields"]["final_refit"]["probability_max"] == .1
    assert empty["atom_count"] == empty["observed_atom_count"] == 0
    assert empty["fields"]["final_refit"]["bce"] is None
    assert result["matrix"]["atom_count"] == 3
    assert result["whole"]["atom_count"] == 5
    assert result["purpose"] == "scoring_only_never_selection"
    json.dumps(result, allow_nan=False)


def test_scoring_invalid_predictions_are_visible_not_filtered_and_truth_overlap_rejected():
    points = np.array([[0, 0, 0], [.1, 0, 0], [3, 3, 3]])
    result = score_particle_signal({"bad": lambda p: np.array([np.nan, .5, .3])}, points,
                                   [1, 1, 0], [[0, 0, 0]], [1])
    bad_particle = result["particles"][0]["fields"]["bad"]
    assert bad_particle["invalid_probability_count"] == 1
    assert bad_particle["bce"] is None
    assert result["matrix"]["fields"]["bad"]["bce"] is not None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="must not overlap"):
        score_particle_signal({"zero": lambda p: np.zeros(len(p))}, points, [1, 1, 0],
                               [[0, 0, 0], [.5, 0, 0]], [1, 1])


def test_training_failures_keep_unselected_candidate_and_optional_refit_separate():
    good = {"nonfinite": False, "stopped": "fixed", "history": [{"fit_bce": .2}]}
    candidate_bad = {"nonfinite": True, "stopped": "nonfinite", "history": [{"fit_bce": .2}]}
    diagnostic_bad = {"nonfinite": False, "stopped": "fixed", "history": [{"fit_bce": None}]}
    flags = training_failure_flags([{"candidates": [good, candidate_bad], "selected": good,
                                     "refit": good, "diagnostic_refit": diagnostic_bad}])
    assert flags["primary"] == [{"record": "seed_0/candidate_1", "reasons": ["nonfinite_or_missing_status"]}]
    assert flags["diagnostic"] == [{"record": "seed_0/diagnostic_refit", "reasons": ["invalid_history_loss"]}]
