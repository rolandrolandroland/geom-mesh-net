"""Tests for the Stage 2 field: implementation facts only.

Whether a larger σ recovers edges better, or whether refitting helps, are questions about data
and belong to the development pilot (``experiments/reconstruction/pilot/check_fourier_field.py``).
These tests pin what the code must do regardless: coordinates scaled by the known box,
frequencies scaled by σ and saved with the model, an exact constant at update 0, the best
checkpoint restored, and a split that is reproducible and disjoint.
"""

import io
import math

import numpy as np
import pytest
import torch

from geom_mesh_net.neural import implicit

BOX = implicit.BOX_LENGTH


def one_frequency_model(frequency, sigma=1.0, hidden=()):
    return implicit.FourierFeatureField(np.asarray([frequency], dtype=np.float32), sigma, hidden=hidden)


def test_coordinates_are_divided_by_the_box_and_features_have_the_right_size():
    model = one_frequency_model([1.0, 0.0, 0.0])
    x = torch.tensor([[0.0, 0.0, 0.0], [BOX / 4, 5.0, 7.0]])
    features = model.encode(x)
    assert features.shape == (2, 2)                       # one frequency: a sine and a cosine
    assert torch.allclose(features[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(features[1], torch.tensor([1.0, 0.0]), atol=1e-6)   # 2π · (15 / 60) = π / 2

    base = implicit.base_frequencies(256, seed=0)
    assert implicit.FourierFeatureField(base, 6.0).encode(torch.zeros(3, 3)).shape == (3, 512)
    raw = implicit.RawCoordinateField(hidden=(8,))
    corners = raw.encode(torch.tensor([[0.0, BOX / 2, BOX]]))
    assert torch.allclose(corners, torch.tensor([[-1.0, 0.0, 1.0]]))


def test_sigma_scales_the_shared_base_frequencies():
    base = implicit.base_frequencies(16, seed=3)
    slow, fast = implicit.FourierFeatureField(base, 3.0), implicit.FourierFeatureField(base, 6.0)
    assert torch.allclose(fast.frequencies(), 2 * slow.frequencies())
    x = torch.rand(20, 3) * BOX / 2
    assert torch.allclose(fast.encode(x), slow.encode(2 * x), atol=1e-4)   # z = 2π σ B0 · x / L


def test_the_same_seed_gives_the_same_base_frequencies_and_weights():
    assert np.array_equal(implicit.base_frequencies(8, 5), implicit.base_frequencies(8, 5))
    assert not np.array_equal(implicit.base_frequencies(8, 5), implicit.base_frequencies(8, 6))
    base = implicit.base_frequencies(8, 5)
    a = implicit.build_field("fourier", 11, base=base, sigma=3.0, hidden=(16,))
    b = implicit.build_field("fourier", 11, base=base, sigma=6.0, hidden=(16,))
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name                  # σ candidates start from the same weights


def test_frequency_buffers_survive_save_and_load():
    trained = implicit.build_field("fourier", 1, base=implicit.base_frequencies(32, 1), sigma=12.0, hidden=(16, 16))
    buffer = io.BytesIO()
    torch.save(trained.state_dict(), buffer)
    buffer.seek(0)
    restored = implicit.build_field("fourier", 2, base=implicit.base_frequencies(32, 9), sigma=3.0, hidden=(16, 16))
    restored.load_state_dict(torch.load(buffer))
    assert torch.equal(restored.base_frequencies, trained.base_frequencies)
    assert float(restored.sigma) == 12.0
    x = np.random.default_rng(0).uniform(0, BOX, (50, 3))
    assert np.array_equal(implicit.predict(restored, x), implicit.predict(trained, x))


@pytest.mark.parametrize("rate", [0.137, 0.0, 1.0])
def test_initialisation_is_exactly_the_constant(rate):
    model = implicit.build_field("fourier", 4, base=implicit.base_frequencies(64, 4), sigma=24.0)
    model.initialise_constant(rate)
    x = np.random.default_rng(1).uniform(0, BOX, (500, 3))
    p = implicit.predict(model, x)
    expected = min(max(rate, implicit.RATE_CLIP), 1 - implicit.RATE_CLIP)
    assert np.all(np.isfinite(implicit.predict_logits(model, x)))
    assert np.allclose(p, expected, rtol=1e-5, atol=1e-7)
    assert p.max() - p.min() == 0.0                       # constant, not merely close


def test_logits_and_probabilities_are_separate_paths():
    model = implicit.build_field("raw", 0, hidden=(16, 16))
    x = torch.rand(40, 3) * BOX
    with torch.no_grad():
        assert torch.allclose(model(x), torch.sigmoid(model.logits(x)))
    points = x.numpy()
    assert np.allclose(implicit.predict(model, points), 1 / (1 + np.exp(-implicit.predict_logits(model, points))),
                       atol=1e-6)


def test_chunked_prediction_matches_one_pass():
    """Agreement to float32 precision: a matrix product's rounding depends on the batch's shape."""
    model = implicit.build_field("fourier", 3, base=implicit.base_frequencies(32, 3), sigma=6.0, hidden=(16,))
    x = np.random.default_rng(2).uniform(0, BOX, (1003, 3))
    assert np.allclose(implicit.predict(model, x, chunk=7), implicit.predict(model, x, chunk=100_000),
                       rtol=0, atol=1e-6)


def test_batches_cover_each_pass_before_repeating():
    stream = implicit._BatchStream(10, 4, seed=0)
    drawn = torch.cat([stream.next() for _ in range(5)])  # 20 atoms: exactly two passes of 10
    assert sorted(drawn[:10].tolist()) == list(range(10))
    assert sorted(drawn[10:].tolist()) == list(range(10))
    assert implicit._BatchStream(10, 32, seed=0).next() is None     # the whole set fits: full batch


def test_the_best_checkpoint_is_restored_and_update_zero_is_a_candidate():
    """Validation labels anti-correlated with the fitting labels: every update makes validation worse."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, BOX, (4000, 3))
    inside = np.linalg.norm(x - BOX / 2, axis=1) < 15
    fit_y = (rng.random(4000) < np.where(inside, 0.9, 0.1)).astype(float)
    val_x = rng.uniform(0, BOX, (1000, 3))
    val_inside = np.linalg.norm(val_x - BOX / 2, axis=1) < 15
    val_y = (rng.random(1000) < np.where(val_inside, 0.1, 0.9)).astype(float)
    model = implicit.build_field("fourier", 0, base=implicit.base_frequencies(32, 0), sigma=3.0, hidden=(32,))
    model.initialise_constant(fit_y.mean())
    record = implicit.fit_field(model, x, fit_y, val_x, val_y, learning_rate=1e-2, max_updates=100, eval_every=10,
                                patience=50)
    assert record["best_update"] == 0
    assert record["updates_run"] > 0
    assert np.allclose(implicit.predict(model, val_x), fit_y.mean(), atol=1e-6)   # restored to the constant


def test_a_short_fit_learns_a_simple_field():
    """A sphere filling 15% of the box: the fit must close half the constant-to-truth gap."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0, BOX, (12000, 3))
    p = np.where(np.linalg.norm(x - BOX / 2, axis=1) < 20, 0.9, 0.05)
    y = (rng.random(12000) < p).astype(float)
    model = implicit.build_field("fourier", 0, base=implicit.base_frequencies(64, 0), sigma=1.5, hidden=(64, 64))
    model.initialise_constant(y[:9000].mean())
    record = implicit.fit_field(model, x[:9000], y[:9000], x[9000:], y[9000:], learning_rate=3e-3,
                                max_updates=300, eval_every=25, patience=300)
    truth = float(-np.mean(y[9000:] * np.log(p[9000:]) + (1 - y[9000:]) * np.log(1 - p[9000:])))
    gap = record["initial_validation_bce"] - truth
    assert gap > 0.1
    assert record["best_validation_bce"] < record["initial_validation_bce"] - 0.5 * gap
    assert record["history"][-1]["fit_bce"] < record["history"][0]["fit_bce"]
    assert np.isfinite(record["mean_residual"]) and not record["nonfinite"]


def test_refit_length_distinguishes_full_batch_from_minibatch():
    from experiments.reconstruction import stage2_field as s2
    full = {"best_update": 400, "best_passes": 400.0}            # 17,280 fitting atoms, one batch per update
    assert s2.refit_updates(full, 21_600, 32_768, "updates") == 400
    assert s2.refit_updates(full, 21_600, 32_768, "passes") == 400          # still full batch: same steps
    mini = {"best_update": 400, "best_passes": 400 * 32_768 / 63_936}      # minibatches in both fits
    assert s2.refit_updates(mini, 79_920, 32_768, "passes") == math.ceil(400 * 79_920 / 63_936)


def test_random_features_approximate_the_gaussian_kernel():
    """The features' mean inner product is exp(−2π²σ²|Δu|²): a Gaussian of width L / (2πσ).

    It is a property of the features, not of the trained network, whose effective kernel differs.
    """
    sigma, n = 6.0, 40_000
    model = implicit.FourierFeatureField(implicit.base_frequencies(n, 7), sigma, hidden=())
    width = BOX / (2 * math.pi * sigma)
    origin = torch.tensor([[30.0, 30.0, 30.0]])
    with torch.no_grad():
        f0 = model.encode(origin)
        for distance in (0.0, 0.5 * width, width, 2 * width):
            f1 = model.encode(origin + torch.tensor([[distance, 0.0, 0.0]]))
            estimate = float((f0 * f1).sum()) / n
            exact = math.exp(-distance ** 2 / (2 * width ** 2))
            assert abs(estimate - exact) < 0.02, (distance, estimate, exact)   # sampling sd is about 0.004


DATA = pytest.importorskip("experiments.reconstruction.benchmark").DATA_DIR


@pytest.mark.skipif(not (DATA / "clust_pattern_4.npz").exists(), reason="benchmark patterns not present")
def test_the_split_is_reproducible_disjoint_and_uses_only_observed_atoms():
    from experiments.reconstruction import stage2_field as s2
    design = s2.load_design()
    first, scoring, checks = s2.prepare_cell(4, 0.1, design)
    again, _, _ = s2.prepare_cell(4, 0.1, design)
    assert checks["leakage"] == "passed"
    assert first.split_hash == again.split_hash and np.array_equal(first.fit_x, again.fit_x)
    n_observed = len(first.observed_x)
    assert len(first.fit_x) + len(first.validation_x) == n_observed
    assert n_observed + len(scoring.removed) == len(scoring.all_x)
    assert abs(len(first.validation_x) / n_observed - design["split"]["validation_fraction"]) < 0.02
    joined = np.concatenate([first.fit_x, first.validation_x])
    assert np.array_equal(np.unique(joined, axis=0), np.unique(first.observed_x, axis=0))
    removed = {tuple(row) for row in scoring.all_x[scoring.removed[:2000]]}
    assert not removed & {tuple(row) for row in first.observed_x}
    assert first.fit_rate == pytest.approx(first.fit_y.mean())
    other_efficiency, _, _ = s2.prepare_cell(4, 0.37, design)
    assert other_efficiency.split_hash != first.split_hash


def test_the_gradient_penalty_refuses_mps():
    """Input gradients through the field are unreliable on MPS (Stage 2 pilot, 2026-09-18)."""
    model = implicit.build_field("raw", 0, hidden=(8,))
    x = np.zeros((4, 3))
    with pytest.raises(ValueError, match="MPS"):
        implicit.fit_field(model, x, np.zeros(4), gradient_penalty=1e-3, device="mps", max_updates=1)
