"""Tests for the classical guest-field estimators of reconstruction Stage 1.

The binned Nadaraya–Watson estimator is checked against the kernel sum it
approximates. Its limits are checked against closed forms, and the scoring
identity the whole track relies on is checked exactly.
"""

import numpy as np
import pytest

from geom_mesh_net.core_functions import field_baselines as fb


def brute_force_nadaraya_watson(coords, guest, points, h):
    d2 = ((points[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    w = np.exp(-0.5 * d2 / h ** 2)
    return (w * guest[None, :]).sum(axis=1) / w.sum(axis=1)


@pytest.fixture
def clustered_points():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 20, size=(6000, 3))
    centre = np.array([10.0, 10.0, 10.0])
    p = np.where(np.linalg.norm(coords - centre, axis=1) < 4.0, 0.7, 0.05)
    guest = rng.random(len(coords)) < p
    return coords, guest


def test_binned_estimator_matches_the_kernel_sum_it_approximates(clustered_points):
    coords, guest = clustered_points
    rng = np.random.default_rng(1)
    points = rng.uniform(7, 13, size=(300, 3))  # far from the box faces
    h = 1.5
    counts = fb.SmoothedCounts(coords, guest, bandwidths=(h,), grid=0.25, lower=0.0, upper=20.0)
    binned = counts.fixed(points, 0)
    exact = brute_force_nadaraya_watson(coords, guest.astype(float), points, h)
    assert np.median(np.abs(binned - exact)) < 0.005
    assert np.max(np.abs(binned - exact)) < 0.03


def test_all_guests_gives_probability_one_everywhere():
    coords = np.random.default_rng(2).uniform(0, 20, size=(2000, 3))
    counts = fb.SmoothedCounts(coords, np.ones(2000, dtype=bool), bandwidths=(1.0,), lower=0.0, upper=20.0)
    assert np.allclose(counts.fixed(coords[:100], 0), 1.0)


def test_a_very_wide_kernel_approaches_the_global_fraction(clustered_points):
    coords, guest = clustered_points
    counts = fb.SmoothedCounts(coords, guest, bandwidths=(200.0,), grid=1.0, lower=0.0, upper=20.0)
    assert np.allclose(counts.fixed(coords[:200], 0), guest.mean(), atol=0.01)


def test_adaptive_prediction_at_a_grid_bandwidth_equals_the_fixed_prediction(clustered_points):
    coords, guest = clustered_points
    counts = fb.SmoothedCounts(coords, guest, lower=0.0, upper=20.0)
    points = coords[:500]
    for i, h in enumerate(fb.BANDWIDTHS):
        assert np.allclose(counts.adaptive(points, np.full(len(points), h)), counts.fixed(points, i), atol=1e-6)


def test_adaptive_prediction_lies_between_its_neighbouring_bandwidths(clustered_points):
    coords, guest = clustered_points
    counts = fb.SmoothedCounts(coords, guest, lower=0.0, upper=20.0)
    points = coords[:500]
    mid = np.sqrt(fb.BANDWIDTHS[2] * fb.BANDWIDTHS[3])  # halfway between 1.5 and 2 in log bandwidth
    q = counts.adaptive(points, np.full(len(points), mid))
    lo, hi = counts.fixed(points, 2), counts.fixed(points, 3)
    assert np.all(q >= np.minimum(lo, hi) - 1e-6) and np.all(q <= np.maximum(lo, hi) + 1e-6)


def test_guest_neighbour_distances_skip_the_query_point_itself():
    guests = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    d = fb.guest_neighbour_distances(guests, guests[:1], k_max=2)
    assert np.allclose(d, [[1.0, 3.0]])


def test_cross_validation_picks_a_narrow_kernel_for_a_sharp_cluster_and_a_wide_one_for_noise():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 30, size=(30000, 3))
    sharp = np.linalg.norm(coords - 15.0, axis=1) < 6.0
    guest_sharp = rng.random(len(coords)) < np.where(sharp, 0.9, 0.02)
    guest_flat = rng.random(len(coords)) < 0.05
    fit_sharp = fb.fit_baselines(coords, guest_sharp, folds=3, upper=30.0)
    fit_flat = fb.fit_baselines(coords, guest_flat, folds=3, upper=30.0)
    assert fit_sharp.bandwidth < fit_flat.bandwidth


def test_expected_log_loss_is_entropy_plus_kl_divergence():
    # The primary metric's identity (ROADMAP F5), checked in closed form.
    for p, q in [(0.1, 0.3), (0.5, 0.5), (0.93, 0.8)]:
        expected = -(p * np.log(q) + (1 - p) * np.log(1 - q))
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        assert expected == pytest.approx(entropy + kl)
    assert fb.log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
