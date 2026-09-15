"""Tests for the replay oracle of clustersim's guest probabilities.

Every reconstruction in ``experiments/reconstruction/`` is scored against this
oracle. If it is wrong, every score is wrong in the same direction, and nothing
downstream can detect it. The grid it replaces was measured wrong inside clusters, so here
the oracle is checked against things that cannot share its mistakes:

- exact enumeration of successive sampling for a small population;
- the simulator's own labelling functions, replayed thousands of times.
"""

import itertools

import numpy as np
import pytest

from geom_mesh_net.fields import oracle as fo
from geom_mesh_net.simulation import clustersim as csim


def enumerate_successive_sampling(weights, n_selected):
    """Exact inclusion probabilities, by summing over every ordered sequence of picks."""
    weights = np.asarray(weights, dtype=float)
    inclusion = np.zeros(len(weights))
    for sequence in itertools.permutations(range(len(weights)), n_selected):
        remaining, probability = weights.sum(), 1.0
        for i in sequence:
            probability *= weights[i] / remaining
            remaining -= weights[i]
        inclusion[list(sequence)] += probability
    return inclusion


SMALL_WEIGHTS = np.array([0.05, 0.2, 0.35, 0.5, 0.8, 1.0])


# --------------------------------------------------------------------------
# Successive sampling
# --------------------------------------------------------------------------


def test_enumeration_sums_to_the_number_of_picks():
    exact = enumerate_successive_sampling(SMALL_WEIGHTS, 3)
    assert exact.sum() == pytest.approx(3.0)
    assert np.all(np.diff(exact) > 0)  # heavier atoms are picked more often


def test_exponential_clocks_reproduce_exact_inclusion_probabilities():
    exact = enumerate_successive_sampling(SMALL_WEIGHTS, 3)
    replays = 200_000
    estimated = fo.inclusion_probabilities(SMALL_WEIGHTS, 3, replays=replays,
                                           rng=np.random.default_rng(0), smooth=False)
    se = np.sqrt(exact * (1 - exact) / replays)
    assert np.all(np.abs(estimated - exact) < 4 * se)


def test_numpy_choice_without_replacement_is_successive_sampling():
    # clustersim calls rng.choice(replace=False, p=w). The oracle's clocks are only
    # valid if that call has the same distribution as successive sampling.
    exact = enumerate_successive_sampling(SMALL_WEIGHTS, 3)
    rng = np.random.default_rng(1)
    p = SMALL_WEIGHTS / SMALL_WEIGHTS.sum()
    replays = 40_000
    counts = np.zeros(len(p))
    for _ in range(replays):
        counts[rng.choice(len(p), size=3, replace=False, p=p)] += 1
    se = np.sqrt(exact * (1 - exact) / replays)
    assert np.all(np.abs(counts / replays - exact) < 4 * se)


def test_equal_weights_give_simple_random_sampling_exactly():
    pi = fo.inclusion_probabilities(np.ones(40), 7, rng=np.random.default_rng(0))
    assert np.array_equal(pi, np.full(40, 7 / 40))


def test_smoothed_probabilities_are_monotone_and_keep_the_expected_count():
    rng = np.random.default_rng(2)
    weights = rng.uniform(0, 1, size=500)
    pi = fo.inclusion_probabilities(weights, 180, replays=300, rng=rng, smooth=True)
    order = np.argsort(weights)
    assert np.all(np.diff(pi[order]) >= -1e-12)
    assert pi.sum() == pytest.approx(180.0)


def test_isotonic_fit_matches_a_hand_worked_example():
    fitted = fo.isotonic_increasing([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    assert np.allclose(fitted, [1.0, 2.5, 2.5, 3.5, 3.5, 5.0])
    assert np.array_equal(fo.isotonic_increasing([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------
# The simulator, replayed
# --------------------------------------------------------------------------


def simulator_guests(X, centres, radii, rho_c, rho_b):
    """One labelling of fixed atoms and geometry, by clustersim's own functions and loop."""
    labels = np.zeros(len(X), dtype=np.int8)
    for centre, radius in zip(centres, radii):
        dists = ((((X[:, 0] - centre[0]) ** 2) + ((X[:, 1] - centre[1]) ** 2)
                  + ((X[:, 2] - centre[2]) ** 2)) ** (1 / 2))
        weighted = csim.calc_weighted_dist(X[:, 0], X[:, 1], X[:, 2], center=np.asarray(centre))
        new = csim.assign_clust_points(dist=dists, weighted_dist=weighted, rho_c=rho_c, r_max=radius,
                                       r_weighted_max=None, prob_function="Gaussian_decay",
                                       prob_exp=-3, selection="sampled")
        labels = np.maximum(labels, new)
    background = np.where(labels == 0)[0]
    labels[background] = csim.gen_back_guest(labels[background], rho_b, assign_val=3)
    return np.isin(labels, (2, 3))


def replay_frequencies(X, centres, radii, rho_c, rho_b, replays):
    total = np.zeros(len(X))
    for _ in range(replays):
        total += simulator_guests(X, centres, radii, rho_c, rho_b)
    return total / replays


@pytest.fixture
def seeded_simulator(monkeypatch):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(12345))


def test_overlapping_clusters_combine_as_a_union_not_a_maximum(seeded_simulator):
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 16, size=(6000, 3))
    centres, radii = np.array([[6.0, 8.0, 8.0], [10.0, 8.0, 8.0]]), np.array([4.0, 4.0])
    rho_c, rho_b, replays = 0.4, 0.03, 3000

    oracle = fo.replay_oracle(X, centres, radii, rho_c, rho_b, replays=4000, seed=4)
    freq = replay_frequencies(X, centres, radii, rho_c, rho_b, replays)

    overlap = oracle.n_spheres == 2
    assert overlap.sum() > 50
    per_cluster = np.column_stack([
        np.where(fo.point_distances(X, c) <= r, prof.at(fo.point_distances(X, c)), 0.0)
        for c, r, prof in zip(centres, radii, oracle.profiles)
    ])
    union_error = freq[overlap] - oracle.p[overlap]
    max_error = freq[overlap] - per_cluster[overlap].max(axis=1)
    se_mean = np.sqrt(np.mean(oracle.p[overlap] * (1 - oracle.p[overlap])) / replays / overlap.sum())
    assert abs(union_error.mean()) < 4 * se_mean + 0.002
    assert max_error.mean() > 0.02  # the maximum rule misses guests the simulator makes


def test_oracle_matches_the_simulator_atom_by_atom_on_a_clustersim_pattern(seeded_simulator):
    np.random.seed(0)  # estimate_cluster_volume draws from the global generator
    domain = {axis: np.array([0.0, 20.0]) for axis in "xyz"}
    points, labels = csim.gen_rand_points(intensity=1, dim1=20, dim2=20, dim3=20)
    upp = csim.PointPattern3(points, domain=domain, labels=labels)
    opp_points, opp_labels = csim.gen_rand_points(intensity=1, dim1=10, dim2=10, dim3=10)
    opp = csim.PointPattern3(opp_points, domain={a: np.array([0.0, 10.0]) for a in "xyz"}, labels=opp_labels)
    rho_c, rho_b = 0.5, 0.02
    pattern, radii, centres = csim.clustersim(opp=opp, upp=upp, pcp=0.1, rho_c=rho_c, rho_b=rho_b,
                                              cr=3.0, rb=0.2, cut="buffered", buffer_factor=1,
                                              opp_oversample=8)
    X = np.column_stack([pattern.coords[a] for a in "xyz"])
    C = np.column_stack([centres[a] for a in "xyz"])
    assert len(C) >= 5

    # Unsmoothed, so the oracle's own Monte Carlo variance is known exactly and can
    # be counted: both estimates are binomial frequencies of the same probability.
    oracle_replays, replays = 50_000, 1500
    oracle = fo.replay_oracle(pattern.coords, centres, radii, rho_c, rho_b, replays=oracle_replays,
                              seed=5, smooth=False)
    freq = replay_frequencies(X, C, radii, rho_c, rho_b, replays)

    inside = oracle.inside
    p = oracle.p[inside]
    variance = np.maximum(p * (1 - p), 1e-6) * (1 / replays + 1 / oracle_replays)
    z2 = (freq[inside] - p) ** 2 / variance
    assert inside.sum() > 300
    # Mean z^2 is 1 when the deviations are sampling noise; its sd here is about 0.04.
    assert 0.85 < z2.mean() < 1.2
    assert abs(np.mean(freq[inside] - p)) < 0.003
    assert np.allclose(oracle.p[~inside], rho_b)


def test_smoothing_reduces_monte_carlo_error_in_a_large_cluster():
    rng = np.random.default_rng(13)
    d = 12.0 * rng.uniform(0, 1, size=4000) ** (1 / 3)  # radial distances of atoms uniform in a ball
    weights = fo.selection_weights(d)
    n = int(round(len(d) * 0.4))
    reference = fo.inclusion_probabilities(weights, n, replays=40_000, rng=np.random.default_rng(14), smooth=False)
    raw = fo.inclusion_probabilities(weights, n, replays=300, rng=np.random.default_rng(15), smooth=False)
    smoothed = fo.inclusion_probabilities(weights, n, replays=300, rng=np.random.default_rng(15), smooth=True)
    assert np.mean((smoothed - reference) ** 2) < 0.2 * np.mean((raw - reference) ** 2)
    assert abs(np.mean(smoothed - reference)) < 1e-3


# --------------------------------------------------------------------------
# Limits with closed forms
# --------------------------------------------------------------------------


def test_no_clusters_leaves_every_atom_at_the_matrix_concentration():
    X = np.random.default_rng(6).uniform(0, 10, size=(200, 3))
    oracle = fo.replay_oracle(X, np.empty((0, 3)), np.empty(0), rho_c=0.5, rho_b=0.04)
    assert np.array_equal(oracle.p, np.full(200, 0.04))
    assert not oracle.inside.any()


def test_zero_radius_cluster_is_ignored():
    X = np.random.default_rng(7).uniform(0, 10, size=(200, 3))
    oracle = fo.replay_oracle(X, np.array([[5.0, 5.0, 5.0]]), np.array([0.0]), rho_c=0.5, rho_b=0.04)
    assert np.array_equal(oracle.p, np.full(200, 0.04))


def test_full_concentration_makes_every_atom_inside_a_guest():
    X = np.random.default_rng(8).uniform(0, 10, size=(400, 3))
    oracle = fo.replay_oracle(X, np.array([[5.0, 5.0, 5.0]]), np.array([3.0]), rho_c=1.0, rho_b=0.04)
    assert np.all(oracle.p[oracle.inside] == 1.0)
    assert np.all(oracle.p[~oracle.inside] == 0.04)


def test_an_atom_inside_a_sphere_is_never_a_background_guest():
    # One atom inside, rho_c = 0.3: round(0.3) = 0 picks, and the matrix rule does
    # not apply inside a sphere, so the probability is 0, not rho_b.
    X = np.array([[5.0, 5.0, 5.5], [9.0, 9.0, 9.0]])
    oracle = fo.replay_oracle(X, np.array([[5.0, 5.0, 5.0]]), np.array([1.0]), rho_c=0.3, rho_b=0.04)
    assert oracle.p[0] == 0.0 and oracle.p[1] == 0.04


def test_constant_selection_gives_the_picked_fraction():
    X = np.random.default_rng(9).uniform(0, 10, size=(2000, 3))
    oracle = fo.replay_oracle(X, np.array([[5.0, 5.0, 5.0]]), np.array([3.0]), rho_c=0.35, rho_b=0.01,
                              prob_function="constant")
    n_inside = int(oracle.inside.sum())
    assert np.allclose(oracle.p[oracle.inside], round(n_inside * 0.35) / n_inside)


def test_field_at_the_atoms_reproduces_the_atom_probabilities():
    rng = np.random.default_rng(10)
    X = rng.uniform(0, 20, size=(4000, 3))
    centres, radii = np.array([[7.0, 10.0, 10.0], [12.0, 10.0, 10.0], [15.0, 15.0, 4.0]]), np.array([4.0, 3.5, 2.5])
    oracle = fo.replay_oracle(X, centres, radii, rho_c=0.45, rho_b=0.02, replays=500, seed=11)
    assert np.allclose(oracle.field(X), oracle.p, atol=1e-12)


def test_field_between_atoms_is_bounded_by_the_neighbouring_profile():
    X = np.random.default_rng(12).uniform(0, 12, size=(3000, 3))
    oracle = fo.replay_oracle(X, np.array([[6.0, 6.0, 6.0]]), np.array([4.0]), rho_c=0.4, rho_b=0.02, replays=500)
    radial = np.column_stack([np.linspace(0, 5.5, 200), np.full(200, 6.0), np.full(200, 6.0)])
    radial[:, 0] += 6.0
    values = oracle.field(radial)
    d = radial[:, 0] - 6.0
    assert np.all(np.diff(values[d <= 4.0]) <= 1e-12)  # non-increasing outward inside the sphere
    assert np.all(values[d > 4.0] == 0.02)


def test_rejects_unsupported_profiles():
    with pytest.raises(ValueError, match="prob_function"):
        fo.replay_oracle(np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1), 0.5, 0.01, prob_function="inverse")
