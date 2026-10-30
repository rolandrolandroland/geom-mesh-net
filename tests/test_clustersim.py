"""Tests for clustersim's overlying-pattern oversampling.

``opp_oversample`` exists so that the reconstruction benchmark can place cluster
centres at random rather than on a lattice. It must not change a single draw
for the existing datasets, which leave it at its default.
"""

import numpy as np
import pytest

from geom_mesh_net import paths
from geom_mesh_net.simulation import clustersim as csim

SIDE, OPP_SIDE = 24, 12
PARAMS = dict(pcp=0.1, rho_c=0.5, rho_b=0.02, cr=3.0, rb=0.2)


def simulate(seed, monkeypatch, random_opp=False, **kwargs):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(seed))
    np.random.seed(seed)  # estimate_cluster_volume draws from the global generator
    points, labels = csim.gen_rand_points(intensity=1, dim1=SIDE, dim2=SIDE, dim3=SIDE)
    upp = csim.PointPattern3(points, domain={a: np.array([0.0, float(SIDE)]) for a in "xyz"}, labels=labels)
    make_opp = csim.gen_rand_points if random_opp else csim.gen_uniform_points
    opp_points, opp_labels = make_opp(intensity=1, dim1=OPP_SIDE, dim2=OPP_SIDE, dim3=OPP_SIDE)
    opp = csim.PointPattern3(opp_points, domain={a: np.array([0.0, float(OPP_SIDE)]) for a in "xyz"},
                             labels=opp_labels)
    pattern, radii, centres = csim.clustersim(opp=opp, upp=upp, cut="buffered", buffer_factor=1, **PARAMS, **kwargs)
    return pattern, radii, centres


def intended_cluster_count(seed):
    np.random.seed(seed)
    volume = csim.estimate_cluster_volume(r=PARAMS["cr"], weights={"x": 1, "y": 1, "z": 1},
                                          exponents={"x": 2, "y": 2, "z": 2}, r_max_weighted_ratio=None)
    needed = SIDE ** 3 * (PARAMS["pcp"] - PARAMS["rho_b"]) / (PARAMS["rho_c"] - PARAMS["rho_b"])
    return int(round(needed / (volume * (1 + 3 * PARAMS["rb"] ** 2))))


def test_default_oversampling_changes_nothing(monkeypatch):
    first = simulate(7, monkeypatch)
    second = simulate(7, monkeypatch, opp_oversample=1)
    assert np.array_equal(first[0].labels, second[0].labels)
    assert np.array_equal(first[1], second[1])
    for axis in "xyz":
        assert np.array_equal(first[2][axis], second[2][axis])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_oversampled_random_centres_hit_the_intended_count_off_any_lattice(monkeypatch, seed):
    _, radii, centres = simulate(seed, monkeypatch, random_opp=True, opp_oversample=8)
    assert len(radii) == intended_cluster_count(seed)
    # A lattice repeats coordinate values across centres; random centres do not.
    assert len(np.unique(np.round(centres["x"], 6))) == len(radii)


BENCHMARK_PATTERN = paths.RANDOM_CENTRES_DIR / "clust_pattern_0.npz"


@pytest.mark.skipif(not BENCHMARK_PATTERN.exists(), reason="data_random_centres/ is gitignored and not present")
def test_default_path_reproduces_a_stored_benchmark_pattern(monkeypatch):
    """Options added to clustersim must not move a single draw of its default path.

    generate_random_centres drew benchmark pattern 0 first, from the generator as seeded
    at import, so replaying its calls must return the stored pattern exactly.
    """
    monkeypatch.setattr(csim, "rng", np.random.default_rng(42))
    np.random.seed(0)
    rho_c, rho_b, cr, rb = np.load(paths.THETA_PATH)[0]
    points, labels = csim.gen_rand_points(intensity=1, dim1=60, dim2=60, dim3=60)
    upp = csim.PointPattern3(points, domain={a: np.array([0.0, 60.0]) for a in "xyz"}, labels=labels)
    opp_points, opp_labels = csim.gen_rand_points(intensity=1, dim1=30, dim2=30, dim3=30)
    opp = csim.PointPattern3(opp_points, domain={a: np.array([0.0, 30.0]) for a in "xyz"}, labels=opp_labels)
    pattern, radii, centres = csim.clustersim(
        opp=opp, upp=upp, pcp=0.1, rho_c=rho_c, rho_b=rho_b, cr=cr, rb=rb, cut="buffered", buffer_factor=1,
        selection="sampled", prob_function="Gaussian_decay", opp_oversample=8,
    )
    with np.load(BENCHMARK_PATTERN, allow_pickle=True) as stored:
        assert np.array_equal(pattern.labels, stored["labels"])
        assert np.array_equal(radii, stored["radii"])
        stored_coords, stored_centres = stored["coords"].item(), stored["centers"].item()
        for axis in "xyz":
            assert np.array_equal(pattern.coords[axis], stored_coords[axis])
            assert np.array_equal(np.asarray(centres[axis]), np.asarray(stored_centres[axis]))


# --------------------------------------------------------------------------
# Radius floor and non-overlapping placement (reconstruction Stage 5)
# --------------------------------------------------------------------------


def test_radii_without_a_floor_are_clipped_at_zero_as_before(monkeypatch):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(11))
    radii = csim.draw_radii(5000, 1.0, 1.0)
    expected = np.random.default_rng(11).normal(loc=1.0, scale=1.0, size=5000)
    expected[expected < 0] = 0
    assert np.array_equal(radii, expected)


def test_radii_with_a_floor_follow_the_truncated_normal(monkeypatch):
    from scipy.stats import norm

    monkeypatch.setattr(csim, "rng", np.random.default_rng(12))
    mu, sigma, floor, n = 3.0, 1.0, 2.5, 200_000
    radii = csim.draw_radii(n, mu, sigma, r_min=floor)
    assert radii.min() >= floor
    alpha = (floor - mu) / sigma
    mean = mu + sigma * norm.pdf(alpha) / norm.sf(alpha)
    variance = sigma ** 2 * (1 + alpha * norm.pdf(alpha) / norm.sf(alpha) - (norm.pdf(alpha) / norm.sf(alpha)) ** 2)
    assert abs(radii.mean() - mean) < 4 * np.sqrt(variance / n)


def test_an_unreachable_radius_floor_raises(monkeypatch):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(13))
    with pytest.raises(ValueError, match="r_min"):
        csim.draw_radii(10, 1.0, 1e-9, r_min=2.0)


def surface_gaps(centres, radii):
    distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1))
    gaps = distance - radii[:, None] - radii[None, :]
    return gaps[~np.eye(len(radii), dtype=bool)]


def test_placement_keeps_every_gap_and_every_radius(monkeypatch):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(14))
    candidates = np.random.default_rng(15).uniform(0, 40, size=(4000, 3))
    radii = np.random.default_rng(16).uniform(1.5, 3.5, size=40)
    centres = csim.place_without_overlap(candidates, radii, min_gap=1.0)
    assert centres.shape == (40, 3)
    assert surface_gaps(centres, radii).min() >= 1.0
    # Every centre is a distinct candidate point.
    matches = (np.abs(centres[:, None, :] - candidates[None, :, :]).sum(-1) == 0)
    assert np.all(matches.sum(axis=1) == 1) and len(np.unique(matches.argmax(axis=1))) == 40


def test_placement_raises_instead_of_dropping_clusters(monkeypatch):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(17))
    candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="could not place"):
        csim.place_without_overlap(candidates, np.array([1.0, 1.0]), min_gap=0.0)


SPARSE = dict(pcp=0.05, rho_c=0.5, rho_b=0.02, cr=3.0, rb=0.2)


def test_clustersim_places_the_intended_clusters_without_overlap(monkeypatch):
    monkeypatch.setattr(csim, "rng", np.random.default_rng(18))
    np.random.seed(18)
    side = 36
    points, labels = csim.gen_rand_points(intensity=1, dim1=side, dim2=side, dim3=side)
    upp = csim.PointPattern3(points, domain={a: np.array([0.0, float(side)]) for a in "xyz"}, labels=labels)
    opp_points, opp_labels = csim.gen_rand_points(intensity=1, dim1=12, dim2=12, dim3=12)
    opp = csim.PointPattern3(opp_points, domain={a: np.array([0.0, 12.0]) for a in "xyz"}, labels=opp_labels)
    pattern, radii, centres = csim.clustersim(
        opp=opp, upp=upp, cut="buffered", buffer_factor=1, opp_oversample=64, min_gap=1.0, r_min=1.5, **SPARSE,
    )
    np.random.seed(18)
    volume = csim.estimate_cluster_volume(r=SPARSE["cr"], weights={"x": 1, "y": 1, "z": 1},
                                          exponents={"x": 2, "y": 2, "z": 2}, r_max_weighted_ratio=None)
    intended = int(round(side ** 3 * (SPARSE["pcp"] - SPARSE["rho_b"]) / (SPARSE["rho_c"] - SPARSE["rho_b"])
                         / (volume * (1 + 3 * SPARSE["rb"] ** 2))))
    c = np.column_stack([np.asarray(centres[a]) for a in "xyz"])
    assert len(radii) == intended
    assert radii.min() >= 1.5
    assert surface_gaps(c, radii).min() >= 1.0
    # With no overlaps, every in-cluster atom lies inside exactly one sphere.
    x = np.column_stack([pattern.coords[a] for a in "xyz"])
    inside = (np.sqrt(((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)) <= radii[None, :]).sum(axis=1)
    assert np.all(inside[np.isin(pattern.labels, (1, 2))] == 1)
    assert np.all(inside[np.isin(pattern.labels, (0, 3))] == 0)
