"""Tests for clustersim's overlying-pattern oversampling.

``opp_oversample`` exists so that the reconstruction benchmark can place cluster
centres at random rather than on a lattice. It must not change a single draw
for the existing datasets, which leave it at its default.
"""

import numpy as np
import pytest

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
