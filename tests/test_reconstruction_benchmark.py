"""Tests for the frozen reconstruction benchmark.

A benchmark is only a benchmark if it cannot drift. These tests pin the three
things every stage depends on: the splits partition the patterns, a thinning
mask depends on nothing but its pattern and efficiency, and the masks still match
the checksums frozen before any model was scored.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.reconstruction import benchmark as bm

N_ATOMS = 216_000


def test_splits_partition_all_patterns_without_overlap():
    seen = []
    for name in bm.SPLITS:
        seen.extend(bm.split_indices(name))
    assert sorted(seen) == list(range(1000))
    assert bm.split_of(99) == "development" and bm.split_of(900) == "test"


def test_bands_cover_the_prior_including_its_upper_edges():
    assert bm.band(3.0, bm.CR_BANDS) == 0 and bm.band(15.0, bm.CR_BANDS) == 2
    assert bm.band(0.5, bm.RHO_C_BANDS) == 1 and bm.band(1.0, bm.RHO_C_BANDS) == 1
    with pytest.raises(ValueError):
        bm.band(2.9, bm.CR_BANDS)


def test_mask_depends_only_on_pattern_and_efficiency():
    first = bm.thinning_mask(905, 0.37, N_ATOMS)
    _ = bm.thinning_mask(3, 0.8, N_ATOMS)  # an unrelated draw in between
    assert np.array_equal(first, bm.thinning_mask(905, 0.37, N_ATOMS))
    assert not np.array_equal(first, bm.thinning_mask(906, 0.37, N_ATOMS))
    assert not np.array_equal(bm.thinning_mask(905, 0.1, N_ATOMS), bm.thinning_mask(905, 0.8, N_ATOMS))


@pytest.mark.parametrize("eta", bm.EFFICIENCIES)
def test_retained_fraction_is_binomial_around_the_efficiency(eta):
    kept = bm.thinning_mask(950, eta, N_ATOMS).mean()
    assert abs(kept - eta) < 4 * np.sqrt(eta * (1 - eta) / N_ATOMS)


def test_masks_match_the_frozen_checksums():
    frozen = json.loads((bm.BENCHMARK_DIR / "mask_checksums.json").read_text())
    for key in ("0:0.1", "57:0.37", "812:0.8", "999:0.1"):
        index, eta = key.split(":")
        assert bm.mask_checksum(bm.thinning_mask(int(index), float(eta), N_ATOMS)) == frozen[key]


def test_stage2_subset_is_six_test_patterns_per_stratum():
    theta = bm.load_theta()
    subset = bm.stage2_subset(theta)
    assert len(subset) == len(bm.CR_BANDS) * len(bm.RHO_C_BANDS)
    for cell, indices in subset.items():
        assert len(indices) == 6
        assert all(bm.split_of(i) == "test" for i in indices)
        c, r = int(cell[2]), int(cell[-1])
        assert all(bm.band(theta[i, 2], bm.CR_BANDS) == c and bm.band(theta[i, 0], bm.RHO_C_BANDS) == r
                   for i in indices)


def test_headroom_requires_both_the_fraction_and_the_absolute_margin():
    assert bm.has_headroom(l_const=0.10, l_method=0.035, l_oracle=0.02)       # 19% open, 0.015 nats
    assert not bm.has_headroom(l_const=0.30, l_method=0.045, l_oracle=0.02)   # 0.025 nats, but only 9% open
    assert not bm.has_headroom(l_const=0.03, l_method=0.029, l_oracle=0.02)   # 90% open, but only 0.009 nats
