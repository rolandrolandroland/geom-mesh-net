"""Tests for the Stage 3 calibration diagnostics.

Stage 3 is the gate the whole project turns on, so the diagnostic itself has to
be validated before its verdict means anything. Each test constructs a case
whose correct answer is known by construction:

  calibrated     posterior samples drawn from the same law as the truth
  overconfident  posterior too narrow -- the dangerous failure
  underconfident posterior too wide
  biased         posterior centred away from the truth

A diagnostic that passed only the calibrated case would be useless; what matters
is that it *rejects* the other three.
"""

import numpy as np
import pytest

from inference.recover_ground_truth import PARAMETER_NAMES
from inference.validate_posterior import (
    central_interval_coverage,
    ecdf_deviation,
    kolmogorov_band,
    sbc_ranks,
)


N_PATTERNS = 2000
N_SAMPLES = 999


def build(truth_sd, posterior_sd, posterior_shift=0.0, seed=0, dim=1):
    """Truth from N(0, truth_sd); posterior from N(truth_shift, posterior_sd)."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, truth_sd, size=(N_PATTERNS, dim))
    samples = rng.normal(
        posterior_shift, posterior_sd, size=(N_PATTERNS, N_SAMPLES, dim)
    )
    return samples, truth


def max_deviation(samples, truth, position=0):
    ranks = sbc_ranks(samples, truth)
    return ecdf_deviation(ranks[:, position], N_SAMPLES)


# --------------------------------------------------------------------------
# The band
# --------------------------------------------------------------------------


def test_kolmogorov_band_shrinks_with_sample_size():
    """The reason Stage 3 cross-validates instead of using 100 test patterns."""
    assert kolmogorov_band(100) == pytest.approx(0.1358, abs=1e-3)
    assert kolmogorov_band(1000) == pytest.approx(0.0429, abs=1e-3)
    assert kolmogorov_band(1000) < kolmogorov_band(100)


def test_kolmogorov_band_widens_for_a_stricter_alpha():
    assert kolmogorov_band(1000, alpha=0.01) > kolmogorov_band(1000, alpha=0.05)


# --------------------------------------------------------------------------
# Ranks
# --------------------------------------------------------------------------


def test_ranks_lie_in_the_valid_range():
    samples, truth = build(1.0, 1.0)
    ranks = sbc_ranks(samples, truth)
    assert ranks.shape == (N_PATTERNS, 1)
    assert ranks.min() >= 0 and ranks.max() <= N_SAMPLES


def test_rank_is_the_count_of_samples_below_the_truth():
    samples = np.array([[[0.0], [1.0], [2.0], [3.0]]])   # (1, 4, 1)
    assert sbc_ranks(samples, np.array([[2.5]]))[0, 0] == 3
    assert sbc_ranks(samples, np.array([[-1.0]]))[0, 0] == 0
    assert sbc_ranks(samples, np.array([[99.0]]))[0, 0] == 4


# --------------------------------------------------------------------------
# The four regimes
# --------------------------------------------------------------------------


def test_calibrated_posterior_gives_uniform_ranks():
    """Samples drawn from the same law as the truth must pass."""
    deviation = max_deviation(*build(1.0, 1.0, seed=1))
    assert deviation <= kolmogorov_band(N_PATTERNS), deviation


def test_overconfident_posterior_is_detected():
    """The failure that matters. A posterior three times too narrow leaves the
    truth in the tails, piling ranks at both ends."""
    deviation = max_deviation(*build(1.0, 0.33, seed=2))
    assert deviation > kolmogorov_band(N_PATTERNS) * 3, deviation


def test_underconfident_posterior_is_detected():
    deviation = max_deviation(*build(1.0, 3.0, seed=3))
    assert deviation > kolmogorov_band(N_PATTERNS) * 3, deviation


def test_biased_posterior_is_detected():
    deviation = max_deviation(*build(1.0, 1.0, posterior_shift=0.7, seed=4))
    assert deviation > kolmogorov_band(N_PATTERNS) * 3, deviation


def test_overconfidence_pushes_ranks_to_the_edges():
    """Not just that it fails, but that it fails in the diagnosable direction."""
    samples, truth = build(1.0, 0.33, seed=5)
    normalized = (sbc_ranks(samples, truth)[:, 0] + 0.5) / (N_SAMPLES + 1)
    edges = ((normalized < 0.05) | (normalized > 0.95)).mean()
    assert edges > 0.3, edges          # a uniform rank would give 0.10


def test_underconfidence_piles_ranks_in_the_centre():
    samples, truth = build(1.0, 3.0, seed=6)
    normalized = (sbc_ranks(samples, truth)[:, 0] + 0.5) / (N_SAMPLES + 1)
    centre = ((normalized > 0.3) & (normalized < 0.7)).mean()
    assert centre > 0.6, centre        # a uniform rank would give 0.40


def test_bias_shifts_the_mean_rank():
    samples, truth = build(1.0, 1.0, posterior_shift=0.7, seed=7)
    normalized = (sbc_ranks(samples, truth)[:, 0] + 0.5) / (N_SAMPLES + 1)
    assert normalized.mean() < 0.4, normalized.mean()   # uniform would give 0.5


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_calibrated_coverage_matches_the_nominal_level():
    samples, truth = build(1.0, 1.0, seed=8)
    for level in (0.5, 0.8, 0.9, 0.95):
        empirical = central_interval_coverage(samples, truth, level)[0]
        assert empirical == pytest.approx(level, abs=0.035), (level, empirical)


def test_overconfident_coverage_falls_short_of_nominal():
    samples, truth = build(1.0, 0.33, seed=9)
    assert central_interval_coverage(samples, truth, 0.9)[0] < 0.6


def test_underconfident_coverage_exceeds_nominal():
    samples, truth = build(1.0, 3.0, seed=10)
    assert central_interval_coverage(samples, truth, 0.5)[0] > 0.8


def test_coverage_is_computed_per_dimension():
    """One well-calibrated dimension must not mask a broken one."""
    rng = np.random.default_rng(11)
    truth = rng.normal(0.0, 1.0, size=(N_PATTERNS, 2))
    samples = np.stack(
        [
            rng.normal(0.0, 1.0, size=(N_PATTERNS, N_SAMPLES)),   # calibrated
            rng.normal(0.0, 0.2, size=(N_PATTERNS, N_SAMPLES)),   # overconfident
        ],
        axis=2,
    )
    empirical = central_interval_coverage(samples, truth, 0.9)
    assert empirical[0] == pytest.approx(0.9, abs=0.04)
    assert empirical[1] < 0.4


def test_ecdf_deviation_is_zero_for_exactly_uniform_ranks():
    ranks = np.arange(N_SAMPLES + 1).reshape(-1, 1)
    assert ecdf_deviation(ranks[:, 0], N_SAMPLES) < 0.01


def test_parameter_names_match_the_posterior_dimension():
    """Guards the alignment every table in Stage 3 depends on."""
    from inference.fit_posterior import PRIOR_HIGH, PRIOR_LOW

    assert len(PARAMETER_NAMES) == len(PRIOR_LOW) == len(PRIOR_HIGH)
