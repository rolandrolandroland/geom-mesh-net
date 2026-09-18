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

from geom_mesh_net.inference.calibration import (
    central_interval_coverage,
    ecdf_deviation,
    kolmogorov_band,
    sbc_ranks,
)
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES


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
    from geom_mesh_net.simulation.parameters import PRIOR_HIGH, PRIOR_LOW

    assert len(PARAMETER_NAMES) == len(PRIOR_LOW) == len(PRIOR_HIGH)


# --------------------------------------------------------------------------
# Width ratio — the diagnostic Stage 3's original gate lacked
# --------------------------------------------------------------------------


def test_width_ratio_is_one_for_a_calibrated_posterior():
    from geom_mesh_net.inference.calibration import width_ratio

    samples, truth = build(1.0, 1.0, seed=20)
    assert width_ratio(samples, truth)[0] == pytest.approx(1.0, abs=0.08)


def test_width_ratio_exceeds_one_when_overconfident():
    from geom_mesh_net.inference.calibration import width_ratio

    samples, truth = build(1.0, 0.5, seed=21)
    assert width_ratio(samples, truth)[0] > 1.5


def test_width_ratio_falls_below_one_when_underconfident():
    from geom_mesh_net.inference.calibration import width_ratio

    samples, truth = build(1.0, 2.0, seed=22)
    assert width_ratio(samples, truth)[0] < 0.7


def test_robust_and_sd_ratios_separate_typical_behaviour_from_outliers():
    """The two estimators answer different questions, and both are reported.

    Construct a posterior that is correctly wide for 95% of patterns and badly
    wrong for the other 5%. The sd-based ratio is inflated by those few, while
    the robust ratio reports that typical behaviour is fine. Neither is the
    "right" number: the *gap* between them is the diagnostic, and it says a small
    number of patterns carry enormous error rather than the posterior being
    uniformly too narrow.

    This is the distinction the ensemble run turned on. Over all 1,000 patterns
    the sd-based ratio for `rho_c` was 1.38; excluding two zero-cluster patterns
    it was 0.97.
    """
    from geom_mesh_net.inference.calibration import central_interval_coverage, width_ratio

    rng = np.random.default_rng(23)
    truth = rng.normal(0.0, 1.0, size=(N_PATTERNS, 1))
    heavy = rng.random((N_PATTERNS, 1)) < 0.05
    # Centre is correct for most patterns, wildly off for a few.
    centre = truth + np.where(
        heavy,
        rng.normal(0.0, 6.0, size=(N_PATTERNS, 1)),
        rng.normal(0.0, 0.05, size=(N_PATTERNS, 1)),
    )
    samples = centre[:, None, :] + rng.normal(
        0.0, 1.0, size=(N_PATTERNS, N_SAMPLES, 1)
    )

    robust = width_ratio(samples, truth, robust=True)[0]
    fragile = width_ratio(samples, truth, robust=False)[0]
    coverage = central_interval_coverage(samples, truth, 0.9)[0]

    assert robust < 0.5, robust              # typical patterns: posterior ample
    assert fragile > 1.0, fragile            # tails inflate the sd-based figure
    assert fragile > 3 * robust, (fragile, robust)   # the gap is the signal
    assert coverage > 0.85, coverage


def test_robust_sd_matches_ordinary_sd_for_clean_gaussian_data():
    """The 1.4826 scaling has to make the two agree when there are no outliers,
    or the robust ratio would not be comparable to 1.0."""
    from geom_mesh_net.inference.calibration import robust_sd

    rng = np.random.default_rng(30)
    values = rng.normal(0.0, 2.0, size=(8000, 1))
    assert robust_sd(values)[0] == pytest.approx(2.0, rel=0.06)


def test_robust_sd_ignores_a_small_contaminated_fraction():
    from geom_mesh_net.inference.calibration import robust_sd

    rng = np.random.default_rng(31)
    values = rng.normal(0.0, 1.0, size=(8000, 1))
    values[:160] = rng.normal(0.0, 50.0, size=(160, 1))   # 2% contamination
    assert robust_sd(values)[0] == pytest.approx(1.0, rel=0.10)
    assert values.std() > 3.0        # the ordinary sd is wrecked by them


def test_width_ratio_is_per_parameter():
    """One well-calibrated parameter must not mask an overconfident one."""
    from geom_mesh_net.inference.calibration import width_ratio

    rng = np.random.default_rng(24)
    truth = rng.normal(0.0, 1.0, size=(N_PATTERNS, 2))
    # Posterior independent of the truth, as in `build`: the residual spread is
    # then the truth's spread, and the ratio is truth_sd / posterior_sd.
    samples = np.stack(
        [
            rng.normal(0.0, 1.0, size=(N_PATTERNS, N_SAMPLES)),   # ratio ~ 1.0
            rng.normal(0.0, 0.3, size=(N_PATTERNS, N_SAMPLES)),   # ratio ~ 3.3
        ],
        axis=2,
    )
    ratios = width_ratio(samples, truth)
    assert ratios[0] == pytest.approx(1.0, abs=0.15), ratios
    assert ratios[1] > 2.5, ratios


def test_width_ratio_reads_one_for_a_calibrated_posterior_of_any_shape():
    """Both sides of the ratio must use the same estimator of spread.

    A correctly calibrated uniform posterior is as wide as its own errors, so the
    ratio is one. Measuring the numerator with a scaled median absolute deviation
    and the denominator with a standard deviation would read 1.4826 * MAD / sd,
    about 1.28 for a uniform posterior, and call a correct posterior overconfident.
    """
    from geom_mesh_net.inference.calibration import robust_sd, width_ratio

    rng = np.random.default_rng(31)
    centre = rng.normal(0.0, 1.0, size=(N_PATTERNS, 1))
    draws = centre[:, None, :] + rng.uniform(-1.0, 1.0, size=(N_PATTERNS, N_SAMPLES, 1))
    truth = centre + rng.uniform(-1.0, 1.0, size=(N_PATTERNS, 1))   # the truth is a draw from the posterior

    assert width_ratio(draws, truth, robust=True)[0] == pytest.approx(1.0, abs=0.08)
    assert width_ratio(draws, truth, robust=False)[0] == pytest.approx(1.0, abs=0.08)

    mismatched = robust_sd(draws.mean(axis=1) - truth, axis=0) / np.median(draws.std(axis=1), axis=0)
    assert mismatched[0] > 1.2      # the form used until 2026-09-17
