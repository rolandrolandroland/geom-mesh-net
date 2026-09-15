"""Diagnostics for whether a posterior's error bars are honest.

Simulation-based calibration: for each held-out pattern, the rank of the true
parameter among L posterior samples is uniform on {0, ..., L} when the posterior
is correct and the parameters were drawn from the prior. Departures are
diagnostic:

    peaked in the centre    posteriors too wide (underconfident)
    peaked at both edges    posteriors too narrow (overconfident)
    sloped                  biased

``tests/test_calibration.py`` checks these diagnostics against posteriors built to
be calibrated, overconfident, underconfident and biased, so each is known to
reject the right failures in the right direction.
"""

import numpy as np

ECDF_ALPHA = 0.05


def kolmogorov_band(n, alpha=ECDF_ALPHA):
    """Simultaneous band on an ECDF, from the DKW inequality.

    P(sup|F_n - F| > eps) <= 2 exp(-2 n eps^2), so the band is
    sqrt(ln(2/alpha) / (2n)). It is a *simultaneous* bound, so a single
    excursion anywhere is already evidence against uniformity.
    """
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def sbc_ranks(samples, truth):
    """Rank of each true value within its posterior samples.

    samples: (n, L, d).  truth: (n, d).  Returns (n, d) in {0, ..., L}.
    """
    return (samples < truth[:, None, :]).sum(axis=1)


def ecdf_deviation(ranks, n_samples):
    """Maximum absolute deviation of the rank ECDF from uniform."""
    uniform = np.sort((ranks + 0.5) / (n_samples + 1))
    n = len(uniform)
    empirical = np.arange(1, n + 1) / n
    below = np.abs(empirical - uniform).max()
    above = np.abs(uniform - np.arange(0, n) / n).max()
    return float(max(below, above))


def central_interval_coverage(samples, truth, level):
    """Fraction of patterns whose truth lies in the central `level` interval."""
    lower = np.quantile(samples, (1 - level) / 2, axis=1)
    upper = np.quantile(samples, 1 - (1 - level) / 2, axis=1)
    return ((truth >= lower) & (truth <= upper)).mean(axis=0)


def robust_sd(values, axis=0):
    """Median absolute deviation, scaled to match a Gaussian standard deviation.

    The 1.4826 factor makes this agree with the ordinary sd for Gaussian data
    while ignoring a small fraction of extreme values.
    """
    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def width_ratio(samples, theta, robust=True):
    """Residual spread over typical posterior spread, per parameter.

    One means the posterior is as wide as its own errors. Above one is
    overconfident, below one underconfident. Unlike coverage this cannot be
    satisfied by a narrow interval that happens to sit in the right place.

    ``robust=True`` uses a scaled median absolute deviation rather than a
    standard deviation. This is not a cosmetic choice. Measured on the ensemble
    run, the sd-based ratio for `rho_c` was 1.38 over all 1,000 patterns and 0.97
    with two zero-cluster patterns removed -- two rows in a thousand deciding
    whether the posterior looked badly or perfectly calibrated. A standard
    deviation is dominated by its tails, which is the same defect that made the
    mean log-likelihood useless in Stage 2. Coverage, being a fraction, was
    untroubled by those rows and moved only 0.919 to 0.921.

    The non-robust form is kept for comparison, since a large gap between the two
    is itself a signal that a few patterns carry enormous error.
    """
    residual = samples.mean(axis=1) - theta
    if robust:
        return robust_sd(residual, axis=0) / np.median(
            samples.std(axis=1), axis=0
        )
    return residual.std(axis=0) / samples.std(axis=1).mean(axis=0)
