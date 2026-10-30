"""Tests for Gate 5.1's matrix check.

The check asks whether matrix labels follow the diffusion field. Under the field, both of
its z-scores are sums of many independent terms standardised by their exact mean and
variance, so they must behave like standard normal draws. And labels drawn from a uniform
matrix with the same mean must fail it, or the check proves nothing.
"""

import numpy as np

from experiments.reconstruction.stage5_simulator import matrix_check


def smooth_field(n=100_000, seed=0):
    """Values like a Stage 5 matrix: mean near 0.08, varying by a few hundredths."""
    x = np.random.default_rng(seed).uniform(0, 60, size=n)
    return 0.08 + 0.03 * np.sin(x / 4)


def test_labels_drawn_from_the_field_give_standard_normal_scores():
    c = smooth_field()
    rng = np.random.default_rng(1)
    scores = np.array([[r["matrix_fraction_z"], r["log_likelihood_gain_z"]]
                       for r in (matrix_check(c, rng.random(len(c)) < c) for _ in range(300))])
    assert np.all(np.abs(scores.mean(axis=0)) < 4 / np.sqrt(len(scores)))
    assert np.all((scores.std(axis=0) > 0.85) & (scores.std(axis=0) < 1.15))


def test_labels_from_a_uniform_matrix_fail_the_check():
    c = smooth_field()
    result = matrix_check(c, np.random.default_rng(2).random(len(c)) < c.mean())
    assert result["log_likelihood_gain_z"] < -10
    assert not result["matrix_within_3se"]
