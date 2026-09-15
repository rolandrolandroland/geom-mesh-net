"""Tests for the Stage 4 feature ablation."""

import numpy as np
import pytest

from experiments.inference.ablate_features import FEATURE_FAMILIES, evaluate
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES
from geom_mesh_net.statistics.paper_spatial_features import PAPER_FEATURE_NAMES


def test_families_partition_the_feature_set_exactly():
    """Every feature belongs to exactly one summary function. If this drifts,
    the 'drop family' rows silently stop meaning what they claim."""
    members = [name for group in FEATURE_FAMILIES.values() for name in group]
    assert len(members) == len(set(members)), "a feature appears in two families"
    assert set(members) == set(PAPER_FEATURE_NAMES)


def test_family_names_are_real_features():
    for family, members in FEATURE_FAMILIES.items():
        for name in members:
            assert name in PAPER_FEATURE_NAMES, (family, name)


def test_k_family_is_the_five_k_derived_features():
    assert FEATURE_FAMILIES["K"] == ("Tm", "Rm", "Rdm", "Rddm", "Tdm")


def test_evaluate_returns_the_expected_structure():
    rng = np.random.default_rng(0)
    n = 300
    features = rng.normal(size=(n, 6))
    theta = np.column_stack(
        [
            0.2 + 0.8 * rng.random(n),
            0.05 * rng.random(n),
            3.0 + 12.0 * rng.random(n),
            0.5 * rng.random(n),
        ]
    )
    result = evaluate(
        features, theta, columns=[0, 1, 2], seeds=[0], max_epochs=5
    )
    assert result["n_features"] == 3
    assert set(result["contraction_mean"]) == set(PARAMETER_NAMES)
    assert np.isfinite(result["median_log_prob_mean"])
    for value in result["contraction_mean"].values():
        assert value <= 1.0


def test_informative_features_contract_more_than_noise():
    """The ablation's core assumption: contraction tracks information.

    One target is a clean function of a feature, another is independent noise.
    Fitting on that feature must contract the first and not the second.
    """
    rng = np.random.default_rng(1)
    n = 700
    signal = rng.random(n)
    features = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    theta = np.column_stack(
        [
            0.2 + 0.8 * signal,          # determined by feature 0
            0.05 * rng.random(n),        # independent
            3.0 + 12.0 * rng.random(n),  # independent
            0.5 * rng.random(n),         # independent
        ]
    )
    result = evaluate(features, theta, columns=[0, 1, 2], seeds=[0], max_epochs=150)
    contraction = result["contraction_mean"]
    assert contraction["rho_c"] > 0.5, contraction
    for name in ("rho_b", "cr", "rb"):
        assert contraction[name] < 0.25, (name, contraction[name])


def test_dropping_the_only_informative_feature_hurts():
    """Leave-one-out must show a loss when the dropped feature was load-bearing."""
    rng = np.random.default_rng(2)
    n = 700
    signal = rng.random(n)
    features = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    theta = np.column_stack(
        [
            0.2 + 0.8 * signal,
            0.05 * rng.random(n),
            3.0 + 12.0 * rng.random(n),
            0.5 * rng.random(n),
        ]
    )
    with_signal = evaluate(features, theta, [0, 1, 2], [0], max_epochs=150)
    without = evaluate(features, theta, [1, 2], [0], max_epochs=150)
    assert (
        without["median_log_prob_mean"] < with_signal["median_log_prob_mean"]
    ), (without["median_log_prob_mean"], with_signal["median_log_prob_mean"])
    assert without["contraction_mean"]["rho_c"] < with_signal["contraction_mean"]["rho_c"]
