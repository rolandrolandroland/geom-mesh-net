"""Tests for Stage 0 ground-truth recovery.

The scientific stakes here are unusually high for a test file: the mean cluster
radius ``cr`` and radius spread ``rb`` exist nowhere on disk except as a replay
of the simulator's random number generator. If that replay silently drifts, the
parameters become wrong rather than missing, and every downstream posterior
would be fitted against mislabeled targets without any error being raised.
"""

import json

import numpy as np
import pytest

from experiments.inference.recover_ground_truth import STATS_COLUMNS, verify_replay
from geom_mesh_net import paths
from geom_mesh_net.simulation.parameters import (
    FACTORY_N_SIMS,
    FACTORY_PCP,
    FACTORY_SEED,
    PARAMETER_NAMES,
    replay_factory_draws,
)


DATA_DIR = paths.DATA_DIR
GROUND_TRUTH_DIR = paths.GROUND_TRUTH_DIR


# --------------------------------------------------------------------------
# The replay itself, no data required
# --------------------------------------------------------------------------


def test_replay_is_deterministic():
    first = replay_factory_draws()
    second = replay_factory_draws()
    for name in PARAMETER_NAMES:
        assert np.array_equal(first[name], second[name])


def test_replay_returns_all_four_parameters_at_full_length():
    theta = replay_factory_draws()
    assert set(theta) == set(PARAMETER_NAMES)
    for name in PARAMETER_NAMES:
        assert theta[name].shape == (FACTORY_N_SIMS,)


def test_replayed_values_lie_inside_their_declared_priors():
    theta = replay_factory_draws()
    bounds = {
        "rho_c": (FACTORY_PCP * 2, 1.0),
        "rho_b": (0.0, FACTORY_PCP * 0.5),
        "cr": (3.0, 15.0),
        "rb": (0.0, 0.5),
    }
    for name, (low, high) in bounds.items():
        values = theta[name]
        assert values.min() >= low, (name, values.min())
        assert values.max() <= high, (name, values.max())


def test_draw_order_matters():
    """Reordering the draws changes the values, which is exactly the fragility
    this module exists to guard against."""
    rng = np.random.default_rng(FACTORY_SEED)
    # cr drawn first instead of third
    cr_wrong_order = rng.uniform(low=3.0, high=15.0, size=FACTORY_N_SIMS)
    assert not np.allclose(cr_wrong_order, replay_factory_draws()["cr"])


def test_verify_replay_flags_a_mismatch():
    theta = replay_factory_draws()
    stats = np.zeros((FACTORY_N_SIMS, 9))
    stats[:, STATS_COLUMNS["rho_c_true"]] = theta["rho_c"]
    stats[:, STATS_COLUMNS["rho_b_true"]] = theta["rho_b"] + 1e-6  # corrupted
    stats[:, STATS_COLUMNS["pcp_true"]] = FACTORY_PCP

    checks = verify_replay(theta, stats)
    assert checks["rho_c"]["exact"]
    assert not checks["rho_b"]["exact"]
    assert checks["rho_b"]["max_abs_diff"] == pytest.approx(1e-6)


def test_verify_replay_accepts_an_exact_match():
    theta = replay_factory_draws()
    stats = np.zeros((FACTORY_N_SIMS, 9))
    stats[:, STATS_COLUMNS["rho_c_true"]] = theta["rho_c"]
    stats[:, STATS_COLUMNS["rho_b_true"]] = theta["rho_b"]
    stats[:, STATS_COLUMNS["pcp_true"]] = FACTORY_PCP

    checks = verify_replay(theta, stats)
    assert checks["rho_c"]["exact"] and checks["rho_b"]["exact"]
    assert checks["pcp"]["constant"]
    assert checks["pcp"]["value"] == pytest.approx(FACTORY_PCP)


# --------------------------------------------------------------------------
# Against the real dataset, when it is present
# --------------------------------------------------------------------------

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "pattern_stats.npy").exists(),
    reason="data/pattern_stats.npy not present (data/ is gitignored)",
)

needs_ground_truth = pytest.mark.skipif(
    not (GROUND_TRUTH_DIR / "theta.npy").exists(),
    reason="run python -m experiments.inference.recover_ground_truth first",
)


@needs_data
def test_replay_reproduces_the_saved_parameters_exactly():
    """The Stage 0 gate. Exact equality, not approximate.

    These are the same float64 draws from the same seeded generator, so any
    nonzero difference means the draw sequence changed.
    """
    stats = np.load(DATA_DIR / "pattern_stats.npy")
    checks = verify_replay(replay_factory_draws(), stats)
    assert checks["rho_c"]["max_abs_diff"] == 0.0
    assert checks["rho_b"]["max_abs_diff"] == 0.0


@needs_data
def test_pcp_is_constant_and_therefore_not_a_parameter():
    stats = np.load(DATA_DIR / "pattern_stats.npy")
    saved_pcp = stats[:, STATS_COLUMNS["pcp_true"]]
    assert np.all(saved_pcp == FACTORY_PCP)
    # Exact spread, not std: std of identical float64 values is ~1e-17.
    assert saved_pcp.max() - saved_pcp.min() == 0.0


@needs_ground_truth
def test_persisted_theta_matches_a_fresh_replay():
    """Guards against the on-disk file drifting from the code that made it."""
    stored = np.load(GROUND_TRUTH_DIR / "theta.npy")
    theta = replay_factory_draws()
    expected = np.column_stack([theta[name] for name in PARAMETER_NAMES])
    assert stored.shape == (FACTORY_N_SIMS, 4)
    assert np.array_equal(stored, expected)


@needs_ground_truth
def test_provenance_records_the_priors_and_the_verification():
    provenance = json.loads((GROUND_TRUTH_DIR / "provenance.json").read_text())
    assert provenance["factory_seed"] == FACTORY_SEED
    assert provenance["parameter_names"] == list(PARAMETER_NAMES)
    assert provenance["verification"]["rho_c"]["exact"] is True
    assert provenance["verification"]["rho_b"]["exact"] is True
    assert set(provenance["priors"]) == set(PARAMETER_NAMES)


@needs_ground_truth
def test_descriptors_are_consistent_with_theta():
    descriptors = np.load(GROUND_TRUTH_DIR / "descriptors.npz")
    n = len(descriptors["n_clusters"])
    assert n > 0

    assert descriptors["guest_fraction"].min() >= 0.0
    assert descriptors["guest_fraction"].max() <= 1.0
    # Degeneracy flag must agree with the cluster count that defines it.
    assert np.array_equal(
        descriptors["degenerate"], descriptors["n_clusters"] == 0
    )
    # Radius statistics are NaN exactly where there are no clusters.
    assert np.all(np.isnan(descriptors["radius_mean"][descriptors["degenerate"]]))
    assert np.all(
        np.isfinite(descriptors["radius_mean"][~descriptors["degenerate"]])
    )


@needs_ground_truth
def test_identifiability_predictions_hold():
    """Stage 3 predictions, asserted in advance so they cannot be
    rationalized afterwards. See experiments/inference/ROADMAP.md section 5.
    """
    descriptors = np.load(GROUND_TRUTH_DIR / "descriptors.npz")
    theta = replay_factory_draws()
    n = len(descriptors["n_clusters"])
    ok = ~descriptors["degenerate"]

    # cr is strongly determined by the cluster count.
    cr_correlation = np.corrcoef(
        theta["cr"][:n][ok], np.log(descriptors["n_clusters"][ok])
    )[0, 1]
    assert cr_correlation < -0.85, cr_correlation

    # rb barely shows up in the realized mean radius, so its posterior is
    # expected to stay near its prior.
    rb_correlation = np.corrcoef(
        theta["rb"][:n][ok], descriptors["radius_mean"][ok]
    )[0, 1]
    assert abs(rb_correlation) < 0.15, rb_correlation
