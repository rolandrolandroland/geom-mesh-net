"""Tests for Stage 1 feature extraction.

The gate logic gets most of the attention here. It is the thing standing between
degenerate features and a posterior fitted on noise, so it has to fail when it
should — a gate that always passes is worse than no gate, because it looks like
a check.
"""

import numpy as np
import pytest

from experiments.inference.extract_features import (
    GATE_MINIMUM_FINITE_FRACTION,
    GATE_MINIMUM_INTERIOR_RM_FRACTION,
    discover_patterns,
    extract_one,
    report_gate,
)
from geom_mesh_net.statistics import paper_spatial_features as psf
from geom_mesh_net.statistics.presets import STAGE1_CONFIG, build_config


N_FEATURES = len(psf.PAPER_FEATURE_NAMES)


def gate(values, interior, errors=None):
    total = len(values)
    return report_gate(
        np.asarray(values),
        np.asarray(interior),
        np.arange(total),
        errors if errors is not None else [""] * total,
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_stage1_config_is_valid_and_uses_the_documented_choices():
    config = build_config()
    assert config.null_model == "csr"
    assert config.k_transform == "sqrt"
    assert config.k_r_max == 40.0


def test_stage1_k_r_max_is_within_the_domain():
    """The guard rejects k_r_max above the shortest domain side, which is 60
    for this dataset. 40 must stay under it or Stage 1 cannot run at all."""
    assert build_config().k_r_max < 60.0


def test_stage1_workers_is_one_per_process():
    """Each pattern runs in its own process, so scipy must not also try to
    thread across every core inside each worker."""
    assert STAGE1_CONFIG["workers"] == 1


def test_build_config_accepts_overrides():
    config = build_config({"k_transform": "cube_root", "k_r_max": 20.0})
    assert config.k_transform == "cube_root"
    assert config.k_r_max == 20.0
    assert config.null_model == "csr"  # untouched default


def test_build_config_rejects_an_invalid_override():
    with pytest.raises(ValueError, match="k_transform must be one of"):
        build_config({"k_transform": "log"})


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_gate_passes_on_clean_output():
    result = gate(np.zeros((100, N_FEATURES)), np.ones((100, 3), dtype=bool))
    assert result["passed"]
    assert result["finite_fraction"] == 1.0
    assert result["interior_rm_fraction"] == 1.0


def test_gate_fails_when_too_many_features_are_non_finite():
    values = np.zeros((100, N_FEATURES))
    values[:10, 3] = np.nan          # 10% non-finite, threshold is 5%
    result = gate(values, np.ones((100, 3), dtype=bool))
    assert not result["passed"]
    assert result["finite_fraction"] == pytest.approx(0.90)


def test_gate_fails_when_too_many_rm_values_are_grid_endpoints():
    """The failure mode this whole gate exists for: features that are boundary
    artifacts rather than measurements."""
    interior = np.ones((100, 3), dtype=bool)
    interior[:30, 0] = False         # 70% interior, threshold is 80%
    result = gate(np.zeros((100, N_FEATURES)), interior)
    assert not result["passed"]
    assert result["interior_rm_fraction"] == pytest.approx(0.70)


def test_gate_is_not_fooled_by_rdm_or_rddm():
    """Only Rm is gated. Rddm is known to be weak and must not block Stage 2,
    but it must still be reported."""
    interior = np.ones((100, 3), dtype=bool)
    interior[:, 2] = False           # Rddm entirely absent
    result = gate(np.zeros((100, N_FEATURES)), interior)
    assert result["passed"]
    assert result["interior_rddm_fraction"] == 0.0


def test_gate_sits_exactly_on_its_thresholds():
    n = 100
    finite_failures = int(round(n * (1 - GATE_MINIMUM_FINITE_FRACTION)))
    values = np.zeros((n, N_FEATURES))
    values[:finite_failures, 0] = np.nan
    assert gate(values, np.ones((n, 3), dtype=bool))["passed"]

    interior = np.ones((n, 3), dtype=bool)
    interior[: int(round(n * (1 - GATE_MINIMUM_INTERIOR_RM_FRACTION))), 0] = False
    assert gate(np.zeros((n, N_FEATURES)), interior)["passed"]


def test_gate_counts_errored_patterns():
    errors = [""] * 10
    errors[3] = "ValueError: boom"
    errors[7] = "ValueError: boom"
    result = gate(np.zeros((10, N_FEATURES)), np.ones((10, 3), dtype=bool), errors)
    assert result["n_errored"] == 2


def test_infinite_values_count_as_non_finite_not_just_nan():
    values = np.zeros((20, N_FEATURES))
    values[:6, 0] = np.inf
    result = gate(values, np.ones((20, 3), dtype=bool))
    assert result["finite_fraction"] == pytest.approx(0.70)
    assert not result["passed"]


# --------------------------------------------------------------------------
# Worker behaviour
# --------------------------------------------------------------------------


def test_extract_one_records_a_failure_rather_than_raising(tmp_path):
    """A pattern that cannot produce features must be recorded, not dropped.

    Silently omitting failures would bias the training set, so the worker
    returns NaNs plus the error text instead of propagating the exception.
    """
    result = extract_one((0, str(tmp_path), build_config()))
    assert result["index"] == 0
    assert result["error"]
    assert result["values"].shape == (N_FEATURES,)
    assert np.all(np.isnan(result["values"]))
    assert not result["k_extrema_interior"].any()


def test_extract_one_returns_features_for_a_real_pattern(tmp_path):
    """End to end through the worker, on a synthetic pattern written to disk in
    the same format the simulator uses."""
    rng = np.random.default_rng(11)
    side = 60.0
    background = rng.uniform(0.0, side, size=(5000, 3))
    clustered = np.clip(
        np.concatenate(
            [
                centre + rng.normal(0.0, 2.5, size=(220, 3))
                for centre in rng.uniform(12.0, side - 12.0, size=(5, 3))
            ]
        ),
        0.0,
        side,
    )
    points = np.concatenate([background, clustered])
    labels = np.concatenate(
        [
            np.zeros(len(background), dtype=np.int8),
            np.full(len(clustered), 2, dtype=np.int8),
        ]
    )
    np.savez(
        tmp_path / "clust_pattern_0.npz",
        coords={"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]},
        domain={axis: np.array([0.0, side]) for axis in "xyz"},
        labels=labels,
        radii=np.full(5, 4.0),
        centers={"x": np.zeros(5), "y": np.zeros(5), "z": np.zeros(5)},
    )

    config = build_config({"k_r_max": 20.0, "k_num_radii": 200,
                           "g_num_radii": 200, "cross_g_num_radii": 200,
                           "f_grid_points_per_axis": 8, "k_max_points": 600})
    result = extract_one((0, str(tmp_path), config))
    assert result["error"] == "", result["error"]
    # The K features may legitimately be NaN under rapt semantics; the G, F and
    # cross-G minimum and 95% radius never are.
    defined = [i for i, n in enumerate(psf.PAPER_FEATURE_NAMES)
               if n not in ("Tm", "Rm", "Rdm", "Rddm", "Tdm", "GXGH_FWHM")]
    assert np.all(np.isfinite(result["values"][defined]))
    assert result["k_extrema_interior"].shape == (3,)
    assert result["seconds"] > 0.0


# --------------------------------------------------------------------------
# Pattern discovery
# --------------------------------------------------------------------------


def test_discover_patterns_stops_at_the_first_gap(tmp_path):
    for index in (0, 1, 2, 4):        # 3 is missing
        (tmp_path / f"clust_pattern_{index}.npz").touch()
    assert discover_patterns(tmp_path) == [0, 1, 2]


def test_discover_patterns_honours_limit(tmp_path):
    for index in range(10):
        (tmp_path / f"clust_pattern_{index}.npz").touch()
    assert discover_patterns(tmp_path, limit=4) == [0, 1, 2, 3]


def test_discover_patterns_returns_empty_for_an_empty_directory(tmp_path):
    assert discover_patterns(tmp_path) == []


# --------------------------------------------------------------------------
# Feature screen
# --------------------------------------------------------------------------


def test_r_squared_is_zero_for_the_mean_predictor():
    from experiments.inference.screen_features import r_squared

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    mean = truth.mean()
    assert r_squared(truth, np.full(4, mean), mean) == pytest.approx(0.0)


def test_r_squared_is_one_for_a_perfect_predictor():
    from experiments.inference.screen_features import r_squared

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(truth, truth, truth.mean()) == pytest.approx(1.0)


def test_r_squared_goes_negative_for_a_predictor_worse_than_the_mean():
    from experiments.inference.screen_features import r_squared

    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(truth, np.full(4, 100.0), truth.mean()) < 0.0


def test_screen_recovers_a_planted_linear_signal_and_rejects_noise():
    """One target is a linear function of the features, one is pure noise.

    The screen must separate them, which is the whole job it does before
    Stage 2 commits to a flow.
    """
    from geom_mesh_net.simulation.parameters import PARAMETER_NAMES
    from experiments.inference.screen_features import screen

    rng = np.random.default_rng(0)
    n = 400
    features = rng.normal(size=(n, N_FEATURES))
    theta = np.empty((n, 4))
    theta[:, 0] = features @ rng.normal(size=N_FEATURES)      # learnable
    theta[:, 1] = features[:, 3] * 2.0                        # learnable
    theta[:, 2] = features[:, 7] ** 2                         # learnable, quadratic
    theta[:, 3] = rng.normal(size=n)                          # independent noise

    summary, n_train, n_test, n_splits = screen(features, theta, n_splits=5)
    assert n_train + n_test == n
    assert n_splits == 5

    learnable = [PARAMETER_NAMES[i] for i in (0, 1)]
    for name in learnable:
        assert summary[name]["best_r2_mean"] > 0.9, (name, summary[name])
    # The noise target must not be predictable.
    assert summary[PARAMETER_NAMES[3]]["best_r2_mean"] < 0.2


def test_screen_reports_spread_across_splits():
    """A single-split R^2 moves by more than the effect being tested for, so
    the spread has to be reported alongside the mean."""
    from experiments.inference.screen_features import screen

    rng = np.random.default_rng(1)
    features = rng.normal(size=(300, N_FEATURES))
    theta = rng.normal(size=(300, 4))
    summary, _, _, _ = screen(features, theta, n_splits=6)
    for scores in summary.values():
        assert scores["best_r2_sd"] >= 0.0
        assert scores["best_r2_min"] <= scores["best_r2_mean"]
        assert scores["best_r2_max"] >= scores["best_r2_mean"]
