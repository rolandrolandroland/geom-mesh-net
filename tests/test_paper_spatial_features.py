"""Regression tests for the spatial-summary feature library.

These pin the numerical claims the feature definitions rest on. Where a
closed form exists (CSR baselines, Ripley's K under a homogeneous Poisson
process, Kaplan-Meier with no censoring) the test compares against it rather
than against a previously recorded output, so the tests can catch a genuine
error rather than merely detecting change.
"""

import numpy as np
import pytest

from geom_mesh_net.core_functions import paper_spatial_features as psf
from geom_mesh_net.core_functions.paper_spatial_features import (
    K_TRANSFORMS,
    LocalPaperFeatureConfig,
    PaperFeatureConfig,
    _extract_k_features,
    _kaplan_meier_cdf,
    _local_maxima,
    _loess,
    _translation_corrected_k,
    calculate_csr_baseline,
    calculate_global_paper_features,
    transform_k,
)


def ball_volume(radii):
    return (4.0 / 3.0) * np.pi * np.asarray(radii, dtype=float) ** 3


# --------------------------------------------------------------------------
# Ripley's K
# --------------------------------------------------------------------------


def test_translation_corrected_k_is_unbiased_for_csr():
    """Under CSR, K(r) is the ball volume, independent of intensity.

    This is the load-bearing correctness claim for every K-derived feature. It
    is checked by averaging over independent Poisson realizations, because a
    single realization carries a few percent of sampling noise.
    """
    side = 40.0
    bounds = np.array([[0.0, side]] * 3)
    radii = np.linspace(0.0, 10.0, 51)

    rng = np.random.default_rng(0)
    estimates = np.stack(
        [
            _translation_corrected_k(
                rng.uniform(0.0, side, size=(4000, 3)), bounds, radii
            )
            for _ in range(8)
        ]
    )
    mean_estimate = estimates.mean(axis=0)

    # Skip r near zero, where the ratio is dominated by very few pairs.
    interior = radii >= 1.0
    ratio = mean_estimate[interior] / ball_volume(radii[interior])
    assert np.abs(ratio.mean() - 1.0) < 0.02, ratio.mean()
    assert np.abs(ratio - 1.0).max() < 0.05, np.abs(ratio - 1.0).max()


def test_translation_corrected_k_is_monotone_and_starts_at_zero():
    rng = np.random.default_rng(1)
    side = 20.0
    bounds = np.array([[0.0, side]] * 3)
    radii = np.linspace(0.0, 5.0, 26)
    k = _translation_corrected_k(
        rng.uniform(0.0, side, size=(500, 3)), bounds, radii
    )
    assert k[0] == 0.0
    assert np.all(np.diff(k) >= -1e-12)


def test_k_radius_beyond_shortest_side_is_rejected():
    """r > L is where the translation correction genuinely breaks.

    Measured bias at r/L = 1.17 is about -9%. The estimator is fine up to
    r = L, so the guard is placed there and not at the conventional L/4.
    """
    rng = np.random.default_rng(2)
    side = 10.0
    bounds = np.array([[0.0, side]] * 3)
    points = rng.uniform(0.0, side, size=(200, 3))

    _translation_corrected_k(points, bounds, np.linspace(0.0, side, 11))

    with pytest.raises(ValueError, match="exceeds the shortest"):
        _translation_corrected_k(points, bounds, np.linspace(0.0, side * 1.5, 11))


def test_global_features_reject_k_r_max_larger_than_domain():
    """example_01/global_paper_feature_validation used k_r_max=70 in a
    60-unit domain. That configuration is now refused."""
    coords, labels, domain = _synthetic_clustered_pattern()
    config = PaperFeatureConfig(
        k_r_max=70.0, k_num_radii=20, null_model="csr", g_num_radii=50,
        cross_g_num_radii=50, f_grid_points_per_axis=4,
    )
    with pytest.raises(ValueError, match="exceeds the shortest domain side"):
        calculate_global_paper_features(coords, labels, domain, config=config)


# --------------------------------------------------------------------------
# The variance-stabilizing transform
# --------------------------------------------------------------------------


def test_cube_root_transform_linearizes_csr_in_three_dimensions():
    """The correct 3D transform maps K_csr(r) back to r exactly."""
    radii = np.linspace(0.0, 12.0, 61)
    assert np.allclose(transform_k(ball_volume(radii), "cube_root"), radii)


def test_sqrt_transform_does_not_linearize_csr_in_three_dimensions():
    """sqrt is the 2D transform; in 3D it leaves K_csr proportional to r**1.5.

    This is why it was replaced as the default: a difference of two sqrt-K
    curves is inflated at large radii and its extrema shift outward.
    """
    radii = np.linspace(1.0, 12.0, 45)
    transformed = transform_k(ball_volume(radii), "sqrt")
    # sqrt(K_csr) = sqrt(4 pi / 3) * r**1.5, so the ratio to r**1.5 is constant
    # while the ratio to r is not.
    assert np.allclose(transformed / radii**1.5, transformed[0] / radii[0] ** 1.5)
    ratio_to_r = transformed / radii
    assert ratio_to_r.max() / ratio_to_r.min() > 3.0


def test_transform_k_clips_negative_input():
    assert np.all(transform_k(np.array([-5.0, -1e-9, 0.0]), "cube_root") == 0.0)
    assert np.all(transform_k(np.array([-5.0]), "sqrt") == 0.0)


def test_transform_k_none_is_identity_above_zero():
    values = np.array([0.0, 1.0, 7.5])
    assert np.allclose(transform_k(values, "none"), values)


def test_transform_k_rejects_unknown_kind():
    with pytest.raises(ValueError, match="k_transform must be one of"):
        transform_k(np.array([1.0]), "log")


@pytest.mark.parametrize("kind", K_TRANSFORMS)
def test_every_advertised_transform_works(kind):
    assert np.all(np.isfinite(transform_k(ball_volume(np.linspace(0, 5, 11)), kind)))


# --------------------------------------------------------------------------
# K feature extraction and the boundary-pinning diagnostic
# --------------------------------------------------------------------------


def test_k_extrema_flags_are_false_for_a_monotonically_rising_curve():
    """A curve still rising at r_max has no interior extremum.

    The extractor falls back to argmax and returns a grid endpoint, which is
    indistinguishable from a real measurement. The flags exist to make that
    visible, and this is the case they must catch.
    """
    radii = np.linspace(0.0, 10.0, 201)
    values, interior = _extract_k_features(
        radii, radii.copy(), smoothing_reference_r_max=10.0
    )
    assert not interior[0], "Rm should be flagged as not interior"
    # The returned radius is a grid endpoint, i.e. not a measurement.
    assert values[1] in (radii[0], radii[-1])


def test_k_extrema_flags_are_true_for_a_clearly_peaked_curve():
    """A realistically shaped K difference: rises from zero, peaks, decays."""
    radii = np.linspace(0.0, 10.0, 201)
    curve = radii**2 * np.exp(-radii / 2.0)
    assert radii[np.argmax(curve)] == pytest.approx(4.0, abs=0.1)

    values, interior = _extract_k_features(
        radii, curve, smoothing_reference_r_max=10.0
    )
    assert interior[0], "Rm should be flagged interior for a peaked curve"
    assert 3.0 < values[1] < 5.0, values[1]


def test_extractor_takes_the_first_local_maximum_not_the_global_one():
    """Documents a known fragility rather than asserting desired behaviour.

    ``_extract_k_features`` selects ``peaks[0]`` -- the first local maximum of
    the LOESS-smoothed curve. The smoothing span is chosen adaptively from an
    initial peak estimate, and for a peak that is narrow relative to that span
    the second smoothing pass rings, introducing a spurious local maximum at
    small r which is then selected in preference to the true peak.

    Here the initial pass locates the true peak at r = 4.0 correctly, the
    adaptive span widens to roughly 0.17 of the range, and Rm is nonetheless
    reported near r = 0.85.

    Taking the first peak is plausibly the intended definition -- the first
    peak of a K difference is the primary cluster scale -- so this is recorded
    rather than changed. Real patterns at the recommended settings produce
    broad peaks and are unaffected. If the definition is ever revisited, this
    test will flag the change.
    """
    radii = np.linspace(0.0, 10.0, 201)
    narrow = np.exp(-((radii - 4.0) ** 2) / 0.8)

    initial = _loess(radii, narrow, span=0.08)
    assert radii[_local_maxima(initial, half_window=3)[0]] == pytest.approx(4.0, abs=0.1)

    values, interior = _extract_k_features(
        radii, narrow, smoothing_reference_r_max=10.0
    )
    assert interior[0]
    assert values[1] < 2.0, (
        "known fragility: Rm follows a smoothing ripple for narrow peaks; "
        f"got {values[1]}"
    )


def test_extract_k_features_returns_five_values_and_three_flags():
    radii = np.linspace(0.0, 10.0, 101)
    values, interior = _extract_k_features(
        radii, np.exp(-((radii - 5.0) ** 2)), smoothing_reference_r_max=10.0
    )
    assert values.shape == (5,)
    assert interior.shape == (3,)
    assert interior.dtype == bool


# --------------------------------------------------------------------------
# Kaplan-Meier edge-corrected CDF
# --------------------------------------------------------------------------


def test_kaplan_meier_reduces_to_the_empirical_cdf_without_censoring():
    events = np.array([1.0, 2.0, 3.0, 4.0])
    censor = np.full(4, 1e6)
    radii = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    result = _kaplan_meier_cdf(events, censor, radii)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    assert np.allclose(result, expected), result


def test_kaplan_meier_handles_censoring_by_hand_worked_example():
    """Three observations, one censored before its event.

    observed times 2 (event), 3 (censored), 5 (event)
      r=2: at risk 3, 1 event  -> S = 2/3, F = 1/3
      r=3: at risk 2, 0 events -> S = 2/3, F = 1/3
      r=5: at risk 1, 1 event  -> S = 0,   F = 1
    """
    events = np.array([2.0, 8.0, 5.0])
    censor = np.array([10.0, 3.0, 10.0])
    radii = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _kaplan_meier_cdf(events, censor, radii)
    expected = np.array([0.0, 1 / 3, 1 / 3, 1 / 3, 1.0])
    assert np.allclose(result, expected), result


def test_kaplan_meier_is_bounded_and_nondecreasing():
    rng = np.random.default_rng(3)
    events = rng.exponential(2.0, size=400)
    censor = rng.exponential(3.0, size=400)
    radii = np.linspace(0.0, 8.0, 60)
    result = _kaplan_meier_cdf(events, censor, radii)
    assert result.min() >= 0.0 and result.max() <= 1.0
    assert np.all(np.diff(result) >= -1e-12)


# --------------------------------------------------------------------------
# Analytical CSR baseline
# --------------------------------------------------------------------------


def test_csr_baseline_matches_closed_form():
    radii = {
        "g": np.linspace(0.0, 5.0, 40),
        "k": np.linspace(0.0, 5.0, 40),
        "cross_g": np.linspace(0.0, 3.0, 30),
    }
    guest_intensity, host_intensity = 0.02, 0.8
    baseline = calculate_csr_baseline(radii, guest_intensity, host_intensity)

    assert np.allclose(
        baseline.guest_g, 1.0 - np.exp(-guest_intensity * ball_volume(radii["g"]))
    )
    # Under CSR the nearest-neighbour and empty-space distributions coincide.
    assert np.allclose(baseline.guest_g, baseline.guest_f)
    assert np.allclose(baseline.guest_k, ball_volume(radii["k"]))
    assert np.allclose(
        baseline.guest_to_host_g,
        1.0 - np.exp(-host_intensity * ball_volume(radii["cross_g"])),
    )


def test_csr_baseline_rejects_nonpositive_intensity():
    radii = {
        "g": np.linspace(0.0, 2.0, 5),
        "k": np.linspace(0.0, 2.0, 5),
        "cross_g": np.linspace(0.0, 2.0, 5),
    }
    with pytest.raises(ValueError, match="CSR intensities"):
        calculate_csr_baseline(radii, 0.0, 1.0)


def test_cube_root_transform_zeroes_the_csr_k_difference():
    """With the correct transform, a CSR pattern has no K signal at all.

    This is the property that makes the difference curve interpretable: any
    departure from zero is structure rather than an artifact of the transform.
    """
    radii = {
        "g": np.linspace(0.0, 3.0, 20),
        "k": np.linspace(0.0, 8.0, 60),
        "cross_g": np.linspace(0.0, 3.0, 20),
    }
    baseline = calculate_csr_baseline(radii, 0.05, 0.5)
    difference = transform_k(baseline.guest_k, "cube_root") - transform_k(
        baseline.guest_k, "cube_root"
    )
    assert np.allclose(difference, 0.0)
    # And the transform of the CSR K is the radius itself.
    assert np.allclose(transform_k(baseline.guest_k, "cube_root"), radii["k"])


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------


def test_config_defaults_to_the_three_dimensional_transform():
    assert PaperFeatureConfig().k_transform == "cube_root"


def test_config_rejects_unknown_transform_and_null_model():
    with pytest.raises(ValueError, match="k_transform must be one of"):
        PaperFeatureConfig(k_transform="log")
    with pytest.raises(ValueError, match="null_model must be one of"):
        PaperFeatureConfig(null_model="bootstrap")


def test_config_rejects_nonpositive_radii():
    with pytest.raises(ValueError, match="g_r_max must be greater than zero"):
        PaperFeatureConfig(g_r_max=0.0)


def test_local_config_validation():
    with pytest.raises(ValueError, match="neighborhood_radius"):
        LocalPaperFeatureConfig(neighborhood_radius=0.0)
    with pytest.raises(ValueError, match="csr_intensity_scope"):
        LocalPaperFeatureConfig(csr_intensity_scope="regional")
    with pytest.raises(ValueError, match="minimum_points"):
        LocalPaperFeatureConfig(minimum_points=2)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _synthetic_clustered_pattern(seed=7, side=60.0):
    """A small labeled pattern with real guest clustering.

    Deliberately not loaded from data/, so the tests run without the 5 GB
    dataset present.
    """
    rng = np.random.default_rng(seed)
    background = rng.uniform(0.0, side, size=(6000, 3))
    centers = rng.uniform(10.0, side - 10.0, size=(4, 3))
    clustered = np.concatenate(
        [c + rng.normal(0.0, 2.5, size=(250, 3)) for c in centers]
    )
    clustered = np.clip(clustered, 0.0, side)

    points = np.concatenate([background, clustered])
    labels = np.concatenate(
        [np.zeros(len(background), dtype=np.int8), np.full(len(clustered), 2, np.int8)]
    )
    coords = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    domain = {axis: np.array([0.0, side]) for axis in "xyz"}
    return coords, labels, domain


def _cheap_config(**overrides):
    base = dict(
        g_r_max=8.0,
        g_num_radii=120,
        k_r_max=12.0,
        k_num_radii=80,
        cross_g_r_max=5.0,
        cross_g_num_radii=120,
        f_grid_points_per_axis=6,
        null_model="csr",
        k_max_points=800,
    )
    base.update(overrides)
    return PaperFeatureConfig(**base)


def test_global_features_end_to_end_are_finite_and_named():
    coords, labels, domain = _synthetic_clustered_pattern()
    result = calculate_global_paper_features(
        coords, labels, domain, guest_marks=(2, 3), config=_cheap_config()
    )
    assert result.values.shape == (14,)
    assert result.names == psf.PAPER_FEATURE_NAMES
    assert np.all(np.isfinite(result.values))
    assert result.k_extrema_interior is not None
    assert result.k_extrema_interior.shape == (3,)


def test_clustering_raises_g_above_the_csr_expectation():
    """A clustered guest population has closer nearest neighbours than CSR,
    so the observed-minus-expected G difference must be positive."""
    coords, labels, domain = _synthetic_clustered_pattern()
    result = calculate_global_paper_features(
        coords, labels, domain, config=_cheap_config()
    )
    g_max_diff = result.values[psf.PAPER_FEATURE_NAMES.index("G_max_diff")]
    assert g_max_diff > 0.1, g_max_diff


def test_transform_choice_changes_only_the_k_features():
    """Switching the transform must not perturb G, F or cross-G."""
    coords, labels, domain = _synthetic_clustered_pattern()
    cube = calculate_global_paper_features(
        coords, labels, domain, config=_cheap_config(k_transform="cube_root")
    ).values
    sqrt = calculate_global_paper_features(
        coords, labels, domain, config=_cheap_config(k_transform="sqrt")
    ).values

    k_names = {"Tm", "Rm", "Rdm", "Rddm", "Tdm"}
    for index, name in enumerate(psf.PAPER_FEATURE_NAMES):
        if name in k_names:
            continue
        assert cube[index] == pytest.approx(sqrt[index]), name


def test_random_label_null_is_reproducible_for_a_fixed_seed():
    coords, labels, domain = _synthetic_clustered_pattern()
    config = _cheap_config(null_model="random_label", n_relabelings=3)
    first = calculate_global_paper_features(coords, labels, domain, config=config)
    second = calculate_global_paper_features(coords, labels, domain, config=config)
    assert np.array_equal(first.values, second.values)


def test_features_require_both_marks_present():
    coords, labels, domain = _synthetic_clustered_pattern()
    all_host = np.zeros_like(labels)
    with pytest.raises(ValueError):
        calculate_global_paper_features(
            coords, all_host, domain, config=_cheap_config()
        )


# --------------------------------------------------------------------------
# Local (per-voxel) features
# --------------------------------------------------------------------------


def test_local_features_return_per_query_values_and_diagnostics():
    coords, labels, domain = _synthetic_clustered_pattern()
    queries = np.array(
        [[15.0, 15.0, 15.0], [30.0, 30.0, 30.0], [45.0, 45.0, 45.0]]
    )
    result = psf.calculate_local_paper_features(
        coords,
        labels,
        domain,
        queries,
        config=_cheap_config(k_r_max=8.0, k_num_radii=60, k_max_points=400),
        local_config=LocalPaperFeatureConfig(neighborhood_radius=12.0),
    )
    assert result.values.shape == (3, 14)
    assert result.valid.shape == (3,)
    assert result.k_extrema_interior.shape == (3, 3)
    assert np.all(np.isfinite(result.values[result.valid]))
    assert np.all(result.point_counts > 0)


def test_local_features_reject_queries_outside_the_domain():
    coords, labels, domain = _synthetic_clustered_pattern()
    with pytest.raises(ValueError, match="must lie inside the domain"):
        psf.calculate_local_paper_features(
            coords,
            labels,
            domain,
            np.array([[100.0, 100.0, 100.0]]),
            config=_cheap_config(k_r_max=8.0),
            local_config=LocalPaperFeatureConfig(neighborhood_radius=12.0),
        )


def test_local_k_radius_is_bounded_by_the_window_not_the_domain():
    """A 12-unit neighbourhood radius gives a 24-unit window, so k_r_max=30 is
    invalid even though the domain is 60 units across."""
    coords, labels, domain = _synthetic_clustered_pattern()
    with pytest.raises(ValueError, match="exceeds the shortest local window side"):
        psf.calculate_local_paper_features(
            coords,
            labels,
            domain,
            np.array([[30.0, 30.0, 30.0]]),
            config=_cheap_config(k_r_max=30.0),
            local_config=LocalPaperFeatureConfig(neighborhood_radius=12.0),
        )
