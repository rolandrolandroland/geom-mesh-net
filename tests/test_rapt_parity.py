"""Parity between the Python feature extraction and the R package rapt.

rapt is the reference implementation for Bennett, Proudian and Zimmerman (2023),
Ultramicroscopy 247, 113687. The Python module began as a port of it, and a
line-by-line comparison found that the port had diverged in ways that mattered:
most seriously, where rapt returns NA because a K difference curve has no local
maximum, the port returned a grid endpoint as though it were a measurement.

These tests pin the faithful port (``feature_method="rapt"``) to rapt's own
output. The reference values in ``tests/fixtures/rapt_parity.npz`` were produced
by calling the installed rapt package on curves computed from 60 simulated
patterns at the paper's grids -- K to 30 over 100 radii, G and F to 4 over 2000,
guest-to-host G to 3 over 2000 -- so the tests run without R.

``test_fixture_matches_live_rapt`` regenerates the K references from R and
skips when R or rapt is unavailable, so a change to rapt cannot leave the
fixture silently stale.

Agreement is to 1e-11. The residual is floating-point rounding: the logic,
including every NA path, is reproduced exactly.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geom_mesh_net.core_functions import paper_spatial_features as psf


FIXTURE = Path(__file__).parent / "fixtures" / "rapt_parity.npz"
TOLERANCE = 1e-11


@pytest.fixture(scope="module")
def ref():
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def assert_same_including_nan(actual, expected, label):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), (
        f"{label}: NA pattern differs from rapt"
    )
    finite = ~np.isnan(expected)
    if finite.any():
        worst = np.abs(actual[finite] - expected[finite]).max()
        assert worst < TOLERANCE * max(1.0, np.abs(expected[finite]).max()), (
            f"{label}: max |diff| {worst:.3e}"
        )


# --------------------------------------------------------------------------
# loess
# --------------------------------------------------------------------------


def test_loess_matches_r_default_interpolated_surface(ref):
    """rapt calls loess with R's defaults, whose surface is interpolated from a
    kd-tree rather than evaluated at every point. Reproduced exactly."""
    for span, expected in zip(ref["loess_spans"], ref["loess_interpolate"]):
        actual = psf._loess_rapt(ref["loess_x"], ref["loess_y"], span)
        assert_same_including_nan(actual, expected, f"interpolated span {span}")


def test_loess_matches_r_direct_surface(ref):
    for span, expected in zip(ref["loess_spans"], ref["loess_direct"]):
        actual = psf._loess_rapt(
            ref["loess_x"], ref["loess_y"], span, surface="direct"
        )
        assert_same_including_nan(actual, expected, f"direct span {span}")


def test_loess_neighbourhood_uses_floor_with_epsilon():
    """0.57 * 100 is 56.99999999999999 in floating point. R floors it to 57 by
    adding 1e-5 first; a plain floor gives 56 and a different fit."""
    assert int(np.floor(0.57 * 100)) == 56
    assert int(np.floor(0.57 * 100 + 1e-5)) == 57


def test_legacy_loess_diverges_from_r(ref):
    """The earlier port used ceil and clipped spans above one. Recorded so the
    divergence stays visible."""
    spans = list(ref["loess_spans"])
    large = spans.index(1.4)
    legacy = psf._loess(ref["loess_x"], ref["loess_y"], 1.4)
    scale = np.abs(ref["loess_y"]).max()
    assert np.abs(legacy - ref["loess_direct"][large]).max() / scale > 1e-3


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------


def test_k_features_match_rapt(ref):
    for index, curve in enumerate(ref["k_curves"]):
        values, _ = psf._extract_k_features_rapt(ref["radii_k"], curve)
        assert_same_including_nan(
            values, ref["k_features_r"][index], f"K features, pattern {index}"
        )


def test_rapt_returns_nan_where_a_peak_is_missing(ref):
    """The central correction. The fixture contains patterns where rapt reports
    NA; the faithful port must report NaN in exactly the same places."""
    missing = np.isnan(ref["k_features_r"])
    assert missing[:, 1].sum() > 0, "fixture should exercise the Rm NA path"
    assert missing[:, 3].sum() > 0, "fixture should exercise the Rddm NA path"


def test_legacy_k_extraction_fabricates_values_where_rapt_has_none(ref):
    """The defect, pinned. Wherever rapt returns NA for Rm, the legacy port
    returns a finite number instead."""
    missing = np.isnan(ref["k_features_r"][:, 1])
    for index in np.flatnonzero(missing):
        values, _ = psf._extract_k_features(
            ref["radii_k"], ref["k_curves"][index], smoothing_reference_r_max=10.0
        )
        assert np.isfinite(values[1]), index


def test_interior_flags_agree_with_nan(ref):
    for curve in ref["k_curves"]:
        values, interior = psf._extract_k_features_rapt(ref["radii_k"], curve)
        assert np.array_equal(interior, np.isfinite(values[[1, 2, 3]]))


def test_g_features_match_rapt(ref):
    for index in range(len(ref["g_new"])):
        values = psf._extract_g_features_rapt(
            ref["radii_g"], ref["g_new"][index], ref["g_old"][index]
        )
        assert_same_including_nan(
            values, ref["g_features_r"][index], f"G features, pattern {index}"
        )


def test_f_features_match_rapt(ref):
    for index in range(len(ref["f_new"])):
        values = psf._extract_f_features(ref["f_new"][index], ref["f_old"][index])
        assert_same_including_nan(
            values, ref["f_features_r"][index], f"F features, pattern {index}"
        )


def test_cross_g_features_match_rapt(ref):
    for index in range(len(ref["x_new"])):
        values = psf._extract_cross_g_features_rapt(
            ref["radii_x"], ref["x_new"][index], ref["x_old"][index]
        )
        assert_same_including_nan(
            values, ref["x_features_r"][index], f"cross-G features, pattern {index}"
        )


def test_cross_g_fwhm_reproduces_the_off_by_one_in_rapt():
    """rapt computes the right half-maximum index as `ind + which(...)`, one step
    past the true position. Reproduced deliberately, so the width here is one
    grid step wider than the geometrically correct one."""
    radii = np.linspace(0.0, 10.0, 101)
    curve = np.exp(-((radii - 5.0) ** 2) / 2.0)
    faithful = psf._extract_cross_g_features_rapt(radii, curve + 0.5, 0.5 * np.ones_like(radii))
    geometric = psf._extract_cross_g_features(radii, curve + 0.5, 0.5 * np.ones_like(radii))
    assert faithful[2] - geometric[2] == pytest.approx(radii[1] - radii[0])


def test_g_zero_radius_searches_the_whole_curve():
    """rapt looks up G_zero_diff_r with which(diff == zero_diff) over the entire
    difference curve, so an identical value earlier in the curve wins over the
    one inside the window between the extrema."""
    radii = np.arange(10.0)
    difference = np.array([0.0, 0.0, 0.5, 1.0, 0.4, 0.0, -0.3, -0.6, -0.2, -0.1])
    values = psf._extract_g_features_rapt(radii, difference, np.zeros(10))
    assert values[3] == 0.0          # rapt: first zero anywhere, at r = 0
    legacy = psf._extract_g_features(radii, difference, np.zeros(10))
    assert legacy[3] == 5.0          # port: the zero between max (3) and min (7)


def test_default_feature_method_is_rapt():
    assert psf.PaperFeatureConfig().feature_method == "rapt"


def test_unknown_feature_method_is_rejected():
    with pytest.raises(ValueError, match="feature_method must be one of"):
        psf.PaperFeatureConfig(feature_method="port")


# --------------------------------------------------------------------------
# Live check against the installed R package
# --------------------------------------------------------------------------


def _rapt_available():
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        ["Rscript", "-e", "suppressPackageStartupMessages({library(rapt); library(dplyr)})"],
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _rapt_available(), reason="R with rapt and dplyr not installed")
def test_fixture_matches_live_rapt(ref, tmp_path):
    """Regenerate the K references from the installed rapt so the committed
    fixture cannot drift from the package it claims to represent."""
    np.savetxt(tmp_path / "r.csv", ref["radii_k"], delimiter=",")
    np.savetxt(tmp_path / "k.csv", ref["k_curves"][:15], delimiter=",")
    script = tmp_path / "run.R"
    script.write_text(
        "suppressPackageStartupMessages({ library(rapt); library(dplyr) })\n"
        f"r <- read.csv('{tmp_path / 'r.csv'}', header = FALSE)[, 1]\n"
        f"K <- as.matrix(read.csv('{tmp_path / 'k.csv'}', header = FALSE))\n"
        "out <- t(sapply(seq_len(nrow(K)), function(i)\n"
        "  unlist(suppressWarnings(k3features(r, K[i, ], toplot = FALSE)))))\n"
        f"write.csv(out, '{tmp_path / 'out.csv'}', row.names = FALSE)\n"
    )
    subprocess.run(["Rscript", str(script)], check=True, capture_output=True)
    rows = (tmp_path / "out.csv").read_text().strip().splitlines()[1:]
    live = np.array([[np.nan if v == "NA" else float(v) for v in row.split(",")] for row in rows])
    assert_same_including_nan(live, ref["k_features_r"][:15], "live rapt vs fixture")
