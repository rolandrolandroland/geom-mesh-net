"""Named configurations of the 14 global spatial-summary features.

Both experiment tracks compute features with these settings: the inference track
to fit its posterior, and the reconstruction track for its predictive check.
Keeping them in one place is what makes the two comparable.
"""

from geom_mesh_net.statistics import paper_spatial_features as psf

# Radius grids are set from the physical scale of this dataset: a 60-unit
# domain with cluster radii up to 15. See ROADMAP section 8.3 for k_r_max.
STAGE1_CONFIG = dict(
    g_r_max=10.0,
    g_num_radii=1001,
    k_r_max=40.0,
    k_num_radii=801,
    cross_g_r_max=8.0,
    cross_g_num_radii=801,
    f_grid_points_per_axis=24,
    null_model="csr",
    k_transform="sqrt",
    k_max_points=3000,
    k_smoothing_reference_r_max=10.0,
    # Each pattern is a separate process, so let each use one thread rather
    # than have every worker try to claim every core.
    workers=1,
    # The committed Stage 1 results predate the faithful rapt extraction, so this
    # preset pins the legacy port to keep them reproducible. Do not use it for
    # new work: the legacy port returns grid endpoints where rapt returns NA.
    feature_method="legacy_port",
)

# The configuration of rapt's walkthrough.Rmd, which produced Bennett, Proudian
# and Zimmerman (2023): G and F to 4 over 2000 radii, K to 30 over 100,
# guest-to-host G to 3 over 2000, and rapt's feature extraction. The null model
# stays csr because rapt computes its expectation once, by relabeling a single
# point pattern shared by every training pattern; for independently simulated
# patterns the analytic CSR expectation is the equivalent. spatstat's F3est uses
# a chamfer distance on ~4.2 million voxels, which the Euclidean grid here does
# not reproduce -- see ROADMAP section 8.9.
PAPER_CONFIG = dict(
    g_r_max=4.0,
    g_num_radii=2000,
    k_r_max=30.0,
    k_num_radii=100,
    cross_g_r_max=3.0,
    cross_g_num_radii=2000,
    f_grid_points_per_axis=60,
    null_model="csr",
    k_transform="sqrt",
    k_max_points=3000,
    workers=1,
    feature_method="rapt",
)

PRESETS = {"stage1": STAGE1_CONFIG, "paper": PAPER_CONFIG}

# Under rapt semantics these are NaN wherever rapt returns NA. The rest are
# always defined.
MAY_BE_MISSING = ("Tm", "Rm", "Rdm", "Rddm", "Tdm", "GXGH_FWHM")


def build_config(overrides=None, preset="stage1"):
    """A ``PaperFeatureConfig`` from a named preset, with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {tuple(PRESETS)}")
    values = dict(PRESETS[preset])
    values.update(overrides or {})
    return psf.PaperFeatureConfig(**values)
