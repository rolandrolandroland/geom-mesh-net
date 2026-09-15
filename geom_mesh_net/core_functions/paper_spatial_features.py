"""Global and local spatial-summary features from Bennett et al. (2023)."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree


PAPER_FEATURE_NAMES = (
    "G_max_diff",
    "G_max_diff_r",
    "G_min_diff",
    "G_zero_diff_r",
    "F_min_diff",
    "F_min_diff_F",
    "Tm",
    "Rm",
    "Rdm",
    "Rddm",
    "Tdm",
    "GXGH_min_diff",
    "GXGH_95diff_r",
    "GXGH_FWHM",
)

NULL_MODELS = ("csr", "random_label")

K_TRANSFORMS = ("cube_root", "sqrt", "none")


def transform_k(values, kind):
    """Apply a transform to Ripley's K before differencing against a null.

    ``sqrt`` is the default and matches the Bennett et al. feature definition
    that ``Tm``, ``Rm``, ``Rdm``, ``Rddm`` and ``Tdm`` are ported from. It is
    what produced ``example_01/methodology_01_results`` and
    ``example_01/global_paper_feature_validation``.

    ``cube_root`` is the variance-stabilizing transform for a *three*
    dimensional CSR process: K_csr(r) = (4/3) pi r^3, so

        L(r) = (3 K(r) / (4 pi))^(1/3) = r

    which maps the CSR reference onto the identity line and makes a
    difference curve zero under CSR at every radius. Available for comparison,
    but it changes the feature semantics relative to the paper, so switching
    is a deliberate choice rather than a correction.

    Note that the choice interacts with ``k_r_max``: because sqrt(K_csr) grows
    as r^1.5 rather than r, difference curves under ``sqrt`` peak at larger
    radii and need a larger ``k_r_max`` before an interior extremum exists at
    all. See ``inference/ROADMAP.md`` section 8.3.

    ``none`` returns K unchanged, for callers that want to do their own
    scaling.
    """
    if kind not in K_TRANSFORMS:
        raise ValueError(f"k_transform must be one of {K_TRANSFORMS}")
    clipped = np.maximum(values, 0.0)
    if kind == "cube_root":
        return np.cbrt(3.0 * clipped / (4.0 * np.pi))
    if kind == "sqrt":
        return np.sqrt(clipped)
    return clipped


@dataclass(frozen=True)
class PaperFeatureConfig:
    g_r_max: float = 5.0
    g_num_radii: int = 1000
    k_r_max: float = 10.0
    k_num_radii: int = 200
    cross_g_r_max: float = 3.0
    cross_g_num_radii: int = 1000
    f_grid_points_per_axis: int = 24
    n_relabelings: int = 99
    random_seed: int = 42
    workers: int = -1
    k_max_points: int | None = None
    k_smoothing_reference_r_max: float = 10.0
    null_model: str = "random_label"
    k_transform: str = "sqrt"
    # "rapt" reproduces the R package's feature extraction exactly (see the
    # section on faithful ports near the end of this module). "legacy_port" is
    # the earlier Python port, kept to reproduce results generated with it.
    feature_method: str = "rapt"

    def __post_init__(self):
        positive_values = {
            "g_r_max": self.g_r_max,
            "g_num_radii": self.g_num_radii,
            "k_r_max": self.k_r_max,
            "k_num_radii": self.k_num_radii,
            "cross_g_r_max": self.cross_g_r_max,
            "cross_g_num_radii": self.cross_g_num_radii,
            "f_grid_points_per_axis": self.f_grid_points_per_axis,
            "n_relabelings": self.n_relabelings,
            "k_smoothing_reference_r_max": (
                self.k_smoothing_reference_r_max
            ),
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.k_max_points is not None and self.k_max_points < 2:
            raise ValueError("k_max_points must be at least two")
        if self.null_model not in NULL_MODELS:
            raise ValueError(
                f"null_model must be one of {NULL_MODELS}"
            )
        if self.k_transform not in K_TRANSFORMS:
            raise ValueError(
                f"k_transform must be one of {K_TRANSFORMS}"
            )
        if self.feature_method not in ("rapt", "legacy_port"):
            raise ValueError(
                "feature_method must be one of ('rapt', 'legacy_port')"
            )


@dataclass(frozen=True)
class LocalPaperFeatureConfig:
    neighborhood_radius: float = 12.0
    csr_intensity_scope: str = "global"
    minimum_points: int = 3
    minimum_guest_points: int = 0
    minimum_host_points: int = 1

    def __post_init__(self):
        if self.neighborhood_radius <= 0:
            raise ValueError("neighborhood_radius must be greater than zero")
        if self.csr_intensity_scope not in {"global", "local"}:
            raise ValueError(
                "csr_intensity_scope must be either 'global' or 'local'"
            )
        if self.minimum_points < 3:
            raise ValueError("minimum_points must be at least three")
        if self.minimum_guest_points < 0:
            raise ValueError("minimum_guest_points must be nonnegative")
        if self.minimum_host_points < 1:
            raise ValueError("minimum_host_points must be at least one")


@dataclass(frozen=True)
class PaperSummaryCurves:
    guest_g: np.ndarray
    guest_f: np.ndarray
    guest_k: np.ndarray
    guest_to_host_g: np.ndarray


@dataclass(frozen=True)
class PaperFeatureResult:
    values: np.ndarray
    names: tuple[str, ...]
    observed: PaperSummaryCurves
    expected: PaperSummaryCurves
    radii: dict[str, np.ndarray]
    # Whether Rm, Rdm and Rddm came from real detected extrema rather than an
    # argmax fallback onto a grid endpoint. See _extract_k_features.
    k_extrema_interior: np.ndarray | None = None


@dataclass(frozen=True)
class LocalPaperFeatureResult:
    values: np.ndarray
    valid: np.ndarray
    point_counts: np.ndarray
    guest_counts: np.ndarray
    query_points: np.ndarray
    # (n_queries, 3) booleans for Rm, Rdm, Rddm. See _extract_k_features.
    k_extrema_interior: np.ndarray | None = None


def calculate_global_paper_features(
    coords,
    labels,
    domain,
    guest_marks=(2, 3),
    config=None,
):
    config = config or PaperFeatureConfig()
    points = _coordinate_array(coords)
    labels_array = np.asarray(labels)
    _validate_inputs(points, labels_array)

    guest_mask = np.isin(labels_array, np.atleast_1d(guest_marks))
    _validate_mark_counts(guest_mask)

    bounds = _domain_bounds(domain)
    _validate_k_radius(
        config.k_r_max,
        bounds[:, 1] - bounds[:, 0],
        "domain",
    )
    query_points = _regular_query_grid(
        bounds,
        config.f_grid_points_per_axis,
    )
    radii = _radius_grids(config)

    observed = _calculate_summary_curves(
        points,
        guest_mask,
        bounds,
        query_points,
        radii,
        config,
    )
    expected = calculate_expected_summary_curves(
        points,
        guest_mask,
        bounds,
        query_points,
        radii,
        config,
    )
    values, k_extrema_interior = extract_paper_features(
        observed,
        expected,
        radii,
        k_smoothing_reference_r_max=(
            config.k_smoothing_reference_r_max
        ),
        k_transform=config.k_transform,
        return_diagnostics=True,
        feature_method=config.feature_method,
    )

    return PaperFeatureResult(
        values=values.astype(np.float32),
        names=PAPER_FEATURE_NAMES,
        observed=observed,
        expected=expected,
        radii=radii,
        k_extrema_interior=k_extrema_interior,
    )


def calculate_expected_summary_curves(
    points,
    guest_mask,
    bounds,
    query_points,
    radii,
    config,
    relabeling_masks=None,
    csr_intensities=None,
    allow_sparse=False,
):
    if config.null_model == "csr":
        if csr_intensities is None:
            volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))
            csr_intensities = (
                float(np.sum(guest_mask)) / volume,
                float(np.sum(~guest_mask)) / volume,
            )
        return calculate_csr_baseline(
            radii,
            guest_intensity=csr_intensities[0],
            host_intensity=csr_intensities[1],
        )

    if relabeling_masks is None:
        return calculate_random_relabeling_baseline(
            points,
            int(np.sum(guest_mask)),
            bounds,
            query_points,
            radii,
            config,
        )
    return _calculate_relabeling_baseline_from_masks(
        points,
        relabeling_masks,
        bounds,
        query_points,
        radii,
        config,
        allow_sparse=allow_sparse,
    )


def _validate_k_radius(k_r_max, side_lengths, context):
    """Reject a K radius larger than the shortest side of the observation window.

    The translation-corrected estimator stays unbiased for r all the way up to
    r = L (measured ratio to analytic CSR: 1.000 at r/L = 1.0). Beyond L it
    breaks: pairs separated by more than the shortest side can only exist along
    diagonals, the positive-overlap filter discards them inconsistently, and the
    estimate biases downward by roughly 9% at r/L = 1.17.

    The conventional r <= L/4 guidance concerns variance and interpretability
    for clustered patterns rather than unbiasedness under CSR, so it is not
    enforced here; some datasets legitimately need r_max > L/4 to reach an
    interior K extremum. See ``inference/ROADMAP.md`` section 8.4.
    """
    shortest = float(np.min(side_lengths))
    if k_r_max > shortest:
        raise ValueError(
            f"k_r_max ({k_r_max:g}) exceeds the shortest {context} side "
            f"({shortest:g}). The translation-corrected K estimator is only "
            f"valid for r <= the shortest side."
        )


def calculate_csr_baseline(radii, guest_intensity, host_intensity):
    if guest_intensity <= 0 or host_intensity <= 0:
        raise ValueError("CSR intensities must be greater than zero")
    sphere_volumes = (4.0 / 3.0) * np.pi * radii["g"] ** 3
    cross_sphere_volumes = (
        (4.0 / 3.0) * np.pi * radii["cross_g"] ** 3
    )
    k_sphere_volumes = (4.0 / 3.0) * np.pi * radii["k"] ** 3
    guest_cdf = 1.0 - np.exp(-guest_intensity * sphere_volumes)
    return PaperSummaryCurves(
        guest_g=guest_cdf.copy(),
        guest_f=guest_cdf.copy(),
        guest_k=k_sphere_volumes,
        guest_to_host_g=(
            1.0 - np.exp(-host_intensity * cross_sphere_volumes)
        ),
    )


def create_random_relabeling_masks(
    point_count,
    guest_count,
    n_relabelings,
    random_seed,
):
    if guest_count < 2 or guest_count >= point_count:
        raise ValueError(
            "guest_count must leave at least two guests and one host"
        )
    rng = np.random.default_rng(random_seed)
    masks = np.zeros((n_relabelings, point_count), dtype=bool)
    for relabeling_index in range(n_relabelings):
        selected = rng.choice(
            point_count,
            size=guest_count,
            replace=False,
        )
        masks[relabeling_index, selected] = True
    return masks


def calculate_random_relabeling_baseline(
    points,
    guest_count,
    bounds,
    query_points,
    radii,
    config,
):
    relabeling_masks = create_random_relabeling_masks(
        len(points),
        guest_count,
        config.n_relabelings,
        config.random_seed,
    )
    return _calculate_relabeling_baseline_from_masks(
        points,
        relabeling_masks,
        bounds,
        query_points,
        radii,
        config,
    )


def _calculate_relabeling_baseline_from_masks(
    points,
    relabeling_masks,
    bounds,
    query_points,
    radii,
    config,
    allow_sparse=False,
):
    guest_g_curves = []
    guest_f_curves = []
    guest_k_curves = []
    guest_to_host_curves = []

    for guest_mask in np.asarray(relabeling_masks, dtype=bool):
        if allow_sparse:
            if int(np.sum(~guest_mask)) < 1:
                continue
        else:
            try:
                _validate_mark_counts(guest_mask)
            except ValueError:
                continue
        curves = _calculate_summary_curves(
            points,
            guest_mask,
            bounds,
            query_points,
            radii,
            config,
            allow_sparse=allow_sparse,
        )
        guest_g_curves.append(curves.guest_g)
        guest_f_curves.append(curves.guest_f)
        guest_k_curves.append(curves.guest_k)
        guest_to_host_curves.append(curves.guest_to_host_g)

    if not guest_g_curves:
        raise ValueError("no valid random relabelings were available")
    return PaperSummaryCurves(
        guest_g=np.median(np.stack(guest_g_curves), axis=0),
        guest_f=np.median(np.stack(guest_f_curves), axis=0),
        guest_k=np.median(np.stack(guest_k_curves), axis=0),
        guest_to_host_g=np.median(
            np.stack(guest_to_host_curves),
            axis=0,
        ),
    )


def calculate_local_paper_features(
    coords,
    labels,
    domain,
    query_points,
    guest_marks=(2, 3),
    config=None,
    local_config=None,
):
    config = config or PaperFeatureConfig()
    local_config = local_config or LocalPaperFeatureConfig()
    points = _coordinate_array(coords)
    labels_array = np.asarray(labels)
    _validate_inputs(points, labels_array)

    queries = np.asarray(query_points, dtype=float)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("query_points must have shape (n_points, 3)")

    bounds = _domain_bounds(domain)
    if np.any(queries < bounds[:, 0]) or np.any(queries > bounds[:, 1]):
        raise ValueError("query_points must lie inside the domain")

    guest_mask = np.isin(labels_array, np.atleast_1d(guest_marks))
    _validate_mark_counts(guest_mask)
    # Local K is estimated inside the local window, not the full domain, so the
    # nominal window extent is the relevant bound. Windows clipped at a domain
    # edge are smaller still; those raise inside _translation_corrected_k and
    # are reported through the `valid` mask.
    _validate_k_radius(
        config.k_r_max,
        np.full(3, 2.0 * local_config.neighborhood_radius),
        "local window",
    )
    radii = _radius_grids(config)
    point_tree = cKDTree(points)
    relabeling_masks = None
    if config.null_model == "random_label":
        relabeling_masks = create_random_relabeling_masks(
            len(points),
            int(np.sum(guest_mask)),
            config.n_relabelings,
            config.random_seed,
        )

    global_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))
    global_intensities = (
        float(np.sum(guest_mask)) / global_volume,
        float(np.sum(~guest_mask)) / global_volume,
    )
    values = np.full(
        (len(queries), len(PAPER_FEATURE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    valid = np.zeros(len(queries), dtype=bool)
    k_extrema_interior = np.zeros((len(queries), 3), dtype=bool)
    point_counts = np.zeros(len(queries), dtype=np.int32)
    guest_counts = np.zeros(len(queries), dtype=np.int32)

    for query_index, query_point in enumerate(queries):
        local_indices = np.asarray(
            point_tree.query_ball_point(
                query_point,
                r=local_config.neighborhood_radius,
                p=np.inf,
            ),
            dtype=int,
        )
        local_points = points[local_indices]
        local_guest_mask = guest_mask[local_indices]
        point_counts[query_index] = len(local_points)
        guest_counts[query_index] = int(np.sum(local_guest_mask))

        host_count = len(local_points) - guest_counts[query_index]
        if (
            len(local_points) < local_config.minimum_points
            or guest_counts[query_index]
            < local_config.minimum_guest_points
            or host_count < local_config.minimum_host_points
        ):
            continue

        local_bounds = _local_bounds(
            query_point,
            bounds,
            local_config.neighborhood_radius,
        )
        local_query_grid = _regular_query_grid(
            local_bounds,
            config.f_grid_points_per_axis,
        )
        query_config = replace(
            config,
            random_seed=config.random_seed + query_index + 1,
        )

        try:
            observed = _calculate_summary_curves(
                local_points,
                local_guest_mask,
                local_bounds,
                local_query_grid,
                radii,
                query_config,
                allow_sparse=True,
            )
            if config.null_model == "random_label":
                expected = calculate_expected_summary_curves(
                    local_points,
                    local_guest_mask,
                    local_bounds,
                    local_query_grid,
                    radii,
                    query_config,
                    relabeling_masks=(
                        relabeling_masks[:, local_indices]
                    ),
                    allow_sparse=True,
                )
            else:
                csr_intensities = global_intensities
                if local_config.csr_intensity_scope == "local":
                    local_volume = float(
                        np.prod(local_bounds[:, 1] - local_bounds[:, 0])
                    )
                    csr_intensities = (
                        guest_counts[query_index] / local_volume,
                        host_count / local_volume,
                    )
                expected = calculate_expected_summary_curves(
                    local_points,
                    local_guest_mask,
                    local_bounds,
                    local_query_grid,
                    radii,
                    query_config,
                    csr_intensities=csr_intensities,
                )
            (
                values[query_index],
                k_extrema_interior[query_index],
            ) = extract_paper_features(
                observed,
                expected,
                radii,
                k_smoothing_reference_r_max=(
                    config.k_smoothing_reference_r_max
                ),
                k_transform=config.k_transform,
                return_diagnostics=True,
                feature_method=config.feature_method,
            )
            valid[query_index] = True
        except ValueError:
            continue

    return LocalPaperFeatureResult(
        values=values,
        valid=valid,
        point_counts=point_counts,
        guest_counts=guest_counts,
        query_points=queries.astype(np.float32),
        k_extrema_interior=k_extrema_interior,
    )


def extract_paper_features(
    observed,
    expected,
    radii,
    k_smoothing_reference_r_max=10.0,
    k_transform="sqrt",
    return_diagnostics=False,
    feature_method="rapt",
):
    """Extract the 14 named features from observed and expected curves.

    ``feature_method="rapt"`` (the default) reproduces the R package exactly,
    including returning NaN for any K feature rapt reports as NA -- which happens
    whenever the difference curve has no local maximum where one is needed.
    Callers must decide what to do with those rows; rapt's own training script
    drops them.

    ``feature_method="legacy_port"`` is the earlier Python port, kept only to
    reproduce results generated with it. It returns a grid endpoint where rapt
    returns NA, and raises on non-finite values.

    With ``return_diagnostics=True`` returns ``(values, k_extrema_interior)``,
    a boolean triple for whether Rm, Rdm and Rddm are defined.
    """
    if feature_method not in FEATURE_METHODS:
        raise ValueError(f"feature_method must be one of {FEATURE_METHODS}")

    transformed_k = (
        transform_k(observed.guest_k, k_transform)
        - transform_k(expected.guest_k, k_transform)
    )
    if feature_method == "rapt":
        g_features = _extract_g_features_rapt(
            radii["g"], observed.guest_g, expected.guest_g
        )
        k_features, k_extrema_interior = _extract_k_features_rapt(
            radii["k"], transformed_k
        )
        cross_g_features = _extract_cross_g_features_rapt(
            radii["cross_g"], observed.guest_to_host_g, expected.guest_to_host_g
        )
    else:
        g_features = _extract_g_features(
            radii["g"], observed.guest_g, expected.guest_g
        )
        k_features, k_extrema_interior = _extract_k_features(
            radii["k"],
            transformed_k,
            smoothing_reference_r_max=k_smoothing_reference_r_max,
        )
        cross_g_features = _extract_cross_g_features(
            radii["cross_g"], observed.guest_to_host_g, expected.guest_to_host_g
        )
    f_features = _extract_f_features(observed.guest_f, expected.guest_f)

    features = np.concatenate(
        [g_features, f_features, k_features, cross_g_features]
    )
    if feature_method == "legacy_port" and not np.all(np.isfinite(features)):
        raise ValueError("paper spatial features contain non-finite values")
    if return_diagnostics:
        return features, k_extrema_interior
    return features


def _calculate_summary_curves(
    points,
    guest_mask,
    bounds,
    query_points,
    radii,
    config,
    allow_sparse=False,
):
    guest_points = points[guest_mask]
    host_points = points[~guest_mask]
    if allow_sparse:
        if len(host_points) < 1:
            raise ValueError("at least one host point is required")
    else:
        _validate_mark_counts(guest_mask)

    guest_g = np.zeros_like(radii["g"])
    guest_f = np.zeros_like(radii["g"])
    guest_k = np.zeros_like(radii["k"])
    guest_to_host_g = np.zeros_like(radii["cross_g"])

    if len(guest_points) > 0:
        guest_tree = cKDTree(guest_points)
        empty_space_distances = guest_tree.query(
            query_points,
            k=1,
            workers=config.workers,
        )[0]
        guest_f = _kaplan_meier_cdf(
            empty_space_distances,
            _boundary_distances(query_points, bounds),
            radii["g"],
        )

        host_tree = cKDTree(host_points)
        guest_to_host_distances = host_tree.query(
            guest_points,
            k=1,
            workers=config.workers,
        )[0]
        guest_to_host_g = _kaplan_meier_cdf(
            guest_to_host_distances,
            _boundary_distances(guest_points, bounds),
            radii["cross_g"],
        )

    if len(guest_points) > 1:
        guest_neighbor_distances = guest_tree.query(
            guest_points,
            k=2,
            workers=config.workers,
        )[0][:, 1]
        guest_g = _kaplan_meier_cdf(
            guest_neighbor_distances,
            _boundary_distances(guest_points, bounds),
            radii["g"],
        )
        k_points = _sample_k_points(
            guest_points,
            config.k_max_points,
            config.random_seed,
        )
        guest_k = _translation_corrected_k(
            k_points,
            bounds,
            radii["k"],
        )
    return PaperSummaryCurves(
        guest_g=guest_g,
        guest_f=guest_f,
        guest_k=guest_k,
        guest_to_host_g=guest_to_host_g,
    )


def _translation_corrected_k(points, bounds, radii):
    point_count = len(points)
    if point_count < 2:
        raise ValueError("at least two guest points are required for K")
    _validate_k_radius(
        float(radii[-1]),
        bounds[:, 1] - bounds[:, 0],
        "window",
    )

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=float(radii[-1]), output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros_like(radii)

    pair_offsets = np.abs(points[pairs[:, 0]] - points[pairs[:, 1]])
    side_lengths = bounds[:, 1] - bounds[:, 0]
    overlap_lengths = side_lengths - pair_offsets
    valid = np.all(overlap_lengths > 0.0, axis=1)
    pair_offsets = pair_offsets[valid]
    overlap_lengths = overlap_lengths[valid]
    if len(pair_offsets) == 0:
        return np.zeros_like(radii)

    distances = np.linalg.norm(pair_offsets, axis=1)
    volume = np.prod(side_lengths)
    overlap_volume = np.prod(overlap_lengths, axis=1)
    pair_weights = (
        2.0
        * volume
        * volume
        / (point_count * (point_count - 1) * overlap_volume)
    )

    order = np.argsort(distances)
    sorted_distances = distances[order]
    cumulative_weights = np.cumsum(pair_weights[order])
    radius_indices = np.searchsorted(
        sorted_distances,
        radii,
        side="right",
    )
    result = np.zeros_like(radii)
    included = radius_indices > 0
    result[included] = cumulative_weights[radius_indices[included] - 1]
    return result


def _sample_k_points(points, maximum_points, random_seed):
    if maximum_points is None or len(points) <= maximum_points:
        return points
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(
        len(points),
        size=maximum_points,
        replace=False,
    )
    return points[selected]


def _kaplan_meier_cdf(event_distances, censor_distances, radii):
    event_distances = np.asarray(event_distances, dtype=float)
    censor_distances = np.asarray(censor_distances, dtype=float)
    events = np.isfinite(event_distances) & (
        event_distances <= censor_distances
    )
    observed_times = np.minimum(event_distances, censor_distances)

    unique_times, inverse = np.unique(
        observed_times,
        return_inverse=True,
    )
    total_counts = np.bincount(inverse)
    event_counts = np.bincount(
        inverse,
        weights=events.astype(float),
        minlength=len(unique_times),
    )
    removed_before = np.concatenate(
        [[0], np.cumsum(total_counts[:-1])]
    )
    at_risk = len(observed_times) - removed_before
    survival_steps = 1.0 - np.divide(
        event_counts,
        at_risk,
        out=np.zeros_like(event_counts),
        where=at_risk > 0,
    )
    cumulative_distribution = 1.0 - np.cumprod(survival_steps)

    time_indices = np.searchsorted(
        unique_times,
        radii,
        side="right",
    ) - 1
    result = np.zeros_like(radii)
    included = time_indices >= 0
    result[included] = cumulative_distribution[time_indices[included]]
    return result


def _extract_g_features(radii, observed, expected):
    difference = observed - expected
    maximum_index = int(np.argmax(difference))
    minimum_index = int(np.argmin(difference))
    lower = min(maximum_index, minimum_index)
    upper = max(maximum_index, minimum_index) + 1
    zero_index = lower + int(
        np.argmin(np.abs(difference[lower:upper]))
    )
    return np.array(
        [
            difference[maximum_index],
            radii[maximum_index],
            difference[minimum_index],
            radii[zero_index],
        ],
        dtype=float,
    )


def _extract_f_features(observed, expected):
    difference = observed - expected
    minimum_index = int(np.argmin(difference))
    return np.array(
        [
            difference[minimum_index],
            observed[minimum_index],
        ],
        dtype=float,
    )


def _extract_cross_g_features(radii, observed, expected):
    difference = observed - expected
    minimum_difference = float(np.min(difference))
    percentile_radius = radii[int(np.argmin(np.abs(0.95 - observed)))]

    peak_index = int(np.argmax(np.abs(difference)))
    half_peak = difference[peak_index] / 2.0
    left_index = int(
        np.argmin(np.abs(difference[: peak_index + 1] - half_peak))
    )
    right_index = peak_index + int(
        np.argmin(np.abs(difference[peak_index:] - half_peak))
    )
    full_width_half_maximum = radii[right_index] - radii[left_index]

    return np.array(
        [
            minimum_difference,
            percentile_radius,
            full_width_half_maximum,
        ],
        dtype=float,
    )


def _extract_k_features(
    radii,
    transformed_k,
    smoothing_reference_r_max,
):
    """Extract Tm, Rm, Rdm, Rddm and Tdm from a transformed K difference curve.

    Returns ``(values, interior)`` where ``interior`` is a boolean triple
    reporting whether Rm, Rdm and Rddm came from a genuine detected local
    extremum rather than an ``argmax`` fallback.

    The fallback fires when the difference curve has no interior extremum
    because it is still rising at ``radii[-1]``, and it returns a grid endpoint
    (0.0 or ``k_r_max``) that is indistinguishable from a real measurement. At
    small ``k_r_max`` relative to the cluster scale this is the common case
    rather than the exception, so callers must check these flags before
    treating the radius-valued K features as measurements. See
    ``inference/ROADMAP.md`` section 8.3.
    """
    radius_scale = smoothing_reference_r_max / radii[-1]
    minimum_span = 3.0 / len(radii)
    initial_span = max(0.08 * radius_scale, minimum_span)
    initial_smoothed = _loess(
        radii,
        transformed_k,
        span=initial_span,
    )
    initial_peaks = _local_maxima(initial_smoothed, half_window=3)
    if len(initial_peaks) == 0:
        initial_peak = int(np.argmax(initial_smoothed))
    else:
        initial_peak = int(initial_peaks[0])

    span = max(
        (radii[initial_peak] / 7.0) * 0.3 * radius_scale,
        minimum_span,
    )
    smoothed = _loess(radii, transformed_k, span=span)
    peaks = _local_maxima(smoothed, half_window=3)
    rm_interior = bool(len(peaks))
    peak_index = (
        int(peaks[0])
        if len(peaks)
        else int(np.argmax(smoothed))
    )

    negative_smoothed = _loess(radii, -transformed_k, span=span)
    negative_peaks = _local_maxima(
        negative_smoothed,
        half_window=3,
    )
    negative_peak_index = (
        int(negative_peaks[0])
        if len(negative_peaks)
        else int(np.argmax(negative_smoothed))
    )

    derivative = np.gradient(smoothed, radii)
    negative_derivative_smoothed = _loess(
        radii,
        -derivative,
        span=span,
    )
    derivative_peaks = _local_maxima(
        negative_derivative_smoothed,
        half_window=3,
    )
    rdm_interior = bool(len(derivative_peaks))
    derivative_peak_index = (
        int(derivative_peaks[0])
        if len(derivative_peaks)
        else int(np.argmax(negative_derivative_smoothed))
    )

    second_derivative = np.gradient(
        -negative_derivative_smoothed,
        radii,
    )
    second_derivative_smoothed = _loess(
        radii,
        second_derivative,
        span=span,
    )
    third_derivative = np.gradient(
        second_derivative_smoothed,
        radii,
    )
    third_derivative_smoothed = _loess(
        radii,
        third_derivative,
        span=span,
    )
    third_derivative_peaks = _local_maxima(
        third_derivative_smoothed,
        half_window=3,
    )

    upper_bound = int(
        (derivative_peak_index + 2 * negative_peak_index) / 3
    )
    upper_bound = min(max(upper_bound, peak_index + 2), len(radii))
    candidates = third_derivative_peaks[
        (third_derivative_peaks > peak_index)
        & (third_derivative_peaks < upper_bound)
    ]
    rddm_interior = bool(len(candidates))
    if len(candidates):
        second_derivative_radius_index = int(candidates[0])
    else:
        search_start = peak_index + 1
        if search_start >= upper_bound:
            second_derivative_radius_index = peak_index
        else:
            search = third_derivative_smoothed[
                search_start:upper_bound
            ]
            second_derivative_radius_index = (
                search_start + int(np.argmax(search))
            )

    values = np.array(
        [
            smoothed[peak_index],
            radii[peak_index],
            radii[derivative_peak_index],
            radii[second_derivative_radius_index],
            -negative_derivative_smoothed[derivative_peak_index],
        ],
        dtype=float,
    )
    interior = np.array(
        [rm_interior, rdm_interior, rddm_interior],
        dtype=bool,
    )
    return values, interior


def _loess(x_values, y_values, span):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    point_count = len(x_values)
    neighbor_count = min(
        point_count,
        max(3, int(np.ceil(float(span) * point_count))),
    )
    fitted = np.empty(point_count, dtype=float)

    for index, center in enumerate(x_values):
        distances = np.abs(x_values - center)
        bandwidth = np.partition(
            distances,
            neighbor_count - 1,
        )[neighbor_count - 1]
        if bandwidth == 0.0:
            fitted[index] = y_values[index]
            continue

        scaled_distances = np.clip(distances / bandwidth, 0.0, 1.0)
        weights = (1.0 - scaled_distances**3) ** 3
        offsets = x_values - center
        design = np.column_stack(
            [
                np.ones(point_count),
                offsets,
                offsets**2,
            ]
        )
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = y_values * np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            weighted_design,
            weighted_values,
            rcond=None,
        )[0]
        fitted[index] = coefficients[0]

    return fitted


def _local_maxima(values, half_window):
    values = np.asarray(values)
    maxima = []
    for index in range(half_window, len(values) - half_window):
        window = values[
            index - half_window : index + half_window + 1
        ]
        if values[index] >= np.max(window):
            maxima.append(index)
    return np.asarray(maxima, dtype=int)


def _radius_grids(config):
    return {
        "g": np.linspace(0.0, config.g_r_max, config.g_num_radii),
        "k": np.linspace(0.0, config.k_r_max, config.k_num_radii),
        "cross_g": np.linspace(
            0.0,
            config.cross_g_r_max,
            config.cross_g_num_radii,
        ),
    }


def _regular_query_grid(bounds, points_per_axis):
    axes = []
    for lower, upper in bounds:
        spacing = (upper - lower) / points_per_axis
        axes.append(
            lower + (np.arange(points_per_axis) + 0.5) * spacing
        )
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.ravel() for axis in mesh])


def _local_bounds(query_point, domain_bounds, neighborhood_radius):
    return np.column_stack(
        [
            np.maximum(
                domain_bounds[:, 0],
                query_point - neighborhood_radius,
            ),
            np.minimum(
                domain_bounds[:, 1],
                query_point + neighborhood_radius,
            ),
        ]
    )


def _boundary_distances(points, bounds):
    lower_distances = points - bounds[:, 0]
    upper_distances = bounds[:, 1] - points
    return np.min(
        np.concatenate([lower_distances, upper_distances], axis=1),
        axis=1,
    )


def _coordinate_array(coords):
    if isinstance(coords, dict):
        return np.column_stack(
            [
                np.asarray(coords["x"], dtype=float),
                np.asarray(coords["y"], dtype=float),
                np.asarray(coords["z"], dtype=float),
            ]
        )
    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_points, 3)")
    return points


def _domain_bounds(domain):
    bounds = np.asarray(
        [domain[axis] for axis in ("x", "y", "z")],
        dtype=float,
    )
    if bounds.shape != (3, 2):
        raise ValueError("domain must contain x, y, and z bounds")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("domain upper bounds must exceed lower bounds")
    return bounds


def _validate_inputs(points, labels):
    if len(points) != len(labels):
        raise ValueError("coordinates and labels must have equal lengths")
    if len(points) < 3:
        raise ValueError("at least three points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates must be finite")


def _validate_mark_counts(guest_mask):
    guest_count = int(np.sum(guest_mask))
    host_count = len(guest_mask) - guest_count
    if guest_count < 2:
        raise ValueError("at least two guest points are required")
    if host_count < 1:
        raise ValueError("at least one host point is required")


# ==========================================================================
# Faithful ports of the rapt feature extractors
# ==========================================================================
#
# The functions above (`_extract_g_features`, `_extract_cross_g_features`,
# `_extract_k_features`, `_loess`) are the original Python port. Checked line by
# line against the R package they were ported from -- rapt, the reference
# implementation for Bennett, Proudian and Zimmerman (2023), Ultramicroscopy 247,
# 113687 -- they diverge in several places:
#
#   * When the K difference curve has no local maximum, rapt's `k3features`
#     returns NA and the training script drops the row with `complete.cases`.
#     The port fell back to `np.argmax` and returned a grid endpoint as though it
#     were a measurement. Under the Stage 1 settings this fabricated Rddm for
#     about 63% of patterns and Rdm for about 29%.
#   * rapt's smoothing span is `(Rm / 7) * 0.3` as a fraction of points. The port
#     multiplied it by a radius scale and clamped it from below.
#   * R's loess uses floor(span * n) neighbours; the port used ceil. For spans
#     above one R widens the bandwidth by sqrt(span); the port clipped at n.
#   * The Rddm window upper bound was clamped to at least Rm + 2 and truncated to
#     an integer, where rapt compares against the unrounded value.
#   * Every fallback (no negative peak, no derivative peak, no Rddm candidate)
#     returned an argmax where rapt propagates NA.
#
# The functions below reproduce rapt, including two small quirks in the R code
# that are kept deliberately so the features match the published ones:
#
#   * `g3features` looks up G_zero_diff_r with `which(diff == zero_diff)` over the
#     whole curve rather than the window between the extrema, so an identical
#     value earlier in the curve wins.
#   * `g3Xfeatures` computes the right-hand half-maximum index as
#     `ind + which(...)`, one step past `ind + which(...) - 1`, so GXGH_FWHM is one
#     radius step wider.
#
# Parity against the installed R package is tested in tests/test_rapt_parity.py.

FEATURE_METHODS = ("rapt", "legacy_port")


def _loess_local_fit(x_values, y_values, centre, span):
    """One degree-2 tricube local regression at `centre`: (value, slope).

    Follows R's loess (netlib dloess): the neighbourhood holds
    floor(n * span + 1e-5) points, and for spans above one the bandwidth is the
    distance to the farthest point scaled by sqrt(span). The epsilon matters:
    0.57 * 100 is 56.99999999999999 in floating point, and without it the
    neighbourhood is one point short of R's.
    """
    count = len(x_values)
    distances = np.abs(x_values - centre)
    if span > 1.0:
        bandwidth = distances.max() * np.sqrt(span)
    else:
        neighbors = max(min(count, int(np.floor(span * count + 1e-5))), 1)
        bandwidth = np.partition(distances, neighbors - 1)[neighbors - 1]
    if bandwidth <= 0.0:
        weights = (distances == 0.0).astype(float)
    else:
        weights = (1.0 - np.clip(distances / bandwidth, 0.0, 1.0) ** 3) ** 3
    offsets = x_values - centre
    design = np.column_stack([np.ones(count), offsets, offsets**2])
    root = np.sqrt(weights)
    coefficients = np.linalg.lstsq(
        design * root[:, None], y_values * root, rcond=None
    )[0]
    return coefficients[0], coefficients[1]


def _loess_rapt(x_values, y_values, span, surface="interpolate", cell=0.2):
    """R's `loess(y ~ x, span = span)` for one predictor, reproduced exactly.

    rapt calls loess with R's defaults, and the default ``surface =
    "interpolate"`` does not evaluate the local regression at every point. It
    builds a kd-tree over x -- the data range padded by 0.5% each side, split at
    the median point until a cell holds at most floor(n * span * cell) points --
    fits the local regression only at the cell vertices, recording value and
    slope, and blends between vertices with cubic Hermite interpolation.

    The two surfaces differ by up to about 0.2% of the curve range, and that is
    enough to move Rm by up to three grid steps and flip whether Rddm exists at
    all, so the interpolated surface is reproduced rather than approximated.
    ``surface="direct"`` evaluates the local fit at every point instead.

    Both match R to machine precision on the curves in tests/test_rapt_parity.py.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    count = len(x_values)
    if surface == "direct":
        return np.array([
            _loess_local_fit(x_values, y_values, centre, span)[0]
            for centre in x_values
        ])
    if surface != "interpolate":
        raise ValueError("surface must be 'interpolate' or 'direct'")

    order = np.argsort(x_values, kind="stable")
    ordered = x_values[order]
    cell_capacity = int(np.floor(count * span * cell))
    padding = 0.005 * max(
        ordered[-1] - ordered[0],
        1e-10 * max(abs(ordered[0]), abs(ordered[-1])) + 1e-30,
    )
    lower_box, upper_box = ordered[0] - padding, ordered[-1] + padding

    splits = []
    queue = [(1, count, lower_box, upper_box)]      # 1-based, as in ehg124
    while queue:
        low, high, left_vertex, right_vertex = queue.pop(0)
        leaf = (high - low + 1) <= cell_capacity
        if not leaf:
            middle = (low + high) // 2
            split = ordered[middle - 1]
            leaf = split == left_vertex or split == right_vertex
        if leaf:
            continue
        splits.append(split)
        queue.append((low, middle, left_vertex, split))
        queue.append((middle + 1, high, split, right_vertex))

    vertices = np.array(sorted({lower_box, upper_box, *splits}))
    fits = np.array([
        _loess_local_fit(ordered, y_values[order], vertex, span)
        for vertex in vertices
    ])

    cells = np.clip(
        np.searchsorted(vertices, x_values, side="left"), 1, len(vertices) - 1
    )
    left, right = vertices[cells - 1], vertices[cells]
    width = right - left
    t = (x_values - left) / width
    return (
        (1 - t) ** 2 * (1 + 2 * t) * fits[cells - 1, 0]
        + t ** 2 * (3 - 2 * t) * fits[cells, 0]
        + t * (1 - t) ** 2 * width * fits[cells - 1, 1]
        - t ** 2 * (1 - t) * width * fits[cells, 1]
    )


def _argmax_rapt(x_values, y_values, half_window, span):
    """rapt's `argmax`: all local maxima of a loess-smoothed curve.

    Returns (indices, smoothed) with 0-based indices in [w, n - w - 1], the
    Python equivalent of rapt's 1-based [w + 1, n - w]. Empty when there is no
    local maximum, which is where rapt's `x[1]` becomes NA.
    """
    smoothed = _loess_rapt(x_values, y_values, span)
    count = len(smoothed)
    indices = [
        index
        for index in range(half_window, count - half_window)
        if smoothed[index]
        >= smoothed[index - half_window : index + half_window + 1].max()
    ]
    return np.asarray(indices, dtype=int), smoothed


def _finite_deriv(x_values, y_values):
    """rapt's `finite_deriv`: one-sided at the ends, central in between."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    derivative = np.empty_like(y_values)
    derivative[0] = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
    derivative[-1] = (y_values[-1] - y_values[-2]) / (x_values[-1] - x_values[-2])
    derivative[1:-1] = (y_values[2:] - y_values[:-2]) / (x_values[2:] - x_values[:-2])
    return derivative


def _extract_k_features_rapt(radii, transformed_k):
    """A line-by-line port of rapt's `k3features`. NaN wherever rapt gives NA.

    Returns (values, interior) where values are Tm, Rm, Rdm, Rddm, Tdm and
    interior flags whether Rm, Rdm and Rddm are defined.
    """
    radii = np.asarray(radii, dtype=float)
    curve = np.asarray(transformed_k, dtype=float)
    missing = np.full(5, np.nan)
    if np.any(np.isinf(curve)):
        return missing, np.zeros(3, dtype=bool)

    first, _ = _argmax_rapt(radii, curve, 3, 0.08)
    if len(first) == 0:
        return missing, np.zeros(3, dtype=bool)

    span = (radii[first[0]] / 7.0) * 0.3
    peaks, smoothed = _argmax_rapt(radii, curve, 3, span)
    negative, _ = _argmax_rapt(radii, -curve, 3, span)
    derivative = _finite_deriv(radii, smoothed)
    derivative_peaks, derivative_smoothed = _argmax_rapt(
        radii, -derivative, 3, span
    )
    second = _finite_deriv(radii, -derivative_smoothed)
    _, second_smoothed = _argmax_rapt(radii, second, 3, span)
    third = _finite_deriv(radii, second_smoothed)
    third_peaks, _ = _argmax_rapt(radii, third, 3, span)

    def first_or_nan(indices):
        return int(indices[0]) if len(indices) else None

    peak = first_or_nan(peaks)
    derivative_peak = first_or_nan(derivative_peaks)
    negative_peak = first_or_nan(negative)

    tm = smoothed[peak] if peak is not None else np.nan
    rm = radii[peak] if peak is not None else np.nan
    rdm = radii[derivative_peak] if derivative_peak is not None else np.nan
    tdm = (
        -derivative_smoothed[derivative_peak]
        if derivative_peak is not None
        else np.nan
    )

    # rapt works in 1-based indices: lb = i[1], ub = (d + 2 * neg) / 3, and
    # candidates must satisfy lb < i < ub. Translating to 0-based indices j = i - 1
    # gives j + 1 > lb1 and j + 1 < ub1, evaluated without rounding ub.
    rddm = np.nan
    if peak is not None:
        lower_1 = peak + 1
        if derivative_peak is None or negative_peak is None:
            upper_1 = float(len(radii))
        else:
            upper_1 = ((derivative_peak + 1) + 2 * (negative_peak + 1)) / 3.0
        candidates = [
            j for j in third_peaks if lower_1 < (j + 1) < upper_1
        ]
        if candidates:
            rddm = radii[candidates[0]]

    values = np.array([tm, rm, rdm, rddm, tdm], dtype=float)
    interior = np.array(
        [np.isfinite(rm), np.isfinite(rdm), np.isfinite(rddm)], dtype=bool
    )
    return values, interior


def _extract_g_features_rapt(radii, observed, expected):
    """A port of rapt's `g3features`, including its whole-curve zero lookup."""
    difference = np.asarray(observed, dtype=float) - np.asarray(expected, dtype=float)
    maximum_index = int(np.argmax(difference))
    minimum_index = int(np.argmin(difference))
    if radii[minimum_index] == radii[maximum_index]:
        # rapt leaves zero_diff undefined here and errors; the row is lost.
        return np.array(
            [difference[maximum_index], radii[maximum_index],
             difference[minimum_index], np.nan],
            dtype=float,
        )
    lower = min(maximum_index, minimum_index)
    upper = max(maximum_index, minimum_index) + 1
    window = difference[lower:upper]
    zero_value = window[int(np.argmin(np.abs(window)))]
    zero_index = int(np.flatnonzero(difference == zero_value)[0])
    return np.array(
        [difference[maximum_index], radii[maximum_index],
         difference[minimum_index], radii[zero_index]],
        dtype=float,
    )


def _extract_cross_g_features_rapt(radii, observed, expected):
    """A port of rapt's `g3Xfeatures` (min, 95% radius, FWHM), quirk included."""
    observed = np.asarray(observed, dtype=float)
    difference = observed - np.asarray(expected, dtype=float)
    minimum_difference = float(np.min(difference))
    percentile_radius = radii[int(np.argmin(np.abs(0.95 - observed)))]

    peak = int(np.argmax(np.abs(difference)))
    half = difference[peak] / 2.0
    left = int(np.argmin(np.abs(difference[: peak + 1] - half)))
    # rapt: second <- ind + which(...), one past the true index.
    right = peak + int(np.argmin(np.abs(difference[peak:] - half))) + 1
    width = radii[right] - radii[left] if right < len(radii) else np.nan
    return np.array([minimum_difference, percentile_radius, width], dtype=float)
