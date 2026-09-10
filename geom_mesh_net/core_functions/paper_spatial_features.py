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


@dataclass(frozen=True)
class LocalPaperFeatureResult:
    values: np.ndarray
    valid: np.ndarray
    point_counts: np.ndarray
    guest_counts: np.ndarray
    query_points: np.ndarray


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
    values = extract_paper_features(
        observed,
        expected,
        radii,
        k_smoothing_reference_r_max=(
            config.k_smoothing_reference_r_max
        ),
    )

    return PaperFeatureResult(
        values=values.astype(np.float32),
        names=PAPER_FEATURE_NAMES,
        observed=observed,
        expected=expected,
        radii=radii,
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
            values[query_index] = extract_paper_features(
                observed,
                expected,
                radii,
                k_smoothing_reference_r_max=(
                    config.k_smoothing_reference_r_max
                ),
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
    )


def extract_paper_features(
    observed,
    expected,
    radii,
    k_smoothing_reference_r_max=10.0,
):
    g_features = _extract_g_features(
        radii["g"],
        observed.guest_g,
        expected.guest_g,
    )
    f_features = _extract_f_features(
        observed.guest_f,
        expected.guest_f,
    )
    transformed_k = (
        np.sqrt(np.maximum(observed.guest_k, 0.0))
        - np.sqrt(np.maximum(expected.guest_k, 0.0))
    )
    k_features = _extract_k_features(
        radii["k"],
        transformed_k,
        smoothing_reference_r_max=k_smoothing_reference_r_max,
    )
    cross_g_features = _extract_cross_g_features(
        radii["cross_g"],
        observed.guest_to_host_g,
        expected.guest_to_host_g,
    )

    features = np.concatenate(
        [
            g_features,
            f_features,
            k_features,
            cross_g_features,
        ]
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("paper spatial features contain non-finite values")
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

    return np.array(
        [
            smoothed[peak_index],
            radii[peak_index],
            radii[derivative_peak_index],
            radii[second_derivative_radius_index],
            -negative_derivative_smoothed[derivative_peak_index],
        ],
        dtype=float,
    )


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
