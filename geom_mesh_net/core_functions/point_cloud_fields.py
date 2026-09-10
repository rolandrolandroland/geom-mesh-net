"""Build voxel fields and point selections from labeled 3D point clouds."""

import numpy as np
from scipy.ndimage import gaussian_filter


def choose_point_cloud(
    source,
    original_coords,
    original_labels,
    thinned_coords,
    thinned_labels,
):
    if source == "original":
        return original_coords, original_labels
    if source == "thinned":
        return thinned_coords, thinned_labels
    raise ValueError("point source must be either 'original' or 'thinned'")


def filter_point_cloud(coords, labels, marks="all"):
    labels_array = np.asarray(labels)
    coordinate_lengths = {len(np.asarray(values)) for values in coords.values()}
    if coordinate_lengths != {len(labels_array)}:
        raise ValueError("coordinate arrays and labels must have the same length")

    if isinstance(marks, str) and marks == "all":
        return coords, labels_array

    selected_marks = np.atleast_1d(marks)
    selected = np.isin(labels_array, selected_marks)
    filtered_coords = {
        axis: np.asarray(values)[selected]
        for axis, values in coords.items()
    }
    return filtered_coords, labels_array[selected]


def build_grid_axes(domain, resolution):
    if resolution <= 0:
        raise ValueError("resolution must be greater than zero")

    edges = []
    centers = []
    for axis in ("x", "y", "z"):
        lower, upper = np.asarray(domain[axis], dtype=float)
        extent = upper - lower
        num_bins = int(round(extent / resolution))
        if num_bins <= 0 or not np.isclose(num_bins * resolution, extent):
            raise ValueError(
                f"domain extent for axis '{axis}' must be divisible by resolution"
            )

        axis_edges = np.linspace(lower, upper, num_bins + 1)
        edges.append(axis_edges)
        centers.append((axis_edges[:-1] + axis_edges[1:]) / 2)

    return edges, centers


def voxelize_point_counts(coords, edges):
    points = np.column_stack(
        [
            np.asarray(coords["x"]),
            np.asarray(coords["y"]),
            np.asarray(coords["z"]),
        ]
    )
    counts, _ = np.histogramdd(points, bins=edges)
    return counts


def estimate_guest_probability_grid(
    coords,
    labels,
    domain,
    resolution,
    guest_marks=(2, 3),
    bandwidth=1.5,
    truncate=4.0,
):
    if bandwidth < 0:
        raise ValueError("bandwidth must be greater than or equal to zero")

    edges, centers = build_grid_axes(domain, resolution)
    labels_array = np.asarray(labels)
    guest_coords, guest_labels = filter_point_cloud(
        coords,
        labels_array,
        marks=guest_marks,
    )

    total_counts = voxelize_point_counts(coords, edges)
    guest_counts = voxelize_point_counts(guest_coords, edges)

    if bandwidth > 0:
        sigma = bandwidth / resolution
        total_density = gaussian_filter(
            total_counts,
            sigma=sigma,
            mode="constant",
            truncate=truncate,
        )
        guest_density = gaussian_filter(
            guest_counts,
            sigma=sigma,
            mode="constant",
            truncate=truncate,
        )
    else:
        total_density = total_counts
        guest_density = guest_counts

    global_guest_fraction = (
        len(guest_labels) / len(labels_array)
        if len(labels_array) > 0
        else 0.0
    )
    guest_probability = np.full(
        total_density.shape,
        global_guest_fraction,
        dtype=np.float64,
    )
    np.divide(
        guest_density,
        total_density,
        out=guest_probability,
        where=total_density > np.finfo(np.float64).eps,
    )
    np.clip(guest_probability, 0.0, 1.0, out=guest_probability)

    xx, yy, zz = np.meshgrid(*centers, indexing="ij")
    return (
        xx.astype(np.float32),
        yy.astype(np.float32),
        zz.astype(np.float32),
        guest_probability.astype(np.float32),
    )
