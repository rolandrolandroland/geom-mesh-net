"""Tests for voxel-field construction from labeled point clouds."""

import numpy as np
import pytest

from geom_mesh_net.fields.point_cloud import (
    build_grid_axes,
    choose_point_cloud,
    estimate_guest_probability_grid,
    filter_point_cloud,
    voxelize_point_counts,
)


DOMAIN = {axis: np.array([0.0, 10.0]) for axis in "xyz"}


def make_coords(points):
    points = np.asarray(points, dtype=float)
    return {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}


# --------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------


def test_build_grid_axes_produces_edges_and_centers():
    edges, centers = build_grid_axes(DOMAIN, resolution=2.0)
    assert len(edges) == 3 and len(centers) == 3
    for axis_edges, axis_centers in zip(edges, centers):
        assert len(axis_edges) == 6          # 10 / 2 bins, plus one
        assert len(axis_centers) == 5
        assert axis_edges[0] == 0.0 and axis_edges[-1] == 10.0
        assert axis_centers[0] == pytest.approx(1.0)
        assert np.allclose(np.diff(axis_edges), 2.0)


def test_build_grid_axes_rejects_indivisible_resolution():
    with pytest.raises(ValueError, match="divisible by resolution"):
        build_grid_axes(DOMAIN, resolution=3.0)


def test_build_grid_axes_rejects_nonpositive_resolution():
    with pytest.raises(ValueError, match="resolution must be greater than zero"):
        build_grid_axes(DOMAIN, resolution=0.0)


def test_voxelize_point_counts_conserves_points():
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 10.0, size=(500, 3))
    edges, _ = build_grid_axes(DOMAIN, resolution=2.0)
    counts = voxelize_point_counts(make_coords(points), edges)
    assert counts.shape == (5, 5, 5)
    assert counts.sum() == len(points)


def test_voxelize_point_counts_places_a_point_in_the_right_voxel():
    edges, _ = build_grid_axes(DOMAIN, resolution=2.0)
    counts = voxelize_point_counts(make_coords([[1.0, 3.0, 5.0]]), edges)
    assert counts[0, 1, 2] == 1
    assert counts.sum() == 1


# --------------------------------------------------------------------------
# Point-cloud selection and filtering
# --------------------------------------------------------------------------


def test_choose_point_cloud_selects_the_named_source():
    original = make_coords([[1.0, 1.0, 1.0]])
    thinned = make_coords([[2.0, 2.0, 2.0]])
    assert choose_point_cloud("original", original, [0], thinned, [1])[1] == [0]
    assert choose_point_cloud("thinned", original, [0], thinned, [1])[1] == [1]
    with pytest.raises(ValueError, match="point source must be"):
        choose_point_cloud("both", original, [0], thinned, [1])


def test_filter_point_cloud_keeps_only_requested_marks():
    coords = make_coords([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    labels = np.array([0, 2, 3, 1])
    filtered, kept = filter_point_cloud(coords, labels, marks=(2, 3))
    assert list(kept) == [2, 3]
    assert np.allclose(filtered["x"], [2.0, 3.0])


def test_filter_point_cloud_all_is_a_passthrough():
    coords = make_coords([[1, 1, 1], [2, 2, 2]])
    labels = np.array([0, 2])
    filtered, kept = filter_point_cloud(coords, labels, marks="all")
    assert np.allclose(filtered["x"], coords["x"])
    assert np.array_equal(kept, labels)


def test_filter_point_cloud_rejects_length_mismatch():
    coords = make_coords([[1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError, match="same length"):
        filter_point_cloud(coords, np.array([0]), marks=(2,))


# --------------------------------------------------------------------------
# Guest probability field
# --------------------------------------------------------------------------


def test_guest_probability_is_a_probability_everywhere():
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 10.0, size=(2000, 3))
    labels = rng.choice([0, 2], size=2000)
    _, _, _, probability = estimate_guest_probability_grid(
        make_coords(points), labels, DOMAIN, resolution=1.0
    )
    assert probability.shape == (10, 10, 10)
    assert probability.min() >= 0.0
    assert probability.max() <= 1.0
    assert np.all(np.isfinite(probability))


def test_all_guest_cloud_gives_probability_one_where_points_exist():
    rng = np.random.default_rng(2)
    points = rng.uniform(0.0, 10.0, size=(3000, 3))
    labels = np.full(3000, 2)
    _, _, _, probability = estimate_guest_probability_grid(
        make_coords(points), labels, DOMAIN, resolution=1.0, bandwidth=1.0
    )
    assert probability.min() > 0.99


def test_all_host_cloud_gives_probability_zero():
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 10.0, size=(3000, 3))
    labels = np.zeros(3000, dtype=int)
    _, _, _, probability = estimate_guest_probability_grid(
        make_coords(points), labels, DOMAIN, resolution=1.0, bandwidth=1.0
    )
    assert probability.max() < 1e-9


def test_empty_voxels_fall_back_to_the_global_guest_fraction():
    """With no smoothing, a voxel containing no points has no local estimate.

    The implementation fills those with the global guest fraction rather than
    leaving a divide-by-zero, and that default must be exact.
    """
    coords = make_coords([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    labels = np.array([2, 0])
    _, _, _, probability = estimate_guest_probability_grid(
        coords, labels, DOMAIN, resolution=1.0, bandwidth=0.0
    )
    assert probability[0, 0, 0] == pytest.approx(1.0)
    assert probability[1, 1, 1] == pytest.approx(0.0)
    # An untouched voxel gets the global fraction, here 1 guest of 2 points.
    assert probability[5, 5, 5] == pytest.approx(0.5)


def test_clustered_guests_raise_local_probability_above_background():
    rng = np.random.default_rng(4)
    background = rng.uniform(0.0, 10.0, size=(4000, 3))
    clumped = np.clip(rng.normal(5.0, 0.6, size=(600, 3)), 0.0, 10.0)
    points = np.concatenate([background, clumped])
    labels = np.concatenate(
        [np.zeros(len(background), dtype=int), np.full(len(clumped), 2)]
    )
    _, _, _, probability = estimate_guest_probability_grid(
        make_coords(points), labels, DOMAIN, resolution=0.5, bandwidth=1.0
    )
    centre = probability[9:11, 9:11, 9:11].mean()
    corner = probability[:3, :3, :3].mean()
    assert centre > 0.5, centre
    assert centre > 10 * max(corner, 1e-6)


def test_guest_probability_rejects_negative_bandwidth():
    coords = make_coords([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="bandwidth"):
        estimate_guest_probability_grid(
            coords, np.array([2]), DOMAIN, resolution=1.0, bandwidth=-1.0
        )


def test_returned_grids_match_the_probability_shape():
    coords = make_coords([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xx, yy, zz, probability = estimate_guest_probability_grid(
        coords, np.array([2, 0]), DOMAIN, resolution=2.0
    )
    assert xx.shape == yy.shape == zz.shape == probability.shape
    # Coordinate grids must be voxel centers on an 'ij' meshgrid.
    assert xx[0, 0, 0] == pytest.approx(1.0)
    assert yy[0, 1, 0] == pytest.approx(3.0)
    assert zz[0, 0, 1] == pytest.approx(3.0)
