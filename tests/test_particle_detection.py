"""Independent geometry and measurement checks for field-to-particle extraction."""

import json

import numpy as np
import pytest

from geom_mesh_net.fields.particle_detection import (
    ParticleGeometry, cluster_points, evaluate_particles, extract_field, measure_particles,
)


def empty_geometry():
    return ParticleGeometry([], [])


def test_voxel_spheres_recover_locations_and_volume_without_truth_inputs():
    grid = 0.25
    lower = np.array([-2.0, 3.0, 0.0])
    points = np.stack(np.meshgrid(*[np.arange(48) + 0.5] * 3, indexing="ij"), axis=-1) * grid + lower
    centres = np.array([[1.0, 6.0, 3.0], [7.0, 12.0, 9.0]])
    radii = np.array([2.0, 1.5])
    inside = np.any(np.linalg.norm(points[..., None, :] - centres, axis=-1) < radii, axis=-1)
    geometry = extract_field(np.where(inside, 0.8, 0.05), grid=grid, lower=lower, threshold=0.4, min_volume=1)
    score = evaluate_particles(geometry, centres, radii)
    assert score["matched_count"] == score["detected_count"] == 2
    assert score["centre_mae"] < 1e-12
    assert score["relative_radius_mae"] < 0.01
    assert not any(p["touches_boundary"] for p in geometry.metadata["particles"])
    json.dumps(score, allow_nan=False)
    json.dumps(geometry.to_dict(), allow_nan=False)


def test_flat_connected_ridge_is_one_particle_even_with_peak_splitting():
    field = np.zeros((12, 5, 5))
    field[1:11, 2, 2] = 0.8
    geometry = extract_field(field, grid=1, threshold=0.5, min_volume=0, peak_distance=2)
    assert len(geometry.radii) == 1
    assert geometry.metadata["particles"][0]["voxel_count"] == 10
    assert not geometry.metadata["particles"][0]["split_region"]


def test_peak_split_preserves_region_volume_and_never_joins_disconnected_regions():
    points = np.stack(np.meshgrid(*[np.arange(25)] * 3, indexing="ij"), axis=-1)
    field = 0.8 * np.exp(-np.sum((points - [8, 12, 12]) ** 2, axis=-1) / 8)
    field += 0.8 * np.exp(-np.sum((points - [16, 12, 12]) ** 2, axis=-1) / 8)
    field[11:14, 12, 12] = 0.15  # A bridge joins two peaks above the region threshold.
    field[2, 2, 2] = 0.3         # A disconnected particle must keep its own seed.
    unsplit = extract_field(field, grid=1, threshold=0.1, min_volume=0)
    split = extract_field(field, grid=1, threshold=0.1, min_volume=0, peak_distance=3)
    assert len(unsplit.radii) == 2
    assert len(split.radii) == 3
    assert np.sum(split.radii ** 3) == pytest.approx(np.sum(unsplit.radii ** 3))
    assert np.any(np.all(split.centres == [2.5, 2.5, 2.5], axis=1))
    assert sum(p["split_region"] for p in split.metadata["particles"]) == 2


def test_invalid_voxels_minimum_volume_and_boundary_flag():
    field = np.zeros((10, 10, 10))
    field[:2, :2, :2] = 0.9
    field[5:8, 5:8, 5:8] = 0.9
    valid = np.ones_like(field, dtype=bool)
    valid[5:8, 5:8, 5:8] = False
    result = extract_field(field, grid=1, threshold=0.8, min_volume=8, valid=valid)
    assert len(result.radii) == 1
    assert result.metadata["particles"][0]["touches_boundary"]
    assert len(extract_field(field, grid=1, threshold=0.8, min_volume=9, valid=valid).radii) == 0
    assert len(extract_field(field, threshold=0.9).radii) == 0  # strict threshold


def test_second_moment_radius_recovers_uniform_spheres_for_both_point_methods():
    rng = np.random.default_rng(29)
    direction = rng.normal(size=(4000, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    sphere = direction * (2 * rng.random(4000) ** (1 / 3))[:, None]
    points = np.concatenate([sphere[:2000], sphere[2000:] + [8, 0, 0], [[100, 100, 100]]])
    guest = np.ones(len(points), dtype=bool)
    for method in ("dbscan", "maximum_separation"):
        geometry = cluster_points(points, guest, method=method, eps=0.5, min_samples=4, min_cluster_size=100)
        score = evaluate_particles(geometry, [[0, 0, 0], [8, 0, 0]], [2, 2])
        assert score["matched_count"] == score["detected_count"] == 2
        assert score["relative_radius_mae"] < 0.025
        assert score["centre_mae"] < 0.1
        assert geometry.metadata["unassigned_guest_count"] == 1


def test_noncore_border_chain_cannot_merge_dbscan_clusters():
    blob = np.array([[0, 0, 0], [.01, 0, 0], [-.01, 0, 0], [0, .01, 0], [0, 0, .01]])
    points = np.concatenate([blob, blob + [4, 0, 0], [[1, 0, 0], [2, 0, 0], [3, 0, 0]]])
    labels = np.ones(len(points), dtype=bool)
    dbscan = cluster_points(points, labels, method="dbscan", eps=1.1, min_samples=5, min_cluster_size=5)
    separation = cluster_points(points, labels, method="maximum_separation", eps=1.1, min_cluster_size=5)
    assert len(dbscan.radii) == 2
    assert len(separation.radii) == 1
    assert "no envelope" in separation.metadata["maximum_separation_variant"]


def test_maximum_separation_accepts_a_sparse_chain_that_dbscan_calls_noise():
    points = np.column_stack([np.arange(8) * 0.9, np.zeros((8, 2))])
    guest = np.ones(8, dtype=bool)
    assert len(cluster_points(points, guest, eps=1, min_samples=4, min_cluster_size=5).radii) == 0
    assert len(cluster_points(points, guest, method="maximum_separation", eps=1, min_cluster_size=5).radii) == 1


def test_guest_clustering_excludes_host_atoms_and_reports_zero_extent_clusters():
    points = np.zeros((10, 3))
    points[5:, 0] = np.arange(5)
    guest = np.arange(10) < 5
    geometry = cluster_points(points, guest, min_samples=2, min_cluster_size=3)
    assert len(geometry.radii) == 0
    assert geometry.metadata["guest_count"] == 5
    assert geometry.metadata["degenerate_clusters"] == 1


def test_composition_counts_hosts_and_assigns_overlapping_atoms_only_once():
    geometry = ParticleGeometry([[0, 0, 0], [1, 0, 0], [10, 0, 0]], [1, 1, 1])
    coords = [[0, 0, 0], [.5, 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0]]
    measurements = measure_particles(geometry, coords, [1, 0, 1, 0, 1], chunk=2)
    assert measurements["atom_counts"] == [2, 2, 0]
    assert measurements["guest_counts"] == [1, 1, 0]
    assert measurements["composition"] == [0.5, 0.5, None]
    assert measurements["overlap_atom_count"] == 3
    assert measurements["unassigned_atom_count"] == 1


def test_composition_can_use_independent_labels_without_changing_geometry():
    geometry = ParticleGeometry([[0, 0, 0]], [1])
    held_out = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    measured = measure_particles(geometry, held_out, [1, 1, 1, 0])
    score = evaluate_particles(geometry, [[0, 0, 0]], [1], true_composition=0.5, measured=measured)
    assert score["composition_bias"] == pytest.approx(0.25)
    assert score["composition_measured_count"] == 1


def test_matching_prioritises_eligible_match_count_over_nearest_pair():
    geometry = ParticleGeometry([[.55, 0, 0], [-.8, 0, 0]], [.25, 1])
    score = evaluate_particles(geometry, [[0, 0, 0], [.7, 0, 0]], [1, .25])
    assert score["matched_count"] == 2
    assert score["matched_detected_indices"] == [1, 0]
    assert score["radius_mae"] == 0


def test_geometric_purity_is_not_chemical_composition():
    geometry = ParticleGeometry([[0, 0, 0]], [2])
    score = evaluate_particles(geometry, [[0, 0, 0]], [1])
    assert score["mean_geometric_purity"] == pytest.approx(1 / 8)
    assert score["mean_sphere_iou"] == pytest.approx(1 / 8)
    assert score["relative_radius_bias"] == 1
    shifted = evaluate_particles(ParticleGeometry([[1, 0, 0]], [1]), [[0, 0, 0]], [1])
    assert shifted["mean_geometric_purity"] == pytest.approx(5 / 16)
    assert shifted["mean_sphere_iou"] == pytest.approx(5 / 27)


def test_split_and_merge_diagnostics_and_unmatched_detections_are_explicit():
    split = evaluate_particles(ParticleGeometry([[-.5, 0, 0], [.5, 0, 0]], [.5, .5]), [[0, 0, 0]], [2])
    assert split["split_count"] == 1
    assert split["false_positive_count"] == 1
    assert split["precision"] == 0.5
    assert split["recall"] == 1
    merged = evaluate_particles(ParticleGeometry([[0, 0, 0]], [2]), [[-1, 0, 0], [1, 0, 0]], [.5, .5])
    assert merged["merge_count"] == 1
    assert merged["missed_count"] == 2
    assert merged["radius_mae"] is None


def test_empty_inputs_have_explicit_finite_metrics_and_serialise():
    empty = empty_geometry()
    assert len(extract_field(np.zeros((2, 2, 2))).radii) == 0
    assert len(cluster_points([], []).radii) == 0
    assert measure_particles(empty, [[0, 0, 0]], [1])["unassigned_atom_count"] == 1
    score = evaluate_particles(empty, [], [])
    assert score["precision"] == score["recall"] == score["f1"] == 1
    assert score["centre_mae"] is None
    missed = evaluate_particles(empty, [[0, 0, 0]], [1])
    assert missed["precision"] == missed["recall"] == missed["f1"] == 0
    false_positive = evaluate_particles(ParticleGeometry([[0, 0, 0]], [1]), [], [])
    assert false_positive["precision"] == 0
    assert false_positive["recall"] == 1
    assert ParticleGeometry.from_dict(empty.to_dict()).centres.shape == (0, 3)
    json.dumps([score, missed, false_positive], allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"grid": 0}, {"threshold": np.nan}, {"threshold": 1.1}, {"min_volume": -1},
    {"peak_distance": 0}, {"lower": [0, 0]}, {"valid": np.ones((3, 3, 3))},
])
def test_invalid_field_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        extract_field(np.zeros((2, 2, 2)), **kwargs)


@pytest.mark.parametrize("field", [np.zeros((2, 2)), np.zeros((0, 2, 2)),
                                  np.full((2, 2, 2), np.nan), np.full((2, 2, 2), 1.01)])
def test_invalid_probability_grid_is_rejected(field):
    with pytest.raises(ValueError):
        extract_field(field)


@pytest.mark.parametrize("kwargs", [{"eps": 0}, {"min_samples": 1.5}, {"min_cluster_size": 0},
                                   {"method": "unspecified"}])
def test_invalid_point_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        cluster_points([[0, 0, 0]], [1], **kwargs)


def test_invalid_labels_geometry_and_measurements_are_rejected():
    with pytest.raises(ValueError, match="guest"):
        cluster_points([[0, 0, 0]], [0.5])
    with pytest.raises(ValueError, match="coords"):
        cluster_points([[0, np.nan, 0]], [1])
    with pytest.raises(ValueError, match="radii"):
        ParticleGeometry([[0, 0, 0]], [0])
    with pytest.raises(ValueError, match="composition"):
        evaluate_particles(ParticleGeometry([[0, 0, 0]], [1]), [[0, 0, 0]], [1], true_composition=0.2)
