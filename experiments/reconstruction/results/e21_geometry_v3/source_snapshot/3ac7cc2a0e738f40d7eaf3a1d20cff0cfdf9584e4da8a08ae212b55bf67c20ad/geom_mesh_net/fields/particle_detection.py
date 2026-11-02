"""Turn concentration fields or guest coordinates into measured precipitates.

The field detector uses a concentration threshold, selected on development data. It
does not borrow Gaussian-count significance thresholds for a fitted neural field.
Regions use face connectivity. Optional peak separation splits each connected region
by its local maxima; no voxel is ever assigned to a peak in another region.

The point comparators deliberately have small, explicit definitions: DBSCAN connects
core guest atoms and attaches border atoms to their nearest core; maximum separation
is the d_max-connected guest graph followed by an N_min cutoff. The latter has no
envelope growth or erosion. Both estimate an equivalent radius from the guest-cloud
second moment, R = sqrt(5 mean(|x-centre|^2) / 3), assuming uniform sampling inside a
sphere. A radial concentration gradient, background contamination, or truncation can
bias that estimate. These methods are comparators, not interchangeable estimators.

All geometry is inferred from the supplied observations. Truth enters only the
evaluation function. Observed compositions are descriptive measurements and are not
corrected for using those same labels to detect a particle.
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _coords(value, name="coords"):
    value = np.asarray(value, dtype=float)
    if value.size == 0:
        value = value.reshape(0, 3)
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (N, 3) array")
    return value


def _positive(value, name, allow_zero=False):
    if not np.isscalar(value) or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def _integer(value, name):
    number = _positive(value, name)
    if number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _observations(coords, guest):
    coords = _coords(coords)
    guest = np.asarray(guest)
    if guest.shape != (len(coords),) or not np.all(np.isin(guest, [0, 1])):
        raise ValueError("guest must contain one boolean or 0/1 label per coordinate")
    return coords, guest.astype(bool)


@dataclass
class ParticleGeometry:
    """Measured spherical equivalents, with method settings and ambiguity flags."""

    centres: np.ndarray
    radii: np.ndarray
    metadata: dict = dataclass_field(default_factory=dict)

    def __post_init__(self):
        self.centres = _coords(self.centres, "centres").copy()
        self.radii = np.asarray(self.radii, dtype=float).copy()
        if (self.radii.shape != (len(self.centres),) or not np.all(np.isfinite(self.radii))
                or np.any(self.radii <= 0)):
            raise ValueError("radii must contain one finite positive radius per centre")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def to_dict(self):
        return {"centres": self.centres.tolist(), "radii": self.radii.tolist(), "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value):
        return cls(value["centres"], value["radii"], value.get("metadata", {}))


def _region_seeds(values, region, grid, peak_distance):
    """One seed per flat local-maximum plateau, followed by distance suppression."""
    window = 2 * int(np.ceil(peak_distance / grid)) + 1
    masked = np.where(region, values, -np.inf)
    maxima = region & (masked == ndimage.maximum_filter(masked, size=window, mode="constant", cval=-np.inf))
    plateaus, count = ndimage.label(maxima, structure=np.ones((3, 3, 3), dtype=bool))
    candidates = []
    for idx, region_slice in enumerate(ndimage.find_objects(plateaus), 1):
        if region_slice is None:
            continue
        points = np.argwhere(plateaus[region_slice] == idx) + np.array([s.start for s in region_slice])
        centre = points.mean(axis=0)
        chosen = points[np.argmin(np.sum((points - centre) ** 2, axis=1))]
        candidates.append(chosen)
    if not count:
        return np.argwhere(region)[:1]
    candidates = np.asarray(candidates)
    order = np.argsort(-values[tuple(candidates.T)], kind="stable")
    seeds = []
    for candidate in candidates[order]:
        if not seeds or np.all(np.linalg.norm((np.asarray(seeds) - candidate) * grid, axis=1) >= peak_distance):
            seeds.append(candidate)
    return np.asarray(seeds)


def extract_field(field, *, grid=0.5, lower=0.0, threshold=0.5, min_volume=14.0,
                  peak_distance=None, valid=None):
    """Extract spheres from a probability grid evaluated at voxel centres.

    ``lower`` is the scalar or three-vector lower *edge* of the grid. Voxels strictly
    above ``threshold`` form regions; ``min_volume`` is in physical volume units.
    ``peak_distance=None`` leaves connected regions intact. Otherwise local maxima
    must dominate a cubic neighbourhood of that half-width, and region voxels are
    assigned to the nearest retained peak. This is a geometric split, not a watershed.
    Its settings must be calibrated independently of the final test cases.

    ``valid`` can exclude voxels unsupported by observations. No uncertainty or
    detection significance is inferred from the field values themselves.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 3 or any(n == 0 for n in values.shape):
        raise ValueError("field must be a nonempty three-dimensional probability grid")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("field probabilities must be finite and in [0, 1]")
    grid = _positive(grid, "grid")
    min_volume = _positive(min_volume, "min_volume", allow_zero=True)
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    lower = np.asarray(lower, dtype=float)
    if lower.ndim == 0:
        lower = np.full(3, lower)
    if lower.shape != (3,) or not np.all(np.isfinite(lower)):
        raise ValueError("lower must be a finite scalar or three-vector")
    if peak_distance is not None:
        peak_distance = _positive(peak_distance, "peak_distance")
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    else:
        valid = np.asarray(valid)
        if valid.shape != values.shape or not np.all(np.isin(valid, [0, 1])):
            raise ValueError("valid must be a boolean grid with the same shape as field")
        valid = valid.astype(bool)

    labels, count = ndimage.label((values > threshold) & valid)
    centres, radii, particles = [], [], []
    for region_id, region_slice in enumerate(ndimage.find_objects(labels), 1):
        if region_slice is None:
            continue
        region = labels[region_slice] == region_id
        points = np.argwhere(region)
        if len(points) * grid ** 3 < min_volume:
            continue
        local_values = values[region_slice]
        if peak_distance is None:
            seeds = np.array([points.mean(axis=0)])
            assignments = np.zeros(len(points), dtype=int)
        else:
            seeds = _region_seeds(local_values, region, grid, peak_distance)
            assignments = cKDTree(seeds).query(points)[1]
        offset = np.array([s.start for s in region_slice])
        for piece in range(len(seeds)):
            voxels = points[assignments == piece] + offset
            volume = len(voxels) * grid ** 3
            if not len(voxels) or volume < min_volume:
                continue
            centres.append(lower + (voxels.mean(axis=0) + 0.5) * grid)
            radii.append((3 * volume / (4 * np.pi)) ** (1 / 3))
            particles.append({"volume": float(volume), "voxel_count": len(voxels),
                              "mean_field": float(values[tuple(voxels.T)].mean()),
                              "touches_boundary": bool(np.any((voxels == 0) | (voxels == np.array(values.shape) - 1))),
                              "split_region": bool(len(seeds) > 1), "region_id": region_id})
    metadata = {"method": "field_threshold", "threshold": float(threshold), "grid": grid,
                "lower": lower.tolist(), "min_volume": min_volume, "peak_distance": peak_distance,
                "connected_regions": int(count), "particles": particles,
                "radius_estimator": "equal segmented volume sphere"}
    return ParticleGeometry(centres, radii, metadata)


def _graph_labels(tree, eps):
    pairs = tree.query_pairs(eps, output_type="ndarray")
    adjacency = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(tree.n, tree.n)).tocsr()
    return connected_components(adjacency, directed=False)[1]


def cluster_points(coords, guest, *, method="dbscan", eps=1.0, min_samples=5, min_cluster_size=10):
    """Guest-only DBSCAN or d_max/N_min maximum-separation comparator.

    DBSCAN's ``min_samples`` includes the point itself. Only core points connect
    clusters; chains of non-core border atoms cannot join them. A border point within
    ``eps`` of several clusters joins its nearest core (equal distances follow tree
    order). Both methods finally discard clusters smaller than ``min_cluster_size``.
    Coincident clusters with zero spatial extent are discarded and counted explicitly.
    """
    coords, guest = _observations(coords, guest)
    eps = _positive(eps, "eps")
    min_samples = _integer(min_samples, "min_samples")
    min_cluster_size = _integer(min_cluster_size, "min_cluster_size")
    if method not in {"dbscan", "maximum_separation"}:
        raise ValueError("method must be 'dbscan' or 'maximum_separation'")
    points = coords[guest]
    assignments = np.full(len(points), -1, dtype=int)
    if len(points):
        tree = cKDTree(points)
        if method == "maximum_separation":
            assignments = _graph_labels(tree, eps)
        else:
            core = tree.query_ball_point(points, eps, return_length=True) >= min_samples
            if core.any():
                core_tree = cKDTree(points[core])
                core_labels = _graph_labels(core_tree, eps)
                assignments[core] = core_labels
                border_ids = np.flatnonzero(~core)
                distance, neighbour = core_tree.query(points[border_ids])
                nearby = distance <= eps
                assignments[border_ids[nearby]] = core_labels[neighbour[nearby]]
    centres, radii, particles = [], [], []
    retained_atoms = degenerate = 0
    for label in np.unique(assignments[assignments >= 0]):
        members = points[assignments == label]
        if len(members) < min_cluster_size:
            continue
        centre = members.mean(axis=0)
        radius = np.sqrt(5 * np.mean(np.sum((members - centre) ** 2, axis=1)) / 3)
        if radius <= 0:
            degenerate += 1
            continue
        centres.append(centre)
        radii.append(radius)
        retained_atoms += len(members)
        particles.append({"guest_members": len(members)})
    return ParticleGeometry(centres, radii, {"method": method, "eps": eps, "min_samples": min_samples,
                            "min_cluster_size": min_cluster_size, "guest_count": len(points),
                            "unassigned_guest_count": len(points) - retained_atoms,
                            "degenerate_clusters": degenerate, "particles": particles,
                            "radius_estimator": "uniform sphere guest second moment",
                            "maximum_separation_variant": "connected guest graph and N_min; no envelope or erosion"})


def measure_particles(geometry, coords, guest, *, chunk=20000):
    """Count all supplied atoms in detected spheres and report guest fractions.

    Overlapping spheres assign each atom to the sphere with the smallest distance/R;
    ties go to the lowest particle index. Atoms outside every sphere remain unassigned.
    No observed atom is counted twice. Empty spheres have composition ``None``.
    """
    coords, guest = _observations(coords, guest)
    chunk = _integer(chunk, "chunk")
    counts = np.zeros(len(geometry.radii), dtype=int)
    guest_counts = counts.copy()
    overlap_count = 0
    for start in range(0, len(coords) if len(counts) else 0, chunk):
        distance = np.linalg.norm(coords[start:start + chunk, None] - geometry.centres[None], axis=-1) / geometry.radii
        overlap_count += int(np.sum(np.sum(distance <= 1, axis=1) > 1))
        closest = distance.argmin(axis=1)
        inside = distance[np.arange(len(distance)), closest] <= 1
        counts += np.bincount(closest[inside], minlength=len(counts))
        guest_counts += np.bincount(closest[inside & guest[start:start + chunk]], minlength=len(counts))
    return {"atom_counts": counts.tolist(), "guest_counts": guest_counts.tolist(),
            "composition": [float(g / n) if n else None for g, n in zip(guest_counts, counts)],
            "unassigned_atom_count": int(len(coords) - counts.sum()), "overlap_atom_count": overlap_count,
            "composition_definition": "observed guest fraction inside detected spheres; overlaps assigned once"}


def _sphere_intersection(r1, r2, distance):
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return 4 * np.pi * min(r1, r2) ** 3 / 3
    return np.pi * (r1 + r2 - distance) ** 2 * (distance ** 2 + 2 * distance * (r1 + r2)
                 - 3 * (r1 - r2) ** 2) / (12 * distance)


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def evaluate_particles(detected, true_centres, true_radii, *, true_composition=None, measured=None,
                       match_radius_factor=1.0):
    """Match spheres and return JSON-safe detection and measurement diagnostics.

    A pair is eligible when centre separation <= match_radius_factor * true radius.
    Matching maximises eligible pair count, then minimises total distance/true radius.
    Split/merge diagnostics count true spheres containing multiple detected centres /
    detected spheres containing multiple true centres. They flag geometrical ambiguity,
    not exact segmentation topology. Purity is matched intersection volume divided by
    detected volume; it is separate from chemical composition.

    With no true particles recall is one; with no detections precision is one only if
    there are also no true particles. Undefined matched-error summaries are ``None``.
    """
    truth = ParticleGeometry(true_centres, true_radii)
    factor = _positive(match_radius_factor, "match_radius_factor")
    n_true, n_found = len(truth.radii), len(detected.radii)
    distance = np.linalg.norm(truth.centres[:, None] - detected.centres[None], axis=-1)
    normalised = distance / truth.radii[:, None]
    eligible = normalised <= factor
    penalty = (min(n_true, n_found) + 1) * (factor + 1)
    rows, cols = linear_sum_assignment(np.where(eligible, normalised, penalty))
    rows, cols = rows[eligible[rows, cols]], cols[eligible[rows, cols]]
    matches = len(rows)
    precision = matches / n_found if n_found else float(n_true == 0)
    recall = matches / n_true if n_true else 1.0
    radius_error = detected.radii[cols] - truth.radii[rows]
    relative_error = radius_error / truth.radii[rows]
    centre_error = distance[rows, cols]
    intersections = np.array([_sphere_intersection(truth.radii[i], detected.radii[j], distance[i, j])
                              for i, j in zip(rows, cols)])
    true_volume = 4 * np.pi * truth.radii[rows] ** 3 / 3
    found_volume = 4 * np.pi * detected.radii[cols] ** 3 / 3
    purity = intersections / found_volume
    iou = intersections / (true_volume + found_volume - intersections)
    result = {"true_count": n_true, "detected_count": n_found, "matched_count": matches,
              "false_positive_count": n_found - matches, "missed_count": n_true - matches,
              "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
              "matched_true_indices": rows.tolist(), "matched_detected_indices": cols.tolist(),
              "radius_error": radius_error.tolist(), "relative_radius_error": relative_error.tolist(),
              "centre_error": centre_error.tolist(), "radius_bias": _mean(radius_error),
              "radius_mae": _mean(abs(radius_error)), "relative_radius_bias": _mean(relative_error),
              "relative_radius_mae": _mean(abs(relative_error)), "centre_mae": _mean(centre_error),
              "matched_geometric_purity": purity.tolist(), "mean_geometric_purity": _mean(purity),
              "matched_sphere_iou": iou.tolist(), "mean_sphere_iou": _mean(iou),
              "split_count": int(np.sum(np.sum(distance <= truth.radii[:, None], axis=1) > 1)),
              "merge_count": int(np.sum(np.sum(distance <= detected.radii[None], axis=0) > 1)),
              "match_radius_factor": factor,
              "ambiguity_definition": "split: multiple detected centres in true sphere; merge: multiple true centres in detected sphere"}
    if true_composition is not None:
        concentration = np.asarray(true_composition, dtype=float)
        if concentration.ndim == 0:
            concentration = np.full(n_true, concentration)
        if (concentration.shape != (n_true,) or not np.all(np.isfinite(concentration))
                or np.any((concentration < 0) | (concentration > 1))):
            raise ValueError("true_composition must be a probability or one probability per true particle")
        if measured is None or len(measured.get("composition", [])) != n_found:
            raise ValueError("composition evaluation requires measurements for every detected particle")
        errors = [float(measured["composition"][j] - concentration[i]) for i, j in zip(rows, cols)
                  if measured["composition"][j] is not None]
        result.update(composition_error=errors, composition_bias=_mean(errors),
                      composition_mae=_mean(np.abs(errors)), composition_measured_count=len(errors))
    return result
