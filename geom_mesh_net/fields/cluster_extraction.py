"""Precipitates from a smoothed guest field, as a measurement would have to find them.

Stage 5.2 needs precipitate centres and radii for its boundary condition, and a matrix
domain for its data and its diffusion equation, without ever reading the true geometry.
Both come from the B1 field of Stage 1: Nadaraya-Watson smoothing of the observed guest
labels, at the bandwidth cross-validation chooses.

- **Threshold.** Otsu's threshold on the field's voxel values separates precipitates
  from matrix. Where the contrast is low, Otsu's threshold falls inside the matrix noise
  and finds precipitates that are not there, so ``method="significance"`` instead keeps
  voxels whose guest fraction is significantly above the matrix level. The noise of the
  smoothed fraction at a voxel is sqrt(m(1 - m) sum_j K_j^2) / sum_j K_j for matrix level m
  and kernel weights K_j of the atoms nearby.
- **Split.** Neighbouring precipitates, only a gap apart, merge under the smoothing, so
  each connected component is split at local maxima of the field: every voxel goes to
  its nearest peak.
- **Radius.** Each piece's radius is that of a sphere of the same volume, and its centre
  is its centroid.
- **Matrix domain.** The matrix domain is kept conservative. Voxels above the midpoint
  between the matrix level and the threshold are excluded, dilated by a voxel, so that a
  precipitate the split misses does not enter the matrix as an unexplained bump.
- **Own label left out.** A domain chosen from the labels selects on them: a matrix atom
  that happens to be a guest raises the field around itself and is more often excluded,
  so the admitted atoms under-represent guests. ``matrix_atoms`` therefore decides each
  observed atom's admission from the field of all the other atoms. Labels are independent
  given the true field, so an atom's admission then says nothing about its own label.

The settings were chosen on development patterns (``experiments/reconstruction/ROADMAP.md``,
Stage 5.2). Stage 4 compares cluster finding methods properly; this is the simplest
extraction good enough to test physics.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

GRID = 0.5
FOOTPRINT = 7            # voxels: local maxima closer than this belong to one precipitate
MIN_VOLUME = 14.0        # smallest precipitate kept, in cubic spacings (radius about 1.5)
EXCLUSION_FRACTION = 0.5 # matrix domain excludes voxels above matrix + fraction * (threshold - matrix)
EXCLUSION_DILATION = 1   # voxels
REGION_Z = 3.0           # significance method: precipitate voxels lie this many noise sds above the matrix level
SEED_Z = 5.0             # significance method: a precipitate needs a peak this many sds above it
EXCLUSION_Z = 1.5        # significance method: matrix domain excludes voxels above this many sds


def smoothed_counts(coords, guest, bandwidth, lower=0.0, upper=60.0, grid=GRID):
    """Gaussian-smoothed guest and atom counts on a voxel grid: (guests, atoms)."""
    coords = np.asarray(coords, dtype=float)
    guest = np.asarray(guest, dtype=bool)
    edges = np.linspace(lower, upper, int(round((upper - lower) / grid)) + 1)
    sigma = bandwidth / grid
    atoms = ndimage.gaussian_filter(np.histogramdd(coords, bins=[edges] * 3)[0], sigma, mode="constant", truncate=4.0)
    guests = ndimage.gaussian_filter(np.histogramdd(coords[guest], bins=[edges] * 3)[0], sigma, mode="constant",
                                     truncate=4.0)
    return guests, atoms


def guest_fraction(guests, atoms):
    return np.where(atoms > 1e-3, guests / np.maximum(atoms, 1e-12), 0.0)


def smoothed_field(coords, guest, bandwidth, lower=0.0, upper=60.0, grid=GRID):
    """B1 on a voxel grid: Gaussian-smoothed guest counts over Gaussian-smoothed atom counts.

    Returns (field, atoms). ``field`` is the guest fraction at each voxel centre, zero
    where no atoms are nearby; ``atoms`` is the smoothed atom count.
    """
    guests, atoms = smoothed_counts(coords, guest, bandwidth, lower, upper, grid)
    return guest_fraction(guests, atoms), atoms


def kernel_weights(bandwidth, grid=GRID, truncate=4.0):
    """The 1-D weights ``gaussian_filter`` uses, for offsets -radius..radius voxels."""
    sigma = bandwidth / grid
    radius = int(truncate * sigma + 0.5)
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return weights / weights.sum()


def squared_kernel_counts(coords, bandwidth, lower=0.0, upper=60.0, grid=GRID):
    """sum_j K(v - x_j)^2 at each voxel: with the smoothed atom count, the noise of the guest fraction."""
    edges = np.linspace(lower, upper, int(round((upper - lower) / grid)) + 1)
    counts = np.histogramdd(np.asarray(coords, dtype=float), bins=[edges] * 3)[0]
    weights = kernel_weights(bandwidth, grid) ** 2
    for axis in range(3):
        counts = ndimage.correlate1d(counts, weights, axis=axis, mode="constant")
    return counts


def otsu_threshold(values, bins=256):
    """The threshold that maximises the between-class variance of ``values``."""
    hist, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    below = np.cumsum(hist)
    above = below[-1] - below
    mean_below = np.cumsum(hist * mids) / np.maximum(below, 1)
    mean_above = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(above, 1)
    return float(mids[np.argmax(below * above * (mean_below - mean_above) ** 2)])


@dataclass
class Segmentation:
    """Precipitates found in a smoothed field, and the matrix domain around them."""

    centres: np.ndarray       # (K, 3)
    radii: np.ndarray         # (K,)
    threshold: float          # Otsu's threshold, or nan for the significance method
    matrix_level: float
    exclusion_level: object   # voxels above this are not matrix: a number, or a voxel grid
    valid: np.ndarray         # boolean voxel grid: enough atoms nearby to judge
    excluded: np.ndarray      # boolean voxel grid: not matrix
    exclusion_dilation: int = EXCLUSION_DILATION
    grid: float = GRID
    lower: float = 0.0

    def voxels(self, points):
        points = np.asarray(points, dtype=float)
        return np.clip(((points - self.lower) / self.grid).astype(int), 0, np.array(self.excluded.shape) - 1)

    def in_matrix(self, points):
        """True where ``points`` lie in the conservative matrix domain."""
        index = self.voxels(points)
        return ~self.excluded[index[:, 0], index[:, 1], index[:, 2]]

    def outside_precipitates(self, points, chunk=20000):
        """True where ``points`` lie outside every segmented sphere, in voxels with enough atoms.

        This is the domain of the diffusion equation: it reaches the segmented interfaces,
        where the boundary condition applies.
        """
        points = np.asarray(points, dtype=float)
        index = self.voxels(points)
        out = self.valid[index[:, 0], index[:, 1], index[:, 2]]
        for start in range(0, len(points) if len(self.radii) else 0, chunk):
            x = points[start:start + chunk]
            gap = np.sqrt(((x[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=-1)) - self.radii[None, :]
            out[start:start + chunk] &= gap.min(axis=1) > 0
        return out


def segment(field, atoms, grid=GRID, lower=0.0, footprint=FOOTPRINT, min_volume=MIN_VOLUME,
            exclusion_fraction=EXCLUSION_FRACTION, exclusion_dilation=EXCLUSION_DILATION, method="otsu",
            squared=None, region_z=REGION_Z, seed_z=SEED_Z, exclusion_z=EXCLUSION_Z):
    """Precipitates and the matrix domain from a B1 voxel field; see the module docstring.

    ``method="significance"`` needs ``squared``, the output of ``squared_kernel_counts``.
    """
    valid = atoms > 0.2 * np.median(atoms)
    matrix_level = float(np.median(field[valid]))
    if method == "otsu":
        threshold = otsu_threshold(field[valid])
        foreground = (field > threshold) & valid
        seeds = foreground
        exclusion_level = matrix_level + exclusion_fraction * (threshold - matrix_level)
    elif method == "significance":
        if squared is None:
            raise ValueError("the significance method needs the squared-kernel counts")
        threshold = float("nan")
        noise = np.sqrt(matrix_level * (1 - matrix_level) * squared) / np.maximum(atoms, 1e-12)
        z = np.where(valid, (field - matrix_level) / np.maximum(noise, 1e-12), 0.0)
        foreground = (z > region_z) & valid
        seeds = foreground & (z > seed_z)
        exclusion_level = matrix_level + exclusion_z * noise
    else:
        raise ValueError(f"unknown method {method!r}")

    labels = np.zeros(field.shape, dtype=np.int64)
    n = 0
    if seeds.any():
        smooth = np.where(foreground, field, 0.0)
        peaks = (smooth == ndimage.maximum_filter(smooth, size=footprint)) & seeds
        markers, n = ndimage.label(peaks)
        nearest = ndimage.distance_transform_edt(markers == 0, return_distances=False, return_indices=True)
        labels = np.where(foreground, markers[tuple(nearest)], 0)
    ids = np.arange(1, n + 1)
    volumes = ndimage.sum(np.ones(field.shape), labels, index=ids) * grid ** 3 if n else np.zeros(0)
    keep = ids[volumes >= min_volume]
    if len(keep):
        centres = np.array(ndimage.center_of_mass(np.ones(field.shape), labels, keep)) * grid + lower + grid / 2
        radii = (3 * volumes[keep - 1] / (4 * np.pi)) ** (1 / 3)
    else:
        centres, radii = np.zeros((0, 3)), np.zeros(0)

    excluded = (field > exclusion_level) | ~valid
    if exclusion_dilation:
        excluded = ndimage.binary_dilation(excluded, iterations=exclusion_dilation)
    return Segmentation(centres, radii, threshold, matrix_level, exclusion_level, valid, excluded, exclusion_dilation,
                        grid, lower)


def matrix_atoms(coords, guest, guests, atoms, bandwidth, segmentation, leave_own_label_out=True):
    """Which observed atoms enter the matrix domain, each judged without its own label.

    ``coords`` and ``guest`` are the observed atoms that made the smoothed counts
    ``guests`` and ``atoms``. The domain rule is ``segmentation.in_matrix``'s: an atom is
    excluded if any voxel within the dilation distance (in city-block steps) is above the
    exclusion level or lacks atoms. Here each voxel's guest fraction is recomputed without
    the atom being judged. The smoothing kernel is separable, so the atom's contribution to
    a voxel depends only on their offset. With ``leave_own_label_out=False`` the result
    equals ``segmentation.in_matrix(coords)``.
    """
    seg = segmentation
    y = np.asarray(guest, dtype=float)
    voxel = seg.voxels(coords)
    shape = np.array(seg.excluded.shape)
    weights = kernel_weights(bandwidth, seg.grid)
    radius = len(weights) // 2
    d = seg.exclusion_dilation
    excluded = np.zeros(len(y), dtype=bool)
    for offset in itertools.product(range(-d, d + 1), repeat=3):
        if sum(abs(o) for o in offset) > d:
            continue
        v = voxel + np.array(offset)
        inside = np.flatnonzero(np.all((v >= 0) & (v < shape), axis=1))
        v = v[inside]
        own = np.prod([weights[radius + o] if abs(o) <= radius else 0.0 for o in offset]) if leave_own_label_out else 0.0
        fraction = guest_fraction(guests[v[:, 0], v[:, 1], v[:, 2]] - own * y[inside],
                                  atoms[v[:, 0], v[:, 1], v[:, 2]] - own)
        level = seg.exclusion_level if np.ndim(seg.exclusion_level) == 0 else seg.exclusion_level[v[:, 0], v[:, 1], v[:, 2]]
        bad = (fraction > level) | ~seg.valid[v[:, 0], v[:, 1], v[:, 2]]
        excluded[inside[bad]] = True
    return ~excluded


def resolve_overlaps(centres, radii, gap=0.0, max_rounds=100):
    """Separated spheres from segmented ones, for a boundary condition that needs whole interfaces.

    First, a pair is merged while either centre lies inside the other sphere, keeping the
    total volume at the volume-weighted centre: a split precipitate becomes one again.
    Then every sphere still closer than ``gap`` to a neighbour shrinks, each pair in
    proportion to its radii, until none is. Returns (centres, radii).
    """
    centres, radii = np.asarray(centres, dtype=float).copy(), np.asarray(radii, dtype=float).copy()
    while len(radii) > 1:
        distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(distance, np.inf)
        inside = distance < np.maximum(radii[:, None], radii[None, :])
        if not inside.any():
            break
        i, j = np.unravel_index(np.argmin(np.where(inside, distance, np.inf)), distance.shape)
        volume = radii[[i, j]] ** 3
        centres[i] = (centres[[i, j]] * volume[:, None]).sum(axis=0) / volume.sum()
        radii[i] = volume.sum() ** (1 / 3)
        centres, radii = np.delete(centres, j, axis=0), np.delete(radii, j)
    for _ in range(max_rounds):
        if len(radii) < 2:
            break
        distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(distance, np.inf)
        factor = np.minimum(1.0, ((distance - gap) / (radii[:, None] + radii[None, :])).min(axis=1))
        if factor.min() >= 1.0:
            break
        radii = radii * np.where(factor < 1.0, factor * (1 - 1e-9), 1.0)
    return centres, radii


def match_precipitates(true_centres, true_radii, centres):
    """Hungarian matching on centre distance; a pair matches only if closer than the true radius.

    Returns (true_index, found_index) arrays of the matched pairs.
    """
    true_centres, centres = np.asarray(true_centres, dtype=float), np.asarray(centres, dtype=float)
    if not len(true_centres) or not len(centres):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    distance = np.linalg.norm(true_centres[:, None, :] - centres[None, :, :], axis=-1)
    cost = np.where(distance < np.asarray(true_radii)[:, None], distance, 1e9)
    rows, cols = linear_sum_assignment(cost)
    ok = cost[rows, cols] < 1e9
    return rows[ok], cols[ok]
