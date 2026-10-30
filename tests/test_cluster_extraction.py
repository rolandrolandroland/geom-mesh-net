"""Tests for precipitate extraction from a smoothed guest field (Stage 5.2).

The matrix domain is checked against the property it exists for: a domain chosen from the
labels must not select on them. Synthetic patterns with a known uniform matrix
concentration make the bias measurable.
"""

import numpy as np
from scipy import ndimage

from geom_mesh_net.fields import cluster_extraction as ce

BOX = 40.0


def synthetic_pattern(seed=0, density=1.0, inside=0.6, matrix=0.1, radius=3.0, count=8):
    """Uniform atoms, separated spheres with guest probability ``inside``, a uniform matrix elsewhere."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, BOX, size=(rng.poisson(density * BOX ** 3), 3))
    centres = []
    while len(centres) < count:
        c = rng.uniform(radius + 2, BOX - radius - 2, size=3)
        if all(np.linalg.norm(c - c2) > 2 * radius + 3 for c2 in centres):
            centres.append(c)
    centres = np.array(centres)
    in_sphere = (np.linalg.norm(x[:, None, :] - centres[None], axis=-1) < radius).any(axis=1)
    guest = rng.random(len(x)) < np.where(in_sphere, inside, matrix)
    return x, guest, in_sphere, centres, np.full(count, radius)


def test_kernel_weights_are_the_ones_gaussian_filter_applies():
    impulse = np.zeros(101)
    impulse[50] = 1.0
    for bandwidth in (1.0, 1.5, 2.0):
        weights = ce.kernel_weights(bandwidth)
        radius = len(weights) // 2
        smoothed = ndimage.gaussian_filter(impulse, bandwidth / ce.GRID, mode="constant", truncate=4.0)
        assert np.allclose(smoothed[50 - radius:51 + radius], weights, rtol=0, atol=1e-15)
        assert np.all(smoothed[:50 - radius] == 0)


def test_segmentation_finds_planted_spheres_at_high_contrast():
    x, guest, _, centres, radii = synthetic_pattern(seed=1, density=3.0, inside=0.9, matrix=0.05)
    guests, atoms = ce.smoothed_counts(x, guest, 1.0, upper=BOX)
    seg = ce.segment(ce.guest_fraction(guests, atoms), atoms)
    rows, cols = ce.match_precipitates(centres, radii, seg.centres)
    assert len(rows) == len(centres) == len(seg.radii)
    assert np.all(np.abs(seg.radii[cols] / radii[rows] - 1) < 0.15)
    assert np.all(np.linalg.norm(seg.centres[cols] - centres[rows], axis=1) < 0.5)


def test_matrix_atoms_with_own_labels_restored_is_the_voxel_rule():
    x, guest, _, _, _ = synthetic_pattern(seed=2)
    guests, atoms = ce.smoothed_counts(x, guest, 1.5, upper=BOX)
    seg = ce.segment(ce.guest_fraction(guests, atoms), atoms)
    same = ce.matrix_atoms(x, guest, guests, atoms, 1.5, seg, leave_own_label_out=False)
    assert np.array_equal(same, seg.in_matrix(x))


def test_leaving_out_one_label_matches_smoothing_without_that_atom():
    x, guest, _, _, _ = synthetic_pattern(seed=3)
    guests, atoms = ce.smoothed_counts(x, guest, 1.5, upper=BOX)
    seg = ce.segment(ce.guest_fraction(guests, atoms), atoms)
    fast = ce.matrix_atoms(x, guest, guests, atoms, 1.5, seg)
    changed = np.flatnonzero(fast != seg.in_matrix(x))
    assert len(changed) > 0
    rng = np.random.default_rng(0)
    for i in np.concatenate([rng.choice(changed, 4, replace=False), rng.choice(len(x), 4, replace=False)]):
        keep = np.arange(len(x)) != i
        g, a = ce.smoothed_counts(x[keep], guest[keep], 1.5, upper=BOX)
        excluded = (ce.guest_fraction(g, a) > seg.exclusion_level) | ~seg.valid
        excluded = ndimage.binary_dilation(excluded, iterations=seg.exclusion_dilation)
        assert fast[i] == (not excluded[tuple(seg.voxels(x[i][None])[0])])


def test_the_voxel_rule_selects_on_labels_and_leaving_them_out_does_not():
    scores = {"voxel": [], "own label left out": []}
    for seed in range(4):
        x, guest, in_sphere, _, _ = synthetic_pattern(seed=10 + seed)
        guests, atoms = ce.smoothed_counts(x, guest, 1.5, upper=BOX)
        seg = ce.segment(ce.guest_fraction(guests, atoms), atoms)
        for name, admitted in (("voxel", seg.in_matrix(x)), ("own label left out", ce.matrix_atoms(x, guest, guests, atoms, 1.5, seg))):
            y = guest[admitted & ~in_sphere]
            scores[name].append((y.sum() - 0.1 * len(y)) / np.sqrt(0.09 * len(y)))
    assert max(scores["voxel"]) < -3
    assert np.all(np.abs(scores["own label left out"]) < 3)
