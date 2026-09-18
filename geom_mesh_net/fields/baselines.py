"""Classical estimators of the guest-probability field (reconstruction Stage 1).

These are the baselines every reconstruction in ``experiments/reconstruction/`` is
compared with:

- B0: a constant, the observed guest fraction;
- B1: Nadaraya–Watson smoothing with one Gaussian bandwidth. The guest fraction
  near x is the kernel-weighted count of observed guests divided by the
  kernel-weighted count of observed atoms. This is the delocalisation estimator
  of standard atom probe practice;
- B2: the same estimator with a bandwidth that varies in space. The width is set
  by the distance to the k-th nearest observed guest atom: narrow inside
  clusters, where guests are dense, and wide in the matrix. Because atom
  positions are uniform, the distance to the k-th nearest atom of any kind would
  not vary at all (``experiments/reconstruction/ROADMAP.md``, Stage 1 correction).
  Its (k, c) grid comes in two versions: the one Stage 1 was measured with, and
  the wider one used from Stage 2 onward, because cross-validation kept choosing
  Stage 1's edge.

Kernel sums are computed on a grid. Atom and guest counts are binned, both are
Gaussian-filtered at every bandwidth, and the fields are trilinearly
interpolated at query points. A bandwidth that falls between two precomputed
fields is interpolated in log bandwidth.

Every hyperparameter is chosen by cross-validated log loss on observed atoms
only, so none of this ever reads the atoms it is scored on.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

BANDWIDTHS = (0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)
ADAPTIVE_K = (4, 8, 16, 32, 64)
ADAPTIVE_C = (0.25, 0.35, 0.5, 0.7, 1.0)
# Stage 1 chose an edge of that grid in 11 of 16 development cells, so it was widened for Stage 2
# onward (``pilot/check_b2_grid.py``, and the Stage 1 correction in the reconstruction roadmap).
# Stage 1's own numbers keep the grid they were measured with, which is why both exist.
ADAPTIVE_K_WIDE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
ADAPTIVE_C_WIDE = (0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0)
GRID = 0.5
EPS = 1e-6


def log_loss(y, q):
    """Mean binary cross-entropy in nats, with q clipped away from 0 and 1."""
    q = np.clip(np.asarray(q, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


class SmoothedCounts:
    """Gaussian-smoothed atom and guest counts of one point set, at several bandwidths."""

    def __init__(self, coords, guest, bandwidths=BANDWIDTHS, grid=GRID, lower=0.0, upper=60.0):
        coords = np.asarray(coords, dtype=float)
        guest = np.asarray(guest, dtype=bool)
        self.bandwidths = np.asarray(bandwidths, dtype=float)
        self.grid, self.lower = grid, lower
        n_bins = int(round((upper - lower) / grid))
        edges = np.linspace(lower, upper, n_bins + 1)
        atoms = np.histogramdd(coords, bins=[edges] * 3)[0].astype(np.float32)
        guests = np.histogramdd(coords[guest], bins=[edges] * 3)[0].astype(np.float32)
        self.atoms = [gaussian_filter(atoms, h / grid, mode="constant", truncate=4.0) for h in self.bandwidths]
        self.guests = [gaussian_filter(guests, h / grid, mode="constant", truncate=4.0) for h in self.bandwidths]
        self.fallback = float(guest.mean()) if guest.size else 0.0

    def _sample(self, field, points):
        coords = ((np.asarray(points, dtype=float) - self.lower) / self.grid - 0.5).T
        return map_coordinates(field, coords, order=1, mode="nearest")

    def _ratio(self, atoms, guests):
        out = np.full(len(atoms), self.fallback)
        ok = atoms > 1e-8
        out[ok] = guests[ok] / atoms[ok]
        return np.clip(out, 0.0, 1.0)

    def fixed(self, points, index):
        """B1 prediction at ``points`` with the bandwidth ``self.bandwidths[index]``."""
        return self._ratio(self._sample(self.atoms[index], points), self._sample(self.guests[index], points))

    def adaptive(self, points, widths):
        """Prediction with a per-point bandwidth, interpolated in log bandwidth."""
        widths = np.clip(np.asarray(widths, dtype=float), self.bandwidths[0], self.bandwidths[-1])
        log_h, log_grid = np.log(widths), np.log(self.bandwidths)
        upper = np.clip(np.searchsorted(log_grid, log_h, side="right"), 1, len(log_grid) - 1)
        t = (log_h - log_grid[upper - 1]) / (log_grid[upper] - log_grid[upper - 1])
        atoms, guests = np.zeros(len(widths)), np.zeros(len(widths))
        for j in np.unique(np.concatenate([upper - 1, upper])):
            weight = np.where(upper - 1 == j, 1 - t, 0.0) + np.where(upper == j, t, 0.0)
            use = weight > 0
            if use.any():
                atoms[use] += weight[use] * self._sample(self.atoms[j], points[use])
                guests[use] += weight[use] * self._sample(self.guests[j], points[use])
        return self._ratio(atoms, guests)


def guest_neighbour_distances(guest_coords, points, k_max=max(ADAPTIVE_K)):
    """Distance from each point to its 1st..k_max-th nearest guest, skipping the point itself.

    Returns an array of shape (n_points, k_max). Column k-1 is the distance to the
    k-th nearest guest other than a guest sitting exactly at the query point.
    """
    guest_coords = np.asarray(guest_coords, dtype=float)
    if len(guest_coords) == 0:
        return np.full((len(points), k_max), np.inf)
    k_query = min(k_max + 1, len(guest_coords))
    d, _ = cKDTree(guest_coords).query(points, k=k_query)
    d = d.reshape(len(points), k_query)
    self_hit = d[:, 0] == 0
    shifted = np.where(self_hit[:, None], d[:, 1:], d[:, :-1]) if k_query > k_max else d
    if shifted.shape[1] < k_max:  # fewer guests than k_max: pad with the farthest
        shifted = np.hstack([shifted, np.repeat(shifted[:, -1:], k_max - shifted.shape[1], axis=1)])
    return shifted


@dataclass
class BaselineFit:
    """Hyperparameters chosen by cross-validation, and the cross-validated losses behind them."""

    bandwidth: float
    cv_loss_fixed: dict
    adaptive_k: int
    adaptive_c: float
    cv_loss_adaptive: dict
    constant: float


def fit_baselines(coords, guest, folds=5, seed=0, lower=0.0, upper=60.0,
                  adaptive_k=ADAPTIVE_K, adaptive_c=ADAPTIVE_C):
    """Choose B1's bandwidth and B2's (k, c) by k-fold cross-validated log loss on observed atoms.

    ``adaptive_k`` and ``adaptive_c`` are B2's grid. They are arguments because a grid whose edge
    cross-validation keeps choosing is a grid that is too small, which is measured rather than
    assumed (``experiments/reconstruction/pilot/check_b2_grid.py``).
    """
    coords = np.asarray(coords, dtype=float)
    guest = np.asarray(guest, dtype=bool)
    fold_of = np.random.default_rng(seed).integers(0, folds, size=len(coords))
    fixed_loss = np.zeros(len(BANDWIDTHS))
    adaptive_loss = {(k, c): 0.0 for k in adaptive_k for c in adaptive_c}
    for fold in range(folds):
        train, valid = fold_of != fold, fold_of == fold
        counts = SmoothedCounts(coords[train], guest[train], lower=lower, upper=upper)
        weight = valid.sum() / len(coords)
        for i in range(len(BANDWIDTHS)):
            fixed_loss[i] += weight * log_loss(guest[valid], counts.fixed(coords[valid], i))
        distances = guest_neighbour_distances(coords[train][guest[train]], coords[valid],
                                              k_max=max(adaptive_k))
        for k in adaptive_k:
            for c in adaptive_c:
                q = counts.adaptive(coords[valid], c * distances[:, k - 1])
                adaptive_loss[(k, c)] += weight * log_loss(guest[valid], q)
    best_fixed = int(np.argmin(fixed_loss))
    best_k, best_c = min(adaptive_loss, key=adaptive_loss.get)
    return BaselineFit(
        bandwidth=float(BANDWIDTHS[best_fixed]),
        cv_loss_fixed={float(h): float(v) for h, v in zip(BANDWIDTHS, fixed_loss)},
        adaptive_k=int(best_k),
        adaptive_c=float(best_c),
        cv_loss_adaptive={f"k{k}_c{c}": float(v) for (k, c), v in adaptive_loss.items()},
        constant=float(guest.mean()),
    )


def predict_baselines(fit, coords, guest, points, lower=0.0, upper=60.0):
    """B0, B1 and B2 predictions at ``points`` from all observed atoms, using ``fit``'s choices."""
    coords = np.asarray(coords, dtype=float)
    guest = np.asarray(guest, dtype=bool)
    counts = SmoothedCounts(coords, guest, lower=lower, upper=upper)
    distances = guest_neighbour_distances(coords[guest], points)
    return {
        "B0": np.full(len(points), fit.constant),
        "B1": counts.fixed(points, BANDWIDTHS.index(fit.bandwidth)),
        "B2": counts.adaptive(points, fit.adaptive_c * distances[:, fit.adaptive_k - 1]),
    }
