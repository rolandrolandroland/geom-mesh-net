"""Exact guest probabilities for patterns simulated by clustersim.

For every atom of a simulated pattern, the probability that it is a guest,

    p*_j = P(atom j is a guest | atom positions, cluster geometry, rho_c, rho_b),

is the ground truth a reconstruction of the solute field is scored against. The
realised labels are only one random draw from it. ``voxelize_clusters.
generate_density_grid`` only approximates it, and gets it wrong inside clusters
(``reconstruction/ROADMAP.md`` section 3.2). It can be computed exactly, up to a
Monte Carlo error this module drives down, because clustersim's labelling is a
known random procedure applied to stored inputs:

- each cluster takes the atoms within its radius and picks round(rho_c * N) of
  them as guests with ``rng.choice(replace=False, p=w)``, where w = 1 - d / d_max;
- an atom picked by any cluster is a guest (``np.maximum`` over labels). Clusters
  draw independently, so overlapping clusters combine as a union;
- atoms outside every sphere are guests with probability rho_b.

``rng.choice`` without replacement is successive sampling: each pick is
proportional to weight among the atoms not yet picked. That has the same
distribution as keeping the n atoms whose exponential clocks E_i / w_i ring
first, which vectorises across replays. ``tests/test_field_oracle.py`` checks
both equivalences against exact enumeration.

Under successive sampling, inclusion probability increases with weight. Each
cluster's replay frequencies are therefore smoothed by isotonic regression on
weight, which removes most of the Monte Carlo noise without assuming a
functional form, and keeps the expected number of picks exactly n.

Supported configuration: clustersim's spherical clusters (unit axis weights,
exponents 2) with ``selection="sampled"``, and ``prob_function`` either
``"Gaussian_decay"`` (linear weights, the data_factory default) or ``"constant"``.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

PROB_FUNCTIONS = ("Gaussian_decay", "constant")


def isotonic_increasing(values):
    """Least-squares non-decreasing fit to ``values``, by pooling adjacent violators.

    Every point has equal weight. Each pool is replaced by its mean, so the sum of
    the fit equals the sum of the input.
    """
    values = np.asarray(values, dtype=float)
    means = np.empty(len(values))
    counts = np.empty(len(values), dtype=np.int64)
    top = 0
    for value in values:
        means[top], counts[top] = value, 1
        while top > 0 and means[top - 1] > means[top]:
            merged = counts[top - 1] + counts[top]
            means[top - 1] = (means[top - 1] * counts[top - 1] + means[top] * counts[top]) / merged
            counts[top - 1] = merged
            top -= 1
        top += 1
    return np.repeat(means[:top], counts[:top])


def point_distances(points, centre):
    """Distances from ``centre``, written as clustersim writes them.

    The expression matches clustersim term for term, so an atom sitting exactly on
    a cluster's radius is classified the same way the simulator classified it.
    """
    return (((points[:, 0] - centre[0]) ** 2) + ((points[:, 1] - centre[1]) ** 2)
            + ((points[:, 2] - centre[2]) ** 2)) ** (1 / 2)


def selection_weights(distances, prob_function="Gaussian_decay"):
    """clustersim's unnormalised selection weights for the atoms inside one cluster."""
    distances = np.asarray(distances, dtype=float)
    if prob_function == "Gaussian_decay":
        # assign_clust_points: 1 - d / r_weighted_max, where r_weighted_max is the
        # largest distance among the atoms inside. Despite the name, the decay is linear.
        d_max = distances.max() if distances.size else 0.0
        weights = 1.0 - distances / d_max if d_max > 0 else np.zeros_like(distances)
    elif prob_function == "constant":
        weights = np.ones_like(distances)
    else:
        raise ValueError(f"prob_function must be one of {PROB_FUNCTIONS}, got {prob_function!r}")
    if weights.size and weights.sum() == 0:
        weights = weights + 1.0  # clustersim does the same when every weight is zero
    return weights


def _tie_average(weights, probabilities):
    """Give atoms of equal weight the mean of their probabilities; returns unique weights too."""
    unique, inverse = np.unique(weights, return_inverse=True)
    means = np.bincount(inverse, weights=probabilities) / np.bincount(inverse)
    return unique, means, means[inverse]


def inclusion_probabilities(weights, n_selected, replays=1000, rng=None, smooth=True, chunk=250):
    """Probability that each atom is among ``n_selected`` picks by successive sampling.

    Picks are simulated with exponential clocks: in each replay, the ``n_selected``
    atoms with the smallest E_i / w_i are picked. With ``smooth`` the frequencies are
    fitted by isotonic regression on weight, then averaged over ties.
    """
    weights = np.asarray(weights, dtype=float)
    n_atoms = len(weights)
    if n_selected <= 0 or n_atoms == 0:
        return np.zeros(n_atoms)
    if n_selected >= n_atoms:
        return np.ones(n_atoms)
    if np.all(weights == weights[0]):
        return np.full(n_atoms, n_selected / n_atoms)  # simple random sampling, exactly

    rng = np.random.default_rng() if rng is None else rng
    with np.errstate(divide="ignore"):
        scale = 1.0 / weights  # a zero weight never rings while positive weights remain
    counts = np.zeros(n_atoms)
    for start in range(0, replays, chunk):
        m = min(chunk, replays - start)
        clocks = rng.exponential(size=(m, n_atoms)) * scale
        picked = np.argpartition(clocks, n_selected - 1, axis=1)[:, :n_selected]
        counts += np.bincount(picked.ravel(), minlength=n_atoms)
    frequencies = counts / replays
    if not smooth:
        return frequencies

    order = np.argsort(weights, kind="stable")
    fitted = np.empty(n_atoms)
    fitted[order] = isotonic_increasing(frequencies[order])
    return _tie_average(weights, fitted)[2]


@dataclass
class ClusterProfile:
    """One cluster's inclusion probability as a function of selection weight."""

    centre: np.ndarray
    radius: float
    d_max: float
    n_inside: int
    n_selected: int
    prob_function: str
    weights: np.ndarray = field(repr=False)      # unique weights, ascending
    inclusion: np.ndarray = field(repr=False)    # inclusion probability at each weight

    def at(self, distances):
        """Inclusion probability of a hypothetical atom at ``distances`` from the centre.

        Interpolated between the weights of the cluster's real atoms, so at those
        atoms it returns exactly their inclusion probability.
        """
        distances = np.asarray(distances, dtype=float)
        if self.n_inside == 0:
            return np.zeros_like(distances)
        if self.prob_function == "Gaussian_decay" and self.d_max > 0:
            w = 1.0 - distances / self.d_max
        else:
            w = np.full_like(distances, self.weights[0])
        return np.interp(w, self.weights, self.inclusion)


@dataclass
class FieldOracle:
    """Guest probabilities of one simulated pattern, at its atoms and anywhere else."""

    p: np.ndarray = field(repr=False)            # (N,) guest probability of each atom
    n_spheres: np.ndarray = field(repr=False)    # (N,) spheres containing each atom
    rho_b: float
    replays: int
    profiles: list = field(repr=False)

    @property
    def inside(self):
        return self.n_spheres > 0

    def field(self, points):
        """Guest probability at arbitrary points, with the same union and matrix rules."""
        points = _as_points(points)
        miss = np.ones(len(points))
        inside = np.zeros(len(points), dtype=bool)
        tree = cKDTree(points)
        for profile in self.profiles:
            idx = np.asarray(tree.query_ball_point(profile.centre, profile.radius + 1e-9), dtype=int)
            if idx.size == 0:
                continue
            d = point_distances(points[idx], profile.centre)
            keep = d <= profile.radius
            idx, d = idx[keep], d[keep]
            inside[idx] = True
            miss[idx] *= 1.0 - profile.at(d)
        return np.where(inside, 1.0 - miss, self.rho_b)


def _as_points(points):
    if isinstance(points, dict):
        return np.column_stack([np.asarray(points[a], dtype=float) for a in ("x", "y", "z")])
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a dict of x, y, z arrays or an (N, 3) array")
    return points


def replay_oracle(coords, centres, radii, rho_c, rho_b, replays=1000, seed=0,
                  prob_function="Gaussian_decay", smooth=True):
    """Exact per-atom guest probabilities for one clustersim pattern.

    ``coords`` and ``centres`` are dicts of x, y, z arrays, as stored in the
    pattern files, or (N, 3) arrays. ``radii`` are the realised cluster radii.
    """
    if prob_function not in PROB_FUNCTIONS:
        raise ValueError(f"prob_function must be one of {PROB_FUNCTIONS}, got {prob_function!r}")
    X, C = _as_points(coords), _as_points(centres)
    radii = np.asarray(radii, dtype=float)
    if len(C) != len(radii):
        raise ValueError("centres and radii must have the same length")
    rng = np.random.default_rng(seed)
    tree = cKDTree(X)
    miss = np.ones(len(X))
    n_spheres = np.zeros(len(X), dtype=np.int32)
    profiles = []
    for centre, radius in zip(C, radii):
        if radius <= 0:
            continue  # no atom lies within a radius of zero
        idx = np.asarray(tree.query_ball_point(centre, radius + 1e-9), dtype=int)
        d = point_distances(X[idx], centre) if idx.size else np.empty(0)
        keep = d <= radius
        idx, d = idx[keep], d[keep]
        n_inside = int(idx.size)
        n_selected = int(round(n_inside * rho_c, 0))  # clustersim's rounding, half to even
        if n_inside == 0:
            profiles.append(ClusterProfile(centre, float(radius), 0.0, 0, 0, prob_function,
                                           np.zeros(1), np.zeros(1)))
            continue
        weights = selection_weights(d, prob_function)
        if n_selected >= n_inside:
            pi = np.ones(n_inside)  # clustersim marks every atom inside as a guest
        else:
            pi = inclusion_probabilities(weights, n_selected, replays, rng, smooth)
        n_spheres[idx] += 1
        miss[idx] *= 1.0 - pi
        unique_w, unique_pi, _ = _tie_average(weights, pi)
        profiles.append(ClusterProfile(centre, float(radius), float(d.max()), n_inside, n_selected,
                                       prob_function, unique_w, unique_pi))
    p = np.where(n_spheres > 0, 1.0 - miss, float(rho_b))
    return FieldOracle(p=p, n_spheres=n_spheres, rho_b=float(rho_b), replays=replays, profiles=profiles)
