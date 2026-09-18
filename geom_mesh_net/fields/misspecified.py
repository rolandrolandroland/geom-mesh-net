"""Matrix fields that break the Stage 5 physics: controls for the acceptance rule.

Gate 5.2(iii) asks whether a physics-informed fit can tell, from the data it has, that its
physics does not hold (``experiments/reconstruction/ROADMAP.md``, Stage 5.2). Three matrix
fields replace the screened diffusion field of ``geom_mesh_net.fields.physics`` on the same
geometry:

- ``InverseSquareField`` gives each precipitate a profile decaying as (R/r)^2 instead of
  (R/r) exp(-(r - R)/xi). The amplitudes are solved as in ``physics``, so every surface
  mean still equals its Gibbs-Thomson value: the boundary condition holds and only the
  equation fails. Applied to one profile,

      (lap - xi^-2) (R/r)^2 = (R^2 / r^4) (2 - r^2 / xi^2),

  which vanishes only on the sphere r = sqrt(2) xi, whatever xi is chosen. The violation
  is nevertheless too slight to detect: with the screening length refitted, the screened
  family comes within a few nats of this field over a whole pattern.
- ``ShuffledSurfaceField`` solves the screened equation exactly, but gives each precipitate
  the Gibbs-Thomson value of another precipitate's radius, drawn by a random permutation.
  The equation holds and the surface values keep their distribution; only the capillarity
  law, the dependence of the surface value on the radius, fails.
- ``SmoothedNoiseField`` is Gaussian-smoothed white noise with a set mean and standard
  deviation. It has no relation to the precipitates, so both the equation and the
  boundary condition fail. Without sources, the screened equation allows no interior
  maximum above c_inf and no minimum below it; the noise has both.

Surface means of (R_j / r_j)^2 over sphere k, whose centre is a distance d from x_j,
have a closed form: R_j^2 ln((d + R_k)/(d - R_k)) / (2 R_k d).
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from geom_mesh_net.fields.physics import (CHUNK, _points, check_separated, coupling_matrix, gibbs_thomson, kernel_sum,
                                         screening_length)


def inverse_square_coupling(centres, radii):
    """M with M[k, k] = 1 and M[k, j] = mean over precipitate k's surface of (R_j / r_j)^2."""
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    check_separated(centres, radii)
    distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distance, 2 * radii.max() + 1.0)  # any d > R_k; the diagonal is overwritten
    rk = radii[:, None]
    matrix = radii[None, :] ** 2 * np.log((distance + rk) / (distance - rk)) / (2 * rk * distance)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def inverse_square_sum(points, centres, radii, amplitudes, chunk=CHUNK):
    """sum_k amplitudes[k] (R_k / r_k)^2 at ``points``."""
    points, centres = _points(points), _points(centres)
    radii, amplitudes = np.asarray(radii, dtype=float), np.asarray(amplitudes, dtype=float)
    out = np.empty(len(points))
    squared_centres = (centres ** 2).sum(axis=1)
    for start in range(0, len(points), chunk):
        x = points[start:start + chunk]
        squared = (x ** 2).sum(axis=1)[:, None] + squared_centres[None, :] - 2 * x @ centres.T
        out[start:start + chunk] = (radii[None, :] ** 2 / np.maximum(squared, 1e-24)) @ amplitudes
    return out


@dataclass
class InverseSquareField:
    """c(x) = c_inf + sum_k a_k (R_k / r_k)^2, with Gibbs-Thomson surface means."""

    centres: np.ndarray
    radii: np.ndarray
    c_eq: float
    ell: float
    c_inf: float
    amplitudes: np.ndarray

    kind = "inverse_square"

    @property
    def surface_values(self):
        return gibbs_thomson(self.radii, self.c_eq, self.ell)

    def __call__(self, points):
        return self.c_inf + inverse_square_sum(points, self.centres, self.radii, self.amplitudes)

    def surface_means(self, nodes=200):
        """Mean of c over each precipitate's surface by Gauss-Legendre integration, not the closed form."""
        mu, weights = np.polynomial.legendre.leggauss(nodes)
        distance = np.sqrt(((self.centres[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=-1))
        means = self.c_inf + self.amplitudes.astype(float)
        for k in range(len(self.radii)):
            others = np.flatnonzero(np.arange(len(self.radii)) != k)
            d = distance[k, others][:, None]
            squared = self.radii[k] ** 2 + d ** 2 - 2 * self.radii[k] * d * mu[None, :]
            averaged = 0.5 * ((self.radii[others][:, None] ** 2 / squared) * weights[None, :]).sum(axis=1)
            means[k] += averaged @ self.amplitudes[others]
        return means

    def to_dict(self):
        return {"kind": self.kind, "centres": self.centres.tolist(), "radii": self.radii.tolist(), "c_eq": self.c_eq,
                "ell": self.ell, "c_inf": self.c_inf, "amplitudes": self.amplitudes.tolist()}


def inverse_square_to_matrix_mean(centres, radii, matrix_points, ell, supersaturation, matrix_mean):
    """The inverse-square field with c_inf = supersaturation * c_eq and the given mean over ``matrix_points``.

    As in ``physics.fit_to_matrix_mean``, everything scales with c_eq once ell and the
    supersaturation are fixed. Returns the field and its values at ``matrix_points``.
    """
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    per_unit_c_eq = np.linalg.solve(inverse_square_coupling(centres, radii), np.exp(ell / radii) - supersaturation)
    shape = supersaturation + inverse_square_sum(matrix_points, centres, radii, per_unit_c_eq)
    c_eq = matrix_mean / shape.mean()
    if not c_eq > 0:
        raise ValueError("no positive c_eq reproduces the requested matrix mean")
    field = InverseSquareField(centres, radii, float(c_eq), float(ell), float(supersaturation * c_eq),
                               c_eq * per_unit_c_eq)
    return field, c_eq * shape


@dataclass
class ShuffledSurfaceField:
    """The screened diffusion field with Gibbs-Thomson values assigned to precipitates by a permutation."""

    centres: np.ndarray
    radii: np.ndarray
    xi: float
    c_eq: float
    ell: float
    c_inf: float
    permutation: np.ndarray   # precipitate k takes the surface value of radius radii[permutation[k]]
    amplitudes: np.ndarray

    kind = "shuffled_surface"

    @property
    def surface_values(self):
        return gibbs_thomson(self.radii[self.permutation], self.c_eq, self.ell)

    def __call__(self, points):
        return self.c_inf + kernel_sum(points, self.centres, self.radii, self.xi, self.amplitudes)

    def to_dict(self):
        return {"kind": self.kind, "centres": self.centres.tolist(), "radii": self.radii.tolist(), "xi": self.xi,
                "c_eq": self.c_eq, "ell": self.ell, "c_inf": self.c_inf, "permutation": self.permutation.tolist(),
                "amplitudes": self.amplitudes.tolist()}


def shuffled_surface_to_matrix_mean(centres, radii, matrix_points, ell, supersaturation, matrix_mean, volume, seed, xi=None):
    """The shuffled-surface field with c_inf = supersaturation * c_eq and the given mean over ``matrix_points``.

    The screening length is ``xi``, or, if it is None, the sink-strength value for ``volume``.
    """
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    xi = screening_length(radii, volume) if xi is None else float(xi)
    permutation = np.random.default_rng(seed).permutation(len(radii))
    per_unit_c_eq = np.linalg.solve(coupling_matrix(centres, radii, xi), np.exp(ell / radii[permutation]) - supersaturation)
    shape = supersaturation + kernel_sum(matrix_points, centres, radii, xi, per_unit_c_eq)
    c_eq = matrix_mean / shape.mean()
    if not c_eq > 0:
        raise ValueError("no positive c_eq reproduces the requested matrix mean")
    field = ShuffledSurfaceField(centres, radii, float(xi), float(c_eq), float(ell), float(supersaturation * c_eq),
                                 permutation, c_eq * per_unit_c_eq)
    return field, c_eq * shape


@dataclass
class SmoothedNoiseField:
    """mean + sd * standardised (white noise smoothed by a Gaussian of width ``length``), trilinear between voxels."""

    seed: int
    length: float
    mean: float
    sd: float
    offset: float = 0.0    # standardisation of the smoothed noise over the matrix points it was fitted to
    scale: float = 1.0
    side: float = 60.0
    grid: float = 0.5

    kind = "smoothed_noise"

    def noise(self):
        n = int(round(self.side / self.grid))
        white = np.random.default_rng(self.seed).standard_normal((n, n, n))
        return ndimage.gaussian_filter(white, self.length / self.grid, mode="wrap", truncate=4.0)

    def standard(self, points):
        """The smoothed noise at ``points``, before standardisation."""
        index = _points(points) / self.grid - 0.5  # voxel centres sit at (i + 1/2) * grid
        return ndimage.map_coordinates(self.noise(), index.T, order=1, mode="nearest")

    def __call__(self, points):
        return self.mean + self.sd * (self.standard(points) - self.offset) / self.scale

    def to_dict(self):
        return {"kind": self.kind, "seed": self.seed, "length": self.length, "mean": self.mean, "sd": self.sd,
                "offset": self.offset, "scale": self.scale, "side": self.side, "grid": self.grid}


def smoothed_noise_to_matrix(seed, length, matrix_points, mean, sd, side=60.0, grid=0.5):
    """Smoothed noise whose mean and standard deviation over ``matrix_points`` are exactly ``mean`` and ``sd``."""
    field = SmoothedNoiseField(int(seed), float(length), float(mean), float(sd), side=float(side), grid=float(grid))
    raw = field.standard(matrix_points)
    field.offset, field.scale = float(raw.mean()), float(raw.std())
    return field, mean + sd * (raw - field.offset) / field.scale
