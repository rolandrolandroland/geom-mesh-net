"""A screened diffusion field around spherical precipitates (reconstruction Stage 5).

``clustersim``'s matrix is uniform, so no differential equation constrains it. Stage 5
replaces the matrix guest probability with a quasi-stationary diffusion field. Outside
every precipitate the field obeys the screened diffusion equation

    (lap - xi^-2) (c - c_inf) = 0,

and its mean over each precipitate's surface equals the Gibbs-Thomson value

    c_k = c_eq * exp(ell / R_k),

where ell is the capillary length. Small precipitates hold more solute at their surface
than large ones. Where c_k is below c_inf, a precipitate sits in a depletion zone;
where it is above, in an enriched one. The critical radius ell / ln(c_inf / c_eq)
separates the two.

The field superposes one screened source per precipitate,

    c(x) = c_inf + sum_k a_k g_k(x),    g_k(x) = (R_k / r_k) exp(-(r_k - R_k) / xi),

with r_k = |x - x_k|. Each g_k solves the equation away from its own centre, so the sum
solves it exactly throughout the matrix. The amplitudes a_k make every surface mean
exact. For any solution of the equation inside a ball of radius R, the mean over the
ball's surface equals the value at its centre times sinh(R / xi) / (R / xi). The surface
conditions therefore become the linear system

    a_k + sum_{j != k} a_j g_j(x_k) sinh(R_k / xi) / (R_k / xi) = c_k - c_inf.

That identity needs every other centre outside the ball, so precipitates must not
overlap. Point by point, the surface value departs from its mean by the higher
multipoles this solution leaves out.

Adding up single-precipitate profiles, a_k = c_k - c_inf, does not solve this boundary
value problem: neighbours' tails shift every surface value, and in dense patterns the
field leaves [0, 1]. See the Stage 5 correction of 2026-09-16 in
``experiments/reconstruction/ROADMAP.md``.

The screening length is set from the precipitates' sink strength,
xi = (4 pi n Rbar)^(-1/2), the mean-field result for number density n and mean radius
Rbar. Because every neighbour is also explicit here, xi acts as a screening constant of
the model rather than a property derived from the geometry.
"""

from dataclasses import dataclass

import numpy as np

CHUNK = 8000


def _points(points):
    if isinstance(points, dict):
        return np.column_stack([np.asarray(points[a], dtype=float) for a in ("x", "y", "z")])
    return np.asarray(points, dtype=float).reshape(-1, 3)


def screening_length(radii, volume):
    """Mean-field screening length (4 pi n Rbar)^(-1/2) = (4 pi sum_k R_k / V)^(-1/2)."""
    total = float(np.sum(radii))
    if total <= 0:
        raise ValueError("the screening length needs at least one precipitate of positive radius")
    return (4 * np.pi * total / volume) ** -0.5


def gibbs_thomson(radii, c_eq, ell):
    """Equilibrium surface concentration c_eq exp(ell / R) of precipitates of radius R."""
    return c_eq * np.exp(ell / np.asarray(radii, dtype=float))


def source_kernel(distance, radius, xi):
    """(R / r) exp(-(r - R) / xi): a precipitate's screened source, equal to 1 on its surface."""
    return (radius / distance) * np.exp(-(distance - radius) / xi)


def check_separated(centres, radii):
    """Raise ValueError if any two precipitates overlap."""
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distance, np.inf)
    overlap = distance < radii[:, None] + radii[None, :]
    if overlap.any():
        i, j = np.argwhere(overlap)[0]
        raise ValueError(f"precipitates {i} and {j} overlap; the surface conditions need separated spheres")


def coupling_matrix(centres, radii, xi):
    """M with M[k, k] = 1 and M[k, j] = mean over precipitate k's surface of g_j."""
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    check_separated(centres, radii)
    distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distance, 1.0)
    x = radii / xi
    matrix = source_kernel(distance, radii[None, :], xi) * (np.sinh(x) / x)[:, None]
    np.fill_diagonal(matrix, 1.0)
    return matrix


def kernel_sum(points, centres, radii, xi, amplitudes, chunk=CHUNK):
    """sum_k amplitudes[k] g_k(points). ``amplitudes`` is (K,) or (K, P); returns (N,) or (N, P)."""
    points, centres = _points(points), _points(centres)
    radii, amplitudes = np.asarray(radii, dtype=float), np.asarray(amplitudes, dtype=float)
    columns = amplitudes.reshape(len(radii), -1)
    out = np.empty((len(points), columns.shape[1]))
    squared_centres = (centres ** 2).sum(axis=1)
    for start in range(0, len(points), chunk):
        x = points[start:start + chunk]
        squared = (x ** 2).sum(axis=1)[:, None] + squared_centres[None, :] - 2 * x @ centres.T
        distance = np.sqrt(np.maximum(squared, 1e-24))
        out[start:start + chunk] = source_kernel(distance, radii[None, :], xi) @ columns
    return out[:, 0] if amplitudes.ndim == 1 else out


@dataclass
class DiffusionField:
    """The matrix concentration c(x) around separated spherical precipitates."""

    centres: np.ndarray      # (K, 3)
    radii: np.ndarray        # (K,)
    xi: float                # screening length
    c_eq: float              # equilibrium concentration at a flat interface
    ell: float               # capillary length
    c_inf: float             # far-field concentration
    amplitudes: np.ndarray   # (K,) source amplitudes a_k

    @property
    def surface_values(self):
        return gibbs_thomson(self.radii, self.c_eq, self.ell)

    @property
    def supersaturation(self):
        return self.c_inf / self.c_eq

    @property
    def critical_radius(self):
        """Radius at which c_k = c_inf; larger precipitates are depleting. Infinite without supersaturation."""
        return self.ell / np.log(self.supersaturation) if self.supersaturation > 1 else np.inf

    def __call__(self, points):
        return self.c_inf + kernel_sum(points, self.centres, self.radii, self.xi, self.amplitudes)

    def torch_values(self, points):
        """c(x) as a differentiable float64 torch expression of an (N, 3) tensor."""
        import torch

        centres = torch.as_tensor(self.centres, dtype=torch.float64)
        radii = torch.as_tensor(self.radii, dtype=torch.float64)
        amplitudes = torch.as_tensor(self.amplitudes, dtype=torch.float64)
        distance = torch.sqrt(((points[:, None, :] - centres[None, :, :]) ** 2).sum(dim=-1))
        kernel = (radii[None, :] / distance) * torch.exp(-(distance - radii[None, :]) / self.xi)
        return self.c_inf + kernel @ amplitudes

    def pde_residual(self, points):
        """Largest |lap c - xi^-2 (c - c_inf)| over ``points``, relative to xi^-2 max |c - c_inf|.

        The Laplacian is taken by automatic differentiation in float64, independently of
        how the amplitudes were solved.
        """
        import torch

        x = torch.as_tensor(_points(points), dtype=torch.float64).requires_grad_(True)
        c = self.torch_values(x)
        gradient = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
        laplacian = sum(torch.autograd.grad(gradient[:, d].sum(), x, retain_graph=True)[0][:, d] for d in range(3))
        deviation = c - self.c_inf
        residual = laplacian - deviation / self.xi ** 2
        return float((residual.abs().max() / (deviation.abs().max() / self.xi ** 2)).detach())

    def to_dict(self):
        return {"centres": self.centres.tolist(), "radii": self.radii.tolist(), "xi": self.xi,
                "c_eq": self.c_eq, "ell": self.ell, "c_inf": self.c_inf, "amplitudes": self.amplitudes.tolist()}

    @classmethod
    def from_dict(cls, values):
        return cls(np.asarray(values["centres"], dtype=float), np.asarray(values["radii"], dtype=float),
                   float(values["xi"]), float(values["c_eq"]), float(values["ell"]), float(values["c_inf"]),
                   np.asarray(values["amplitudes"], dtype=float))


def surface_means(field, nodes=200):
    """Mean of c over each precipitate's surface, by direct integration.

    Precipitate j's source depends only on the angle between a surface point of k and
    the direction from k to j, so each term is a one-dimensional Gauss-Legendre integral
    in the cosine of that angle. The mean-value identity the solve relies on is not
    used, so this verifies the solve rather than restating it.
    """
    mu, weights = np.polynomial.legendre.leggauss(nodes)
    centres, radii = field.centres, field.radii
    distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
    means = field.c_inf + field.amplitudes.astype(float)  # each source is exactly 1 on its own surface
    for k in range(len(radii)):
        others = np.flatnonzero(np.arange(len(radii)) != k)
        d = distance[k, others][:, None]
        r = np.sqrt(radii[k] ** 2 + d ** 2 - 2 * radii[k] * d * mu[None, :])
        averaged = 0.5 * (source_kernel(r, radii[others][:, None], field.xi) * weights[None, :]).sum(axis=1)
        means[k] += averaged @ field.amplitudes[others]
    return means


def solve_field(centres, radii, xi, c_eq, ell, c_inf):
    """The field whose surface means equal the Gibbs-Thomson values, for given constants."""
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    amplitudes = np.linalg.solve(coupling_matrix(centres, radii, xi), gibbs_thomson(radii, c_eq, ell) - c_inf)
    return DiffusionField(centres, radii, float(xi), float(c_eq), float(ell), float(c_inf), amplitudes)


def fit_to_matrix_mean(centres, radii, matrix_points, ell, supersaturation, matrix_mean, volume, xi=None):
    """The field with c_inf = supersaturation * c_eq whose mean over ``matrix_points`` is ``matrix_mean``.

    Every amplitude and the far-field value are proportional to c_eq once ell and the
    supersaturation are fixed, so c_eq follows from one division. Returns the field and
    its values at ``matrix_points``.
    """
    centres, radii = _points(centres), np.asarray(radii, dtype=float)
    xi = screening_length(radii, volume) if xi is None else float(xi)
    per_unit_c_eq = np.linalg.solve(coupling_matrix(centres, radii, xi), np.exp(ell / radii) - supersaturation)
    shape = supersaturation + kernel_sum(matrix_points, centres, radii, xi, per_unit_c_eq)
    c_eq = matrix_mean / shape.mean()
    if not c_eq > 0:
        raise ValueError("no positive c_eq reproduces the requested matrix mean")
    field = DiffusionField(centres, radii, float(xi), float(c_eq), float(ell), float(supersaturation * c_eq),
                           c_eq * per_unit_c_eq)
    return field, c_eq * shape
