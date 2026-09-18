"""Fit the diffusion law and the precipitates together, from every observed atom.

Stage 5.2 detects precipitates first and then fits the law to the matrix atoms
(``experiments/reconstruction/stage5_physics_fit.py``). The capillary length is read from how
surface concentration varies with radius, so errors in the detected radii flatten it
(walkthrough E17). The alternative is to stop treating the geometry as known: make the radii
parameters of the same likelihood, and fit them to the atoms that carry their information,
which are the ones inside the precipitates.

The model is the one the Cramer-Rao bound of ``pilot/check_geometry_bound.py`` is computed
from. For the nearest precipitate k of an atom at distance r:

    p(x) = c(x) + [rho(r / R_k) - c(x)] * S((R_k - r) / w),
    rho(u) = 1 - (1 - rho_edge) u^m,

with c the screened diffusion field of ``physics.py`` for the current radii and constants, S
the logistic, and w the width over which the interface is mixed. The simulator's own interface is a step; w
stands for the mixing a real instrument adds. It is a fixed input by default, since a sharper
interface and a smaller precipitate explain the same rim; ``free_width`` fits it instead, which
is worth checking rather than assuming.

Free parameters: c_eq, ell, xi and c_inf in log space, every radius in log space, and the
interior shape. Centres are optional; they are held at their detected values by default.
Everything is float64 and differentiable, and the fit is L-BFGS on the summed Bernoulli log
loss of all observed atoms.

The interior shape is where the capillary length is lost. ``rho(u) = 1 - (1 - rho_edge) u^m``
cannot follow the simulator's profile exactly (it leaves about 0.03 in probability), and the
rim is also where the surface concentration ``c_eq exp(ell / R)`` is read, so the fit pays for
the misfit with ell: on development pattern 3 a free power law put ell 41 per cent high while
the same fit with the shape fixed at the oracle's own power law put it 8 per cent high, at a
*worse* loss. The remedy cannot be the oracle, so ``knots`` replaces the power law with a
monotone piecewise-linear profile with that many segments: rho(0) is free, and each knot is a
free fraction of the one before it, which is decreasing by construction and flexible enough
that the rim no longer has to borrow from ell.

``fit_joint`` returns the fitted constants, the radii, and the loss, so a joint fit can be
compared with detect-then-fit on the same atoms.

A fit cannot invent a precipitate its starting geometry lacks, and the atoms inside a missed
one then sit in the likelihood as guests the matrix cannot explain, which the constants absorb.
``grow_geometry`` adds the missing ones: it smooths what the current fit leaves over, adds a
precipitate at each significant peak, refits, and keeps the additions only while they earn
their parameters under a Bayesian information criterion.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOUNDS = {"c_eq": (1e-6, 1.0), "ell": (1e-3, 50.0), "xi": (0.5, 200.0), "c_inf": (1e-6, 1.0),
          "rho_edge": (1e-3, 0.999), "m": (0.5, 20.0), "width": (0.05, 3.0)}
RADIUS_BOUNDS = (0.5, 20.0)
CHUNK = 20000


class JointDiffusionModel(nn.Module):
    """The diffusion field, its constants, the precipitate radii and the interior profile, fitted together."""

    def __init__(self, centres, radii, c_eq, ell, xi, c_inf, width, rho_edge=0.3, m=3.0, free_centres=False,
                 free_width=False, knots=None, levels=None):
        super().__init__()
        self.fixed_width = None if free_width else float(width)
        self.knots = None if knots is None else int(knots)
        self.register_buffer("base_centres", torch.as_tensor(np.asarray(centres), dtype=torch.float64))
        self.log_radii = nn.Parameter(torch.log(torch.as_tensor(np.asarray(radii), dtype=torch.float64)))
        self.offsets = nn.Parameter(torch.zeros_like(self.base_centres)) if free_centres else None
        for name, value in (("c_eq", c_eq), ("ell", ell), ("xi", xi), ("c_inf", c_inf),
                            ("rho_edge", rho_edge), ("m", m)):
            setattr(self, f"log_{name}", nn.Parameter(torch.tensor(math.log(value), dtype=torch.float64)))
        if free_width:
            self.log_width = nn.Parameter(torch.tensor(math.log(width), dtype=torch.float64))
        if self.knots is not None:
            start = self.power_law_levels(rho_edge, m) if levels is None else torch.as_tensor(levels,
                                                                                              dtype=torch.float64)
            top = torch.clamp(start[0], 1e-4, 1 - 1e-4)
            ratios = torch.clamp(start[1:] / torch.clamp(start[:-1], min=1e-6), 1e-4, 1 - 1e-4)
            self.logit_top = nn.Parameter(torch.log(top / (1 - top)))
            self.logit_ratios = nn.Parameter(torch.log(ratios / (1 - ratios)))

    def power_law_levels(self, rho_edge, m):
        """The two-parameter profile sampled at the knots, which is where a flexible fit starts."""
        u = torch.linspace(0, 1, self.knots + 1, dtype=torch.float64)
        return 1 - (1 - rho_edge) * u ** m

    def levels(self):
        """The interior profile at the knots: decreasing, in (0, 1), by construction."""
        top = torch.sigmoid(self.logit_top)
        return top * torch.cat([torch.ones(1, dtype=torch.float64),
                                torch.cumprod(torch.sigmoid(self.logit_ratios), dim=0)])

    def interior(self, u):
        """Guest probability inside a precipitate at fractional radius ``u``."""
        if self.knots is None:
            return 1 - (1 - self.value("rho_edge")) * u ** self.value("m")
        levels = self.levels()
        position = torch.clamp(u, 0.0, 1.0) * self.knots
        lower = torch.clamp(position.detach().floor().long(), max=self.knots - 1)
        fraction = position - lower
        return levels[lower] * (1 - fraction) + levels[lower + 1] * fraction

    def width(self):
        """The interface width: a fixed input, or a fitted parameter with ``free_width``."""
        if self.fixed_width is not None:
            return torch.tensor(self.fixed_width, dtype=torch.float64)
        return self.value("width")

    def value(self, name):
        low, high = BOUNDS[name]
        return torch.exp(torch.clamp(getattr(self, f"log_{name}"), math.log(low), math.log(high)))

    def radii(self):
        return torch.exp(torch.clamp(self.log_radii, math.log(RADIUS_BOUNDS[0]), math.log(RADIUS_BOUNDS[1])))

    def centres(self):
        return self.base_centres if self.offsets is None else self.base_centres + self.offsets

    def constants(self):
        return {name: float(self.value(name).detach()) for name in ("c_eq", "ell", "xi", "c_inf")}

    def shape(self):
        if self.knots is not None:
            return {"knots": self.knots, "levels": [float(v) for v in self.levels().detach()],
                    "width": float(self.width().detach() if self.fixed_width is None else self.fixed_width)}
        out = {name: float(self.value(name).detach()) for name in ("rho_edge", "m")}
        out["width"] = float(self.width().detach() if self.fixed_width is None else self.fixed_width)
        return out

    def amplitudes(self):
        """Source amplitudes whose surface means are the Gibbs-Thomson values, for the current radii."""
        radii, xi, c_inf = self.radii(), self.value("xi"), self.value("c_inf")
        centres = self.centres()
        eye = torch.eye(len(radii), dtype=torch.float64, device=radii.device)
        distance = torch.cdist(centres, centres) + eye
        coupling = (radii[None, :] / distance) * torch.exp(-(distance - radii[None, :]) / xi)
        coupling = coupling * (torch.sinh(radii / xi) / (radii / xi))[:, None]
        coupling = coupling * (1 - eye) + eye
        surface = self.value("c_eq") * torch.exp(self.value("ell") / radii)
        return torch.linalg.solve(coupling, surface - c_inf)

    def matrix_field(self, x):
        """The screened field alone, without the interior blend.

        This is what the law says about the matrix, and it is what a matrix score should use:
        blending in the interior would punish the model at every true-matrix atom its own
        interfaces happen to cover.
        """
        radii, centres = self.radii(), self.centres()
        distance = torch.cdist(x, centres)
        return self.value("c_inf") + ((radii[None, :] / torch.clamp(distance, min=1e-6))
                                      * torch.exp(-(distance - radii[None, :]) / self.value("xi"))) @ self.amplitudes()

    def forward(self, x, assignment=None):
        """Guest probability at ``x``; ``assignment`` fixes each atom's nearest precipitate."""
        radii, centres = self.radii(), self.centres()
        distance = torch.cdist(x, centres)
        matrix = self.matrix_field(x)
        if assignment is None:
            assignment = torch.argmin(distance - radii[None, :], dim=1)
        nearest = distance.gather(1, assignment[:, None]).squeeze(1)
        radius = radii[assignment]
        interior = self.interior(torch.clamp(nearest / radius, max=1.0))
        blend = torch.sigmoid((radius - nearest) / self.width())
        return torch.clamp(matrix + (interior - matrix) * blend, 1e-9, 1 - 1e-9)

    def assign(self, x):
        """Each atom's nearest precipitate, by surface distance, without gradients."""
        with torch.no_grad():
            out = []
            radii, centres = self.radii(), self.centres()
            for start in range(0, len(x), CHUNK):
                gap = torch.cdist(x[start:start + CHUNK], centres) - radii[None, :]
                out.append(torch.argmin(gap, dim=1))
            return torch.cat(out)


def fit_joint(model, x, y, iterations=150, rounds=3):
    """Maximum likelihood by L-BFGS over the constants, the radii and the interior shape.

    Each atom's nearest precipitate is held fixed within a round and recomputed between rounds,
    so the assignment cannot oscillate inside the line search.
    """
    tx = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    ty = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    started, losses = time.perf_counter(), []
    for _ in range(rounds):
        assignment = model.assign(tx)
        optimiser = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=iterations, line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            loss = F.binary_cross_entropy(model(tx, assignment), ty)
            loss.backward()
            return loss

        optimiser.step(closure)
        with torch.no_grad():
            losses.append(float(F.binary_cross_entropy(model(tx, assignment), ty)))
    with torch.no_grad():
        final = float(F.binary_cross_entropy(model(tx, model.assign(tx)), ty))
    return {"train_loss": final, "round_losses": losses, "constants": model.constants(), "shape": model.shape(),
            "radii": model.radii().detach().numpy().copy(), "seconds": round(time.perf_counter() - started, 1)}


def residual_peaks(model, x, y, bandwidth=1.5, grid=0.5, lower=0.0, upper=60.0, threshold=5.0, minimum_atoms=20):
    """Places where the fit leaves more guests than it explains: candidate missed precipitates.

    The residual y - p is smoothed over the same grid the detection uses. Its noise at a voxel
    is sqrt(sum_j K_j^2 p_j (1 - p_j)) / sum_j K_j, so the peaks are measured in standard
    deviations. Returns the peak positions and a radius from the volume of each blob.
    """
    from scipy import ndimage

    from geom_mesh_net.fields import cluster_extraction as ce

    x = np.asarray(x, dtype=float)
    p = predict(model, x)
    residual = np.asarray(y, dtype=float) - p
    edges = np.linspace(lower, upper, int(round((upper - lower) / grid)) + 1)
    sigma = bandwidth / grid
    weights = ce.kernel_weights(bandwidth, grid)
    counts = np.histogramdd(x, bins=[edges] * 3)[0]
    smoothed = ndimage.gaussian_filter(np.histogramdd(x, bins=[edges] * 3, weights=residual)[0], sigma,
                                       mode="constant", truncate=4.0)
    atoms = ndimage.gaussian_filter(counts, sigma, mode="constant", truncate=4.0)
    variance = np.histogramdd(x, bins=[edges] * 3, weights=p * (1 - p))[0]
    for axis in range(3):
        variance = ndimage.correlate1d(variance, weights ** 2, axis=axis, mode="constant")
    valid = atoms > 0.2 * np.median(atoms)
    noise = np.sqrt(np.maximum(variance, 1e-12)) / np.maximum(atoms, 1e-12)
    z = np.where(valid, smoothed / np.maximum(atoms, 1e-12) / np.maximum(noise, 1e-12), 0.0)

    foreground = z > threshold
    if not foreground.any():
        return np.zeros((0, 3)), np.zeros(0)
    labels, n = ndimage.label(foreground)
    keep = [k for k in range(1, n + 1) if (labels == k).sum() * grid ** 3 >= minimum_atoms * grid ** 3]
    if not keep:
        return np.zeros((0, 3)), np.zeros(0)
    centres = np.array(ndimage.center_of_mass(np.ones(z.shape), labels, keep)) * grid + lower + grid / 2
    volumes = np.array([float((labels == k).sum()) for k in keep]) * grid ** 3
    return centres, (3 * volumes / (4 * np.pi)) ** (1 / 3)


def grow_geometry(model, x, y, rounds=2, penalty=None, **peak_options):
    """Add precipitates where the fit leaves unexplained guests, keeping those that earn their parameters.

    Each round refits with the candidates added. A round is kept only if the summed log loss
    falls by more than ``penalty`` per added parameter, which defaults to the Bayesian
    information criterion's (log n) / 2. Returns the model and a record of what was added.
    """
    import copy as _copy

    n = len(x)
    penalty = 0.5 * math.log(n) if penalty is None else penalty
    record = {"rounds": []}
    best = fit_joint(model, x, y)
    for _ in range(rounds):
        centres, radii = residual_peaks(model, x, y, **peak_options)
        if not len(radii):
            break
        grown = with_geometry(model, np.vstack([model.centres().detach().numpy(), centres]),
                              np.concatenate([model.radii().detach().numpy(), np.maximum(radii, 1.0)]))
        candidate = fit_joint(grown, x, y)
        gain = (best["train_loss"] - candidate["train_loss"]) * n
        kept = gain > penalty * len(radii)
        record["rounds"].append({"added": int(len(radii)), "gain_nats": float(gain),
                                 "penalty_nats": float(penalty * len(radii)), "kept": bool(kept)})
        if not kept:
            break
        model, best = grown, candidate
    record.update(best)
    record["precipitates"] = int(len(model.radii()))
    return model, record


def with_geometry(model, centres, radii):
    """A copy of ``model`` on a different geometry, carrying its constants and interior profile."""
    shape = model.shape()
    return JointDiffusionModel(centres, radii, **model.constants(),
                               width=shape["width"], free_centres=model.offsets is not None,
                               free_width=model.fixed_width is None, knots=model.knots,
                               levels=shape.get("levels"),
                               **{k: v for k, v in shape.items() if k in ("rho_edge", "m")})


def predict(model, points, chunk=CHUNK, matrix_only=False):
    """The fitted model's probability at ``points`` as a float64 numpy array.

    ``matrix_only`` returns the screened field without the interior blend, for scoring matrix
    atoms against methods that model the matrix alone.
    """
    out = []
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(np.asarray(points[start:start + chunk]), dtype=torch.float64)
            out.append((model.matrix_field(x) if matrix_only else model(x)).numpy())
    return np.clip(np.concatenate(out), 1e-9, 1 - 1e-9) if out else np.zeros(0)
