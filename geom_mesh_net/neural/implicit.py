"""Implicit neural fields for the reconstruction track.

Stage 5.2 (``experiments/reconstruction/ROADMAP.md``) as first designed fits a
physics-informed field to one pattern. The loss has three terms:

- binary cross-entropy on the labels of observed matrix atoms;
- the screened diffusion equation's residual at collocation points in the matrix,
  ξ_ref² (∇²c − ξ⁻²(c − c∞)) / s, computed by automatic differentiation;
- the Gibbs–Thomson condition on each precipitate's surface mean,
  (mean of c over surface k − c_eq·exp(ℓ/R_k)) / s.

s is a concentration scale and ξ_ref a fixed length, the initial screening length, which
make both penalties dimensionless. c_eq, ℓ, ξ and c∞ are trainable, so the fit recovers
the physical constants along with the field: the standard inverse problem for a
physics-informed network. With both penalty weights at zero, the same network is the
unconstrained control.

The reference must not be the trainable ξ. Written as ξ²∇²c − (c − c∞)
(``residual="scaled"``, the form first tried), the network's own approximation error
enters scaled by ξ², and shrinking ξ always lowers the penalty.

This network does not work for Stage 5. On development patterns it fails to recover the
constants even when given the true geometry and matrix domain, because the loss itself
prefers constants under which a flat field satisfies both penalties exactly
(``experiments/reconstruction/pilot/check_soft_pinn.py``; walkthrough E16). It is kept
so that result can be reproduced.

The network is a SIREN, a sine-activated multilayer perceptron. A ReLU network's Laplacian
is zero almost everywhere, so it cannot carry the residual.

``ParametricDiffusionField`` is the analytic family of ``geom_mesh_net.fields.physics`` with
trainable constants and a fixed geometry: the physics as a hard constraint, which is what
Stage 5.2 tests after its correction.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Siren(nn.Module):
    """A sine-activated MLP (Sitzmann et al., 2020) with the initialisation that keeps it trainable."""

    def __init__(self, in_features=3, width=128, depth=3, w0=10.0, out_scale=0.1):
        super().__init__()
        self.w0 = w0
        dims = [in_features] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(width, 1)
        with torch.no_grad():
            for i, layer in enumerate(self.hidden):
                bound = 1 / layer.in_features if i == 0 else math.sqrt(6 / layer.in_features) / w0
                layer.weight.uniform_(-bound, bound)
            bound = out_scale * math.sqrt(6 / width) / w0
            self.out.weight.uniform_(-bound, bound)
            self.out.bias.zero_()

    def forward(self, x):
        for layer in self.hidden:
            x = torch.sin(self.w0 * layer(x))
        return self.out(x).squeeze(-1)


class DiffusionPINN(nn.Module):
    """c(x) = sigmoid(logit(c0) + f(x)) for a SIREN f, with log-parametrised physical constants.

    Positions are in box coordinates; the network sees them rescaled to [-1, 1].
    """

    RESIDUALS = ("reference", "scaled")

    def __init__(self, c0, c_eq, ell, xi, c_inf, box=60.0, xi_ref=None, residual="reference", **siren):
        super().__init__()
        if residual not in self.RESIDUALS:
            raise ValueError(f"residual must be one of {self.RESIDUALS}")
        self.box = box
        self.residual = residual
        self.xi_ref = float(xi if xi_ref is None else xi_ref)
        self.net = Siren(**siren)
        self.register_buffer("offset", torch.tensor(math.log(c0 / (1 - c0))))
        self.log_c_eq = nn.Parameter(torch.tensor(math.log(c_eq)))
        self.log_ell = nn.Parameter(torch.tensor(math.log(ell)))
        self.log_xi = nn.Parameter(torch.tensor(math.log(xi)))
        self.log_c_inf = nn.Parameter(torch.tensor(math.log(c_inf)))

    def forward(self, x):
        return torch.sigmoid(self.offset + self.net(2 * x / self.box - 1))

    def constants(self):
        return {name: float(torch.exp(getattr(self, f"log_{name}")).detach().cpu())
                for name in ("c_eq", "ell", "xi", "c_inf")}

    def physics_parameters(self):
        return [self.log_c_eq, self.log_ell, self.log_xi, self.log_c_inf]

    def network_parameters(self):
        return list(self.net.parameters())

    def pde_residual(self, x, scale):
        """The equation's residual over ``scale`` at the points x, by automatic differentiation.

        ``residual="reference"``: ξ_ref² (∇²c − ξ⁻²(c − c∞)); ``"scaled"``: ξ²∇²c − (c − c∞).
        """
        x = x.detach().requires_grad_(True)
        c = self(x)
        gradient = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
        laplacian = sum(torch.autograd.grad(gradient[:, d].sum(), x, create_graph=True)[0][:, d] for d in range(3))
        xi, c_inf = torch.exp(self.log_xi), torch.exp(self.log_c_inf)
        if self.residual == "scaled":
            return (xi ** 2 * laplacian - (c - c_inf)) / scale
        return self.xi_ref ** 2 * (laplacian - (c - c_inf) / xi ** 2) / scale

    def boundary_residual(self, centres, radii, directions, scale):
        """(mean of c over each surface − c_eq·exp(ℓ/R)) / scale, one value per precipitate."""
        points = centres[:, None, :] + radii[:, None, None] * directions[None, :, :]
        surface_mean = self(points.reshape(-1, 3)).reshape(len(radii), -1).mean(dim=1)
        target = torch.exp(self.log_c_eq + torch.exp(self.log_ell) / radii)
        return (surface_mean - target) / scale


def fibonacci_directions(n):
    i = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)])


def fit_pinn(model, train_x, train_y, valid_x, valid_y, collocation=None, boundary=None, lambda_pde=0.0,
             lambda_bc=0.0, scale=0.1, steps=2000, lr=1e-4, lr_constants=1e-2, n_collocation=4096,
             eval_every=50, seed=0, device="cpu", freeze_constants=0, patience=None, evaluate_initial=False):
    """Fit ``model`` by Adam with early stopping on the held-out observed atoms.

    ``collocation`` is an (N, 3) array of matrix points, sampled ``n_collocation`` at a
    time; ``boundary`` is (centres, radii, directions). The constants stay fixed for the
    first ``freeze_constants`` steps. Training stops once ``patience`` steps pass without a
    better held-out loss. With ``evaluate_initial``, the untrained model, a near-constant
    field, is a candidate too, so early stopping can never end worse on the held-out atoms
    than not training. Returns the best model state and a record of the fit.
    """
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    as_tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    model = model.to(device)
    tx, ty, vx, vy = as_tensor(train_x), as_tensor(train_y), as_tensor(valid_x), as_tensor(valid_y)
    physics = lambda_pde > 0 or lambda_bc > 0
    groups = [{"params": model.network_parameters(), "lr": lr}]
    if physics:
        groups.append({"params": model.physics_parameters(), "lr": lr_constants})
    optimiser = torch.optim.Adam(groups)
    pool = as_tensor(collocation) if collocation is not None and lambda_pde > 0 else None
    if boundary is not None and lambda_bc > 0 and len(boundary[1]):
        centres, radii, directions = (as_tensor(b) for b in boundary)
    else:
        centres = None

    best = {"valid_loss": math.inf, "step": 0, "state": copy.deepcopy(model.state_dict())}
    history, started = [], time.perf_counter()
    if evaluate_initial:
        with torch.no_grad():
            best["valid_loss"] = float(F.binary_cross_entropy(model(vx).clamp(1e-6, 1 - 1e-6), vy))
    for step in range(1, steps + 1):
        optimiser.zero_grad()
        bce = F.binary_cross_entropy(model(tx).clamp(1e-6, 1 - 1e-6), ty)
        loss = bce
        pde = bc = torch.zeros((), device=device)
        if pool is not None:
            pick = torch.as_tensor(generator.integers(0, len(pool), n_collocation), device=device)
            pde = (model.pde_residual(pool[pick], scale) ** 2).mean()
            loss = loss + lambda_pde * pde
        if centres is not None:
            bc = (model.boundary_residual(centres, radii, directions, scale) ** 2).mean()
            loss = loss + lambda_bc * bc
        loss.backward()
        if step <= freeze_constants:
            for parameter in model.physics_parameters():
                parameter.grad = None
        optimiser.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                valid = float(F.binary_cross_entropy(model(vx).clamp(1e-6, 1 - 1e-6), vy))
            history.append({"step": step, "train_bce": float(bce.detach()), "pde": float(pde.detach()), "bc": float(bc.detach()),
                            "valid_bce": valid, **model.constants()})
            if valid < best["valid_loss"]:
                best = {"valid_loss": valid, "step": step, "state": copy.deepcopy(model.state_dict())}
            elif patience is not None and step - best["step"] >= patience:
                break
    model.load_state_dict(best["state"])
    return {"best_step": best["step"], "valid_loss": best["valid_loss"], "constants": model.constants(),
            "history": history, "seconds": round(time.perf_counter() - started, 1)}


def predict(model, points, device="cpu", chunk=65536):
    """The model's field at ``points`` as a float64 numpy array."""
    out = []
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(np.asarray(points[start:start + chunk]), dtype=torch.float32, device=device)
            out.append(model(x).detach().cpu().double().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class ParametricDiffusionField(nn.Module):
    """The analytic diffusion field for a fixed geometry, with trainable constants (float64).

    Mirrors ``geom_mesh_net.fields.physics``: amplitudes solve the surface-mean conditions
    through the mean-value coupling matrix, so gradients reach every constant. Two guards
    keep a line search finite: the constants are clamped to wide physical ranges, and a
    point inside a sphere takes the kernel's surface value, since a segmented geometry can
    put an observed atom there.
    """

    BOUNDS = {"c_eq": (1e-6, 1.0), "ell": (1e-3, 50.0), "xi": (0.5, 200.0), "c_inf": (1e-6, 1.0)}

    def __init__(self, centres, radii, c_eq, ell, xi, c_inf):
        super().__init__()
        self.register_buffer("centres", torch.as_tensor(np.asarray(centres), dtype=torch.float64))
        self.register_buffer("radii", torch.as_tensor(np.asarray(radii), dtype=torch.float64))
        self.log_c_eq = nn.Parameter(torch.tensor(math.log(c_eq), dtype=torch.float64))
        self.log_ell = nn.Parameter(torch.tensor(math.log(ell), dtype=torch.float64))
        self.log_xi = nn.Parameter(torch.tensor(math.log(xi), dtype=torch.float64))
        self.log_c_inf = nn.Parameter(torch.tensor(math.log(c_inf), dtype=torch.float64))

    def value(self, name):
        low, high = self.BOUNDS[name]
        return torch.exp(torch.clamp(getattr(self, f"log_{name}"), math.log(low), math.log(high)))

    def constants(self):
        return {name: float(self.value(name).detach()) for name in self.BOUNDS}

    def amplitudes(self):
        xi, c_inf = self.value("xi"), self.value("c_inf")
        eye = torch.eye(len(self.radii), dtype=torch.float64, device=self.radii.device)
        d = torch.cdist(self.centres, self.centres) + eye
        coupling = (self.radii[None, :] / d) * torch.exp(-(d - self.radii[None, :]) / xi)
        coupling = coupling * (torch.sinh(self.radii / xi) / (self.radii / xi))[:, None]
        coupling = coupling * (1 - eye) + eye
        surface = self.value("c_eq") * torch.exp(self.value("ell") / self.radii)
        return torch.linalg.solve(coupling, surface - c_inf)

    def forward(self, x):
        d = torch.maximum(torch.cdist(x, self.centres), self.radii[None, :])
        kernel = (self.radii[None, :] / d) * torch.exp(-(d - self.radii[None, :]) / self.value("xi"))
        return self.value("c_inf") + kernel @ self.amplitudes()


def fit_parametric(model, x, y, iterations=200):
    """Maximum likelihood by L-BFGS on the observed labels ``y`` at positions ``x``."""
    tx = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    ty = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    optimiser = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=iterations, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = F.binary_cross_entropy(model(tx).clamp(1e-9, 1 - 1e-9), ty)
        loss.backward()
        return loss

    started = time.perf_counter()
    optimiser.step(closure)
    with torch.no_grad():
        loss = float(F.binary_cross_entropy(model(tx).clamp(1e-9, 1 - 1e-9), ty))
    return {"train_loss": loss, "constants": model.constants(), "seconds": round(time.perf_counter() - started, 1)}
