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


# ---------------------------------------------------------------------------------------------
# Stage 2: a field fitted to one pattern, with Fourier features and its two controls
# ---------------------------------------------------------------------------------------------

BOX_LENGTH = 60.0     # the benchmark box, in the simulation's length unit
RATE_CLIP = 1e-6      # a fitting guest fraction of exactly 0 or 1 is clipped to this before its log odds


def base_frequencies(n_features, seed):
    """B0: ``n_features`` frequency vectors with independent standard normal components.

    Every σ candidate of one seed scales the same B0, so candidates differ only in σ and not
    in an unrelated random draw.
    """
    return np.random.default_rng(seed).standard_normal((int(n_features), 3)).astype(np.float32)


class CoordinateField(nn.Module):
    """A guest-probability field over the box: an encoding of position, then a network to one logit.

    ``hidden`` lists the hidden widths; an empty tuple makes the model linear in its encoding.
    ``logits`` is the path for ``BCEWithLogitsLoss``; calling the model returns probabilities, so
    the module works with ``predict``. Coordinates are divided by the known box length, never by
    bounds estimated from a sample.
    """

    def __init__(self, in_features, hidden, box):
        super().__init__()
        self.register_buffer("box", torch.tensor(float(box)))
        layers, width = [], int(in_features)
        for size in hidden:
            layers += [nn.Linear(width, int(size)), nn.ReLU()]
            width = int(size)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.hidden = tuple(int(h) for h in hidden)

    def encode(self, x):
        raise NotImplementedError

    def logits(self, x):
        return self.head(self.body(self.encode(x))).squeeze(-1)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

    def initialise_constant(self, rate):
        """Start exactly at the constant field ``rate``: the output weights are zeroed, so no
        hidden unit contributes until training moves them."""
        rate = min(max(float(rate), RATE_CLIP), 1 - RATE_CLIP)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(math.log(rate / (1 - rate)))
        return self

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))


class FourierFeatureField(CoordinateField):
    """γ(u) = [sin 2πBu, cos 2πBu] with u = x / L and B = σ·B0, then an MLP (Tancik et al., 2020).

    ``hidden=()`` gives the linear Fourier control: a logistic regression on the same features.
    B0 and σ are buffers, so they are saved and restored with every checkpoint.
    """

    def __init__(self, base, sigma, hidden=(256, 256, 256, 256), box=BOX_LENGTH):
        base = torch.as_tensor(np.asarray(base), dtype=torch.float32)
        super().__init__(2 * base.shape[0], hidden, box)
        self.register_buffer("base_frequencies", base)
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def frequencies(self):
        return self.sigma * self.base_frequencies

    def encode(self, x):
        z = 2 * math.pi * (x / self.box) @ self.frequencies().T
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


class RawCoordinateField(CoordinateField):
    """The control without an encoding: an MLP on 2x/L − 1, which lies in [−1, 1]³."""

    def __init__(self, hidden=(256, 256, 256, 256), box=BOX_LENGTH):
        super().__init__(3, hidden, box)

    def encode(self, x):
        return 2 * x / self.box - 1


def build_field(kind, init_seed, base=None, sigma=None, hidden=(256, 256, 256, 256), box=BOX_LENGTH):
    """Construct a Stage 2 model with its weights drawn from ``init_seed`` alone.

    ``kind`` is ``"fourier"`` (the MLP, or the linear control with ``hidden=()``) or ``"raw"``.
    The global random state is left untouched.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        if kind == "fourier":
            return FourierFeatureField(base, sigma, hidden=hidden, box=box)
        if kind == "raw":
            return RawCoordinateField(hidden=hidden, box=box)
    raise ValueError(f"unknown field kind {kind!r}")


def _bce(model, x, y, chunk):
    """Mean binary cross-entropy of ``model`` on (x, y), evaluated in chunks without gradients."""
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            logits = model.logits(x[start:start + chunk])
            total += float(F.binary_cross_entropy_with_logits(logits, y[start:start + chunk], reduction="sum"))
    return total / len(x)


def _mean_probability(model, x, chunk):
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            total += float(model(x[start:start + chunk]).sum())
    return total / len(x)


class _BatchStream:
    """Fixed-size minibatches drawn from successive random permutations of the fitting atoms.

    Every update sees exactly ``min(batch_size, n)`` atoms: a batch that runs past the end of
    one permutation continues into the next, so an update is always the same size and the
    number of passes through the data is exactly updates × size / n. When the whole fitting
    set fits in one batch, every update is full-batch.
    """

    def __init__(self, n, batch_size, seed):
        self.n, self.size = int(n), int(min(batch_size, n))
        self.full = self.size == self.n
        self.generator = torch.Generator().manual_seed(int(seed))
        self.order, self.position = torch.randperm(self.n, generator=self.generator), 0

    def next(self):
        if self.full:
            return None
        take = []
        needed = self.size
        while needed:
            if self.position == self.n:
                self.order, self.position = torch.randperm(self.n, generator=self.generator), 0
            chunk = self.order[self.position:self.position + needed]
            take.append(chunk)
            self.position += len(chunk)
            needed -= len(chunk)
        return torch.cat(take)


def fit_field(model, fit_x, fit_y, val_x=None, val_y=None, *, learning_rate=1e-3, batch_size=32768,
              max_updates=2000, eval_every=25, patience=200, batch_seed=0, gradient_penalty=0.0,
              device="cpu", chunk=65536):
    """Adam on unweighted binary cross-entropy; one *update* is one optimiser step, always.

    With validation data, the model at update 0 (the constant it was initialised to) is the
    first candidate checkpoint; validation BCE is evaluated every ``eval_every`` updates, the
    best checkpoint is kept, training stops after ``patience`` updates without improvement or
    at ``max_updates``, and the best checkpoint is restored before returning. Without
    validation data (a refit) the model trains for exactly ``max_updates`` updates and the
    final state is returned.

    ``gradient_penalty`` adds λ·mean‖∇ₓf‖² over each batch's atoms, where f is the logit and x
    the physical coordinates, computed by automatic differentiation.

    Returns a record with the history (update, passes through the data, full fitting-set BCE,
    validation BCE), the selected update, elapsed time, how training stopped, whether a loss
    went non-finite, and the mean fitted probability minus the fitted guest fraction, which is
    reported and not enforced: an early-stopped fit is not at a stationary point.
    """
    if gradient_penalty > 0 and str(device).startswith("mps"):
        # Measured 2026-09-18 on torch 2.10: through this network, gradients with respect to the
        # input coordinates are unreliable on MPS (repeated computations on one model differed
        # by up to 300% from the CPU, sometimes non-finite), although forward passes and weight
        # gradients agree with the CPU to float32 precision. The penalty needs the former.
        raise ValueError("the gradient penalty needs input gradients, which are unreliable on MPS; use the CPU")
    started = time.perf_counter()
    model = model.to(device)
    fx = torch.as_tensor(np.asarray(fit_x), dtype=torch.float32, device=device)
    fy = torch.as_tensor(np.asarray(fit_y), dtype=torch.float32, device=device)
    validating = val_x is not None
    if validating:
        vx = torch.as_tensor(np.asarray(val_x), dtype=torch.float32, device=device)
        vy = torch.as_tensor(np.asarray(val_y), dtype=torch.float32, device=device)
    stream = _BatchStream(len(fx), batch_size, batch_seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def snapshot():
        return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    def evaluate(update):
        row = {"update": update, "passes": update * stream.size / stream.n, "fit_bce": _bce(model, fx, fy, chunk)}
        if validating:
            row["validation_bce"] = _bce(model, vx, vy, chunk)
        return row

    model.eval()
    history = [evaluate(0)]
    best = {"update": 0, "validation_bce": history[0].get("validation_bce"), "state": snapshot()}
    stopped, nonfinite, update = ("fixed" if not validating else "cap"), False, 0
    model.train()
    while update < max_updates:
        index = stream.next()
        xb, yb = (fx, fy) if index is None else (fx[index.to(device)], fy[index.to(device)])
        if gradient_penalty > 0:
            xb = xb.detach().clone().requires_grad_(True)
        logits = model.logits(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        if gradient_penalty > 0:
            gradient = torch.autograd.grad(logits.sum(), xb, create_graph=True)[0]
            loss = loss + gradient_penalty * (gradient ** 2).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            nonfinite, stopped = True, "nonfinite"
            break
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        update += 1
        if validating and (update % eval_every == 0 or update == max_updates):
            model.eval()
            row = evaluate(update)
            model.train()
            history.append(row)
            if not math.isfinite(row["validation_bce"]):
                nonfinite, stopped = True, "nonfinite"
                break
            if row["validation_bce"] < best["validation_bce"]:
                best = {"update": update, "validation_bce": row["validation_bce"], "state": snapshot()}
            elif update - best["update"] >= patience:
                stopped = "patience"
                break
    model.eval()
    if validating:
        model.load_state_dict(best["state"])
    else:
        history.append(evaluate(update))
        best = {"update": update, "validation_bce": None}
    rate = float(fy.mean())
    return {
        "history": history,
        "best_update": int(best["update"]),
        "best_passes": best["update"] * stream.size / stream.n,
        "best_validation_bce": best["validation_bce"],
        "initial_validation_bce": history[0].get("validation_bce"),
        "updates_run": int(update),
        "passes_run": update * stream.size / stream.n,
        "stopped": stopped,
        "nonfinite": nonfinite,
        "full_batch": stream.full,
        "batch_size": stream.size,
        "n_fit": int(len(fx)),
        "n_validation": int(len(vx)) if validating else 0,
        "fit_rate": rate,
        "mean_residual": _mean_probability(model, fx, chunk) - rate,
        "seconds": round(time.perf_counter() - started, 2),
        "device": str(device),
        "parameters": model.parameter_count() if hasattr(model, "parameter_count") else None,
    }


def predict_logits(model, points, device="cpu", chunk=65536):
    """The model's logit at ``points`` as a float64 numpy array."""
    out = []
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(np.asarray(points[start:start + chunk]), dtype=torch.float32, device=device)
            out.append(model.logits(x).detach().cpu().double().numpy())
    return np.concatenate(out) if out else np.zeros(0)
