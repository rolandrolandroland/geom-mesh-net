"""A conditional autoregressive normalizing flow, for neural posterior estimation.

Models q(theta | s): a distribution over the four cluster parameters given the
14 spatial-summary features. Trained by maximum likelihood on simulated pairs,
its minimizer is the true posterior p(theta | s), because the pairs are drawn
from the joint p(theta) p(x | theta). See ``experiments/inference/ROADMAP.md``
section 3.

Written directly rather than pulled from a library. The parameter space is four
dimensional, so the whole thing is small, and being able to check it against a
case with a closed-form posterior is stronger evidence of correctness than
trusting an unverified dependency. ``tests/test_flow.py`` does exactly that.

Design
------
Each layer applies an autoregressive affine transform: for dimension i, a small
MLP reads the context and the *preceding* dimensions and emits a shift and a log
scale,

    z_i = (w_i - shift_i(context, w_<i)) * exp(-log_scale_i(context, w_<i))

which is triangular, so its Jacobian determinant is just the product of the
scales. Density evaluation is one parallel pass; sampling inverts one dimension
at a time. Dimensions are reversed between layers so every dimension conditions
on every other across the stack.

With only four dimensions there is no need for MADE-style masking: one small
network per (layer, dimension) is more explicit and easier to verify.

Support
-------
The prior is uniform on a box, so the posterior is too. The flow works in an
unconstrained space reached by

    u = (theta - low) / (high - low)  in (0, 1),   w = logit(u)

and samples are mapped back with a sigmoid. Every sample therefore lands inside
the prior box by construction, rather than by rejection, which would bias the
result. The change of variables contributes a Jacobian term to the density,
included in :meth:`BoxFlow.log_prob` so the reported log-likelihood is a genuine
density over theta and can be compared against the uniform prior.
"""

import numpy as np
import torch
import torch.nn as nn


LOGIT_EPSILON = 1e-6


def _mlp(in_features, hidden, out_features, depth):
    layers = []
    width = in_features
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.Tanh()]
        width = hidden
    layers.append(nn.Linear(width, out_features))
    network = nn.Sequential(*layers)
    # Start close to the identity transform: zero shift, unit scale. Without
    # this the initial log scales are arbitrary and early training spends its
    # time undoing them.
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class AutoregressiveAffineLayer(nn.Module):
    """One autoregressive affine transform, conditioned on the context."""

    def __init__(self, dim, context_dim, hidden=64, depth=2, max_log_scale=5.0):
        super().__init__()
        self.dim = dim
        self.max_log_scale = max_log_scale
        # Dimension i sees the context plus dimensions 0..i-1.
        self.conditioners = nn.ModuleList(
            [_mlp(context_dim + i, hidden, 2, depth) for i in range(dim)]
        )

    def _shift_and_log_scale(self, index, context, preceding):
        inputs = context if index == 0 else torch.cat([context, preceding], dim=1)
        output = self.conditioners[index](inputs)
        shift = output[:, 0]
        # Bounded so a diverging scale cannot produce inf or nan mid-training.
        log_scale = self.max_log_scale * torch.tanh(
            output[:, 1] / self.max_log_scale
        )
        return shift, log_scale

    def forward(self, w, context):
        """w -> z. Returns (z, log|det J|). One parallel pass."""
        outputs, log_det = [], torch.zeros(len(w), device=w.device, dtype=w.dtype)
        for index in range(self.dim):
            shift, log_scale = self._shift_and_log_scale(
                index, context, w[:, :index]
            )
            outputs.append((w[:, index] - shift) * torch.exp(-log_scale))
            log_det = log_det - log_scale
        return torch.stack(outputs, dim=1), log_det

    def inverse(self, z, context):
        """z -> w. Sequential over dimensions, since each needs its predecessors."""
        w = torch.zeros_like(z)
        for index in range(self.dim):
            shift, log_scale = self._shift_and_log_scale(
                index, context, w[:, :index].clone()
            )
            w[:, index] = z[:, index] * torch.exp(log_scale) + shift
        return w


class ConditionalFlow(nn.Module):
    """A stack of autoregressive layers over a standard normal base."""

    def __init__(self, dim, context_dim, n_layers=5, hidden=64, depth=2):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                AutoregressiveAffineLayer(dim, context_dim, hidden, depth)
                for _ in range(n_layers)
            ]
        )
        # Reversing between layers lets every dimension condition on every
        # other one somewhere in the stack.
        self.register_buffer("flip", torch.arange(dim - 1, -1, -1))

    def log_prob(self, w, context):
        log_det = torch.zeros(len(w), device=w.device, dtype=w.dtype)
        for position, layer in enumerate(self.layers):
            w, layer_log_det = layer(w, context)
            log_det = log_det + layer_log_det
            if position < len(self.layers) - 1:
                w = w[:, self.flip]
        base = -0.5 * (w**2 + np.log(2.0 * np.pi))
        return base.sum(dim=1) + log_det

    def sample(self, context, generator=None):
        z = torch.randn(
            len(context), self.dim, device=context.device,
            dtype=context.dtype, generator=generator,
        )
        for position, layer in enumerate(reversed(self.layers)):
            if position > 0:
                z = z[:, self.flip]
            z = layer.inverse(z, context)
        return z


class BoxFlow(nn.Module):
    """A conditional flow over a uniform box prior.

    Handles the logit reparameterization and its Jacobian, so ``log_prob``
    returns a density over theta itself and ``sample`` returns theta inside the
    prior box.
    """

    def __init__(self, low, high, context_dim, **flow_kwargs):
        super().__init__()
        low = torch.as_tensor(low, dtype=torch.float32)
        high = torch.as_tensor(high, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("every prior upper bound must exceed its lower bound")
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.flow = ConditionalFlow(len(low), context_dim, **flow_kwargs)

    @property
    def dim(self):
        return len(self.low)

    def to_unconstrained(self, theta):
        """theta -> w, plus the log Jacobian d(theta)/d(w) summed over dims."""
        span = self.high - self.low
        u = (theta - self.low) / span
        if torch.any(u <= 0) or torch.any(u >= 1):
            u = u.clamp(LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
        w = torch.log(u) - torch.log1p(-u)
        log_dtheta_dw = (torch.log(span) + torch.log(u) + torch.log1p(-u)).sum(dim=1)
        return w, log_dtheta_dw

    def to_constrained(self, w):
        u = torch.sigmoid(w)
        return self.low + (self.high - self.low) * u

    def log_prob(self, theta, context):
        """log q(theta | context), a proper density over theta."""
        w, log_dtheta_dw = self.to_unconstrained(theta)
        return self.flow.log_prob(w, context) - log_dtheta_dw

    def sample(self, context, n_samples=1, generator=None):
        """Draw (len(context), n_samples, dim) samples, all inside the box."""
        repeated = context.repeat_interleave(n_samples, dim=0)
        w = self.flow.sample(repeated, generator=generator)
        theta = self.to_constrained(w)
        return theta.view(len(context), n_samples, self.dim)

    def log_prior(self):
        """log density of the uniform prior. The baseline the flow must beat."""
        return -torch.log(self.high - self.low).sum()


def fit(
    model,
    theta_train,
    context_train,
    theta_val,
    context_val,
    learning_rate=1e-3,
    batch_size=128,
    max_epochs=400,
    patience=30,
    seed=0,
    verbose=True,
):
    """Train by maximum likelihood, keeping the best validation checkpoint.

    Returns (history, best_val_log_prob). Early stopping is on validation
    log-likelihood, so the reported number is not the one being optimized
    directly on.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 3
    )

    n = len(theta_train)
    generator = torch.Generator().manual_seed(seed)
    best_val = -np.inf
    best_state = None
    since_improvement = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            loss = -model.log_prob(theta_train[index], context_train[index]).mean()
            optimizer.zero_grad()
            loss.backward()
            # The logit space is unbounded; clipping keeps a rare extreme
            # example from destabilizing the run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(index)

        model.eval()
        with torch.no_grad():
            val = model.log_prob(theta_val, context_val).mean().item()
        scheduler.step(val)
        history.append(
            {"epoch": epoch, "train_log_prob": -total / n, "val_log_prob": val}
        )

        if val > best_val + 1e-4:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break

        if verbose and epoch % 25 == 0:
            print(
                f"  epoch {epoch:4d}  train {-total / n:8.3f}  val {val:8.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val
