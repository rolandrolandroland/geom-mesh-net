"""Tests for the conditional normalizing flow.

The flow was written here rather than taken from a library, so the burden of
proof sits with these tests. Three things have to hold, in increasing order of
how much they matter:

1. The transform is exactly invertible, so sampling and density evaluation
   describe the same distribution.
2. The density integrates to one, so ``log_prob`` is a density and not merely a
   score. This is what makes a comparison against the prior meaningful.
3. On a problem whose posterior is known in closed form, the trained flow
   recovers it. A flow can pass (1) and (2) while learning the wrong thing;
   only this catches that.
"""

import numpy as np
import pytest
import torch

from geom_mesh_net.inference.flow import (
    AutoregressiveAffineLayer,
    BoxFlow,
    ConditionalFlow,
    fit,
)


torch.manual_seed(0)


# --------------------------------------------------------------------------
# 1. Invertibility
# --------------------------------------------------------------------------


def test_layer_forward_and_inverse_are_exact_inverses():
    layer = AutoregressiveAffineLayer(dim=4, context_dim=3, hidden=16)
    # Randomize away from the identity initialization, or the test is vacuous.
    for parameter in layer.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.3)

    context = torch.randn(32, 3)
    w = torch.randn(32, 4)
    z, _ = layer(w, context)
    recovered = layer.inverse(z, context)
    assert torch.allclose(w, recovered, atol=1e-5), (w - recovered).abs().max()


def test_full_stack_sampling_inverts_the_density_pass():
    """The flip permutations between layers must unwind in the right order."""
    flow = ConditionalFlow(dim=4, context_dim=3, n_layers=5, hidden=16)
    for parameter in flow.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.2)

    context = torch.randn(16, 3)
    w = torch.randn(16, 4)

    # Push w through the density path by hand, then invert with the sampler's
    # own logic, and check we land back on w.
    current = w
    for position, layer in enumerate(flow.layers):
        current, _ = layer(current, context)
        if position < len(flow.layers) - 1:
            current = current[:, flow.flip]
    z = current
    for position, layer in enumerate(reversed(flow.layers)):
        if position > 0:
            z = z[:, flow.flip]
        z = layer.inverse(z, context)
    assert torch.allclose(w, z, atol=1e-4), (w - z).abs().max()


def test_layer_log_det_matches_autograd():
    """The analytic Jacobian must agree with the one autograd computes."""
    layer = AutoregressiveAffineLayer(dim=3, context_dim=2, hidden=16)
    for parameter in layer.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.3)

    context = torch.randn(1, 2)
    w = torch.randn(1, 3, requires_grad=True)
    _, log_det = layer(w, context)

    jacobian = torch.autograd.functional.jacobian(
        lambda x: layer(x, context)[0].squeeze(0), w
    ).reshape(3, 3)
    expected = torch.log(torch.abs(torch.det(jacobian)))
    assert log_det.item() == pytest.approx(expected.item(), abs=1e-4)


# --------------------------------------------------------------------------
# 2. It is a density
# --------------------------------------------------------------------------


def test_box_flow_density_integrates_to_one():
    """Numerically integrate over a 2D box. Without this, `log_prob` could be
    any old score and the comparison against the prior would mean nothing."""
    low, high = [0.0, -1.0], [2.0, 1.0]
    model = BoxFlow(low, high, context_dim=2, n_layers=3, hidden=16)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.2)

    steps = 220
    xs = torch.linspace(low[0], high[0], steps + 1)[:-1] + (high[0] - low[0]) / (2 * steps)
    ys = torch.linspace(low[1], high[1], steps + 1)[:-1] + (high[1] - low[1]) / (2 * steps)
    grid = torch.cartesian_prod(xs, ys)
    context = torch.zeros(len(grid), 2)

    with torch.no_grad():
        density = torch.exp(model.log_prob(grid, context))
    cell = ((high[0] - low[0]) / steps) * ((high[1] - low[1]) / steps)
    total = (density.sum() * cell).item()
    assert total == pytest.approx(1.0, abs=0.02), total


def test_samples_always_land_inside_the_prior_box():
    """Support is enforced by the sigmoid reparameterization, not by rejection,
    which would bias the posterior."""
    low, high = [0.2, 0.0, 3.0, 0.0], [1.0, 0.05, 15.0, 0.5]
    model = BoxFlow(low, high, context_dim=5, n_layers=4, hidden=16)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 1.5)  # deliberately wild

    with torch.no_grad():
        samples = model.sample(torch.randn(20, 5), n_samples=200)
    assert samples.shape == (20, 200, 4)
    assert torch.all(samples >= torch.tensor(low)), samples.min(dim=1)
    assert torch.all(samples <= torch.tensor(high)), samples.max(dim=1)


def test_log_prior_matches_the_uniform_density():
    low, high = [0.2, 0.0, 3.0, 0.0], [1.0, 0.05, 15.0, 0.5]
    model = BoxFlow(low, high, context_dim=3)
    volume = np.prod(np.array(high) - np.array(low))
    assert model.log_prior().item() == pytest.approx(-np.log(volume), abs=1e-5)


def test_log_prob_stays_finite_at_the_prior_bounds():
    """The logit transform diverges at the bounds; clamping must contain it."""
    low, high = [0.0, 0.0], [1.0, 1.0]
    model = BoxFlow(low, high, context_dim=2, n_layers=2, hidden=16)
    edge = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with torch.no_grad():
        values = model.log_prob(edge, torch.zeros(3, 2))
    assert torch.all(torch.isfinite(values)), values


def test_box_flow_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="upper bound must exceed"):
        BoxFlow([1.0, 0.0], [0.5, 1.0], context_dim=2)


def test_untrained_flow_starts_near_the_base_distribution():
    """Zero-initialized output layers mean the transform starts as the
    identity, so training does not begin by undoing random scales."""
    flow = ConditionalFlow(dim=3, context_dim=2, n_layers=4, hidden=16)
    w = torch.randn(64, 3)
    with torch.no_grad():
        value = flow.log_prob(w, torch.randn(64, 2))
    expected = (-0.5 * (w**2 + np.log(2 * np.pi))).sum(dim=1)
    assert torch.allclose(value, expected, atol=1e-5)


# --------------------------------------------------------------------------
# 3. It learns the right distribution
# --------------------------------------------------------------------------


def test_flow_recovers_an_analytic_gaussian_posterior():
    """The decisive test: a conjugate problem with a closed-form posterior.

    theta ~ N(0, 1) and s | theta ~ N(theta, sigma^2) give

        theta | s ~ N(s / (1 + sigma^2), sigma^2 / (1 + sigma^2))

    Training on samples from the joint must reproduce that mean and standard
    deviation. A flow can be perfectly invertible and correctly normalized while
    still learning the wrong distribution; this is what would catch it.
    """
    torch.manual_seed(3)
    sigma = 0.5
    n = 6000
    theta = torch.randn(n, 1)
    context = theta + sigma * torch.randn(n, 1)

    flow = ConditionalFlow(dim=1, context_dim=1, n_layers=4, hidden=32)
    optimizer = torch.optim.Adam(flow.parameters(), lr=5e-3)
    for _ in range(600):
        index = torch.randint(0, n, (512,))
        loss = -flow.log_prob(theta[index], context[index]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    posterior_variance = sigma**2 / (1 + sigma**2)
    posterior_sd = np.sqrt(posterior_variance)

    probe = torch.tensor([[-1.5], [0.0], [1.5]])
    with torch.no_grad():
        samples = flow.sample(probe.repeat_interleave(4000, dim=0))
    samples = samples.view(3, 4000)

    for row, s_value in enumerate(probe.squeeze(1).tolist()):
        expected_mean = s_value / (1 + sigma**2)
        assert samples[row].mean().item() == pytest.approx(expected_mean, abs=0.08), (
            s_value, samples[row].mean().item(), expected_mean
        )
        assert samples[row].std().item() == pytest.approx(posterior_sd, abs=0.08), (
            s_value, samples[row].std().item(), posterior_sd
        )


def test_flow_learns_a_context_dependent_width():
    """Heteroscedastic case: the posterior width must track the context, which a
    single global scale could not represent."""
    torch.manual_seed(4)
    n = 6000
    context = torch.rand(n, 1) * 2.0 - 1.0
    width = 0.1 + 0.9 * (context + 1.0) / 2.0
    theta = width * torch.randn(n, 1)

    flow = ConditionalFlow(dim=1, context_dim=1, n_layers=4, hidden=32)
    optimizer = torch.optim.Adam(flow.parameters(), lr=5e-3)
    for _ in range(600):
        index = torch.randint(0, n, (512,))
        loss = -flow.log_prob(theta[index], context[index]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        narrow = flow.sample(torch.full((4000, 1), -0.9)).std().item()
        wide = flow.sample(torch.full((4000, 1), 0.9)).std().item()
    assert narrow == pytest.approx(0.145, abs=0.06), narrow
    assert wide == pytest.approx(0.955, abs=0.12), wide
    assert wide > 3 * narrow


def test_fit_improves_validation_log_prob_over_the_prior():
    """The Stage 2 gate in miniature: a learnable relationship must beat the
    uniform prior on held-out data."""
    torch.manual_seed(5)
    n = 2000
    low, high = [0.0, 0.0], [1.0, 1.0]
    context = torch.rand(n, 3)
    theta = torch.stack(
        [
            (0.8 * context[:, 0] + 0.1 + 0.02 * torch.randn(n)).clamp(0.01, 0.99),
            torch.rand(n),  # independent of the context
        ],
        dim=1,
    )

    model = BoxFlow(low, high, context_dim=3, n_layers=4, hidden=32)
    history, best_val = fit(
        model,
        theta[:1600], context[:1600],
        theta[1600:], context[1600:],
        max_epochs=80, patience=20, verbose=False,
    )
    assert len(history) > 0
    assert best_val > model.log_prior().item() + 0.5, (
        best_val, model.log_prior().item()
    )


def test_fit_does_not_beat_the_prior_on_pure_noise():
    """The complement: when theta is independent of the context there is nothing
    to learn, and the flow must not appear to learn it. Guards against a gate
    that passes on anything."""
    torch.manual_seed(6)
    n = 1500
    low, high = [0.0], [1.0]
    context = torch.rand(n, 3)
    theta = torch.rand(n, 1)  # independent

    model = BoxFlow(low, high, context_dim=3, n_layers=3, hidden=32)
    _, best_val = fit(
        model,
        theta[:1200], context[:1200],
        theta[1200:], context[1200:],
        max_epochs=60, patience=15, verbose=False,
    )
    # A uniform on [0,1] has log density 0; the flow should sit near it.
    assert best_val < 0.3, best_val


# --------------------------------------------------------------------------
# Stage 2 split integrity
# --------------------------------------------------------------------------


def test_split_is_disjoint_and_covers_everything():
    """Leakage between splits would invalidate every calibration number in
    Stage 3, silently and without any error."""
    from experiments.inference.fit_posterior import split_indices

    train, validation, test = split_indices(1000)
    assert len(train) == 800 and len(validation) == 100 and len(test) == 100
    combined = np.concatenate([train, validation, test])
    assert len(np.unique(combined)) == 1000
    assert np.array_equal(np.sort(combined), np.arange(1000))


def test_split_is_deterministic_across_calls():
    """Stage 2 and Stage 3 recompute the split independently; they must agree
    or Stage 3 would calibrate on patterns the flow trained on."""
    from experiments.inference.fit_posterior import split_indices

    first = split_indices(1000)
    second = split_indices(1000)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_split_changes_with_seed():
    from experiments.inference.fit_posterior import split_indices

    _, _, test_a = split_indices(1000, seed=42)
    _, _, test_b = split_indices(1000, seed=43)
    assert not np.array_equal(test_a, test_b)
