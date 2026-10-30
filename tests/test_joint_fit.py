"""Tests for fitting the diffusion law and the precipitates together (Stage 5.2's follow-up).

The model must agree with the analytic field away from interfaces, and the fit must recover
radii and constants from labels it generated itself.
"""

import numpy as np
import pytest
import torch

from geom_mesh_net.fields import joint_fit, physics
from tests.test_field_physics import separated_precipitates


def test_the_matrix_far_from_interfaces_is_the_analytic_field():
    centres, radii = separated_precipitates(seed=11, count=8)
    xi, c_eq, ell, c_inf = 6.0, 0.02, 3.0, 0.09
    field = physics.solve_field(centres, radii, xi, c_eq, ell, c_inf)
    model = joint_fit.JointDiffusionModel(centres, radii, c_eq=c_eq, ell=ell, xi=xi, c_inf=c_inf, width=0.3)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 40, size=(4000, 3))
    gap = np.linalg.norm(x[:, None, :] - centres[None], axis=-1) - radii[None]
    distance = gap.min(axis=1)

    # The interior leaks outward through the logistic blend, by about exp(-d / w) times the
    # jump at the interface. It is negligible where the matrix is fitted, and it is the reason
    # the interface width matters: at w = 0.3 nm, 5 nm out is 1e-7 and 1 nm out is 3%.
    far = x[distance > 5.0]
    assert np.abs(joint_fit.predict(model, far) - field(far)).max() < 1e-6
    near = x[(distance > 0.9) & (distance < 1.1)]
    leak = np.abs(joint_fit.predict(model, near) - field(near)).max()
    assert 1e-3 < leak < 0.1, leak


def test_amplitudes_match_the_analytic_solve_for_the_current_radii():
    centres, radii = separated_precipitates(seed=12, count=6)
    model = joint_fit.JointDiffusionModel(centres, radii, c_eq=0.03, ell=2.5, xi=5.0, c_inf=0.1, width=0.3)
    expected = physics.solve_field(centres, radii, 5.0, 0.03, 2.5, 0.1).amplitudes
    assert np.abs(model.amplitudes().detach().numpy() - expected).max() < 1e-12


def test_the_fit_recovers_radii_and_constants_from_its_own_labels():
    """Labels drawn from the model, then fitted from a deliberately wrong start."""
    centres, radii = separated_precipitates(seed=13, count=10, box=40.0)
    truth = dict(c_eq=0.03, ell=3.0, xi=6.0, c_inf=0.12)
    width = 0.3
    generator = joint_fit.JointDiffusionModel(centres, radii, width=width, rho_edge=0.2, m=3.0, **truth)
    rng = np.random.default_rng(14)
    x = rng.uniform(0, 40, size=(160_000, 3))
    p = joint_fit.predict(generator, x)
    y = (rng.random(len(x)) < p).astype(float)

    start = radii * np.exp(0.12 * rng.standard_normal(len(radii)))
    model = joint_fit.JointDiffusionModel(centres, start, c_eq=0.05, ell=float(radii.mean()), xi=4.0,
                                          c_inf=float(y.mean()), width=width, rho_edge=0.3, m=3.0)
    record = joint_fit.fit_joint(model, x, y, iterations=120, rounds=2)

    before = np.median(np.abs(start / radii - 1))
    after = np.median(np.abs(record["radii"] / radii - 1))
    assert after < before / 2, (before, after)
    assert abs(record["constants"]["ell"] / truth["ell"] - 1) < 0.25, record["constants"]
    assert abs(record["constants"]["xi"] / truth["xi"] - 1) < 0.35, record["constants"]


def test_assignment_is_by_surface_distance():
    centres = np.array([[10.0, 10.0, 10.0], [24.0, 10.0, 10.0]])
    radii = np.array([5.0, 2.0])
    model = joint_fit.JointDiffusionModel(centres, radii, c_eq=0.02, ell=2.0, xi=5.0, c_inf=0.1, width=0.3)
    points = torch.as_tensor(np.array([[16.0, 10.0, 10.0], [23.0, 10.0, 10.0]]), dtype=torch.float64)
    assert model.assign(points).tolist() == [0, 1]   # the large sphere's surface is nearer at x = 16


def test_a_flexible_interior_follows_a_profile_the_power_law_cannot():
    """The knotted profile fits an interior outside the power-law family, at no cost to ell.

    It is not what recovers the capillary length — freeing the centres is (roadmap Stage 5) — and
    it does not recover the profile exactly either, since few atoms sit near any one fractional
    radius. It has to follow it better than the power law can, and not pay for the fit in ell.
    """
    centres, radii = separated_precipitates(seed=15, count=8, box=36.0)
    truth = dict(c_eq=0.03, ell=3.0, xi=6.0, c_inf=0.12)
    shouldered = [0.99, 0.985, 0.97, 0.93, 0.80, 0.45, 0.06]   # flat, then a sharp fall at the rim
    generator = joint_fit.JointDiffusionModel(centres, radii, width=0.3, knots=6, levels=shouldered, **truth)
    rng = np.random.default_rng(16)
    x = rng.uniform(0, 36, size=(120_000, 3))
    y = (rng.random(len(x)) < joint_fit.predict(generator, x)).astype(float)

    def fitted(**shape):
        model = joint_fit.JointDiffusionModel(centres, radii, c_eq=0.05, ell=float(radii.mean()), xi=4.0,
                                              c_inf=float(y.mean()), width=0.3, **shape)
        return joint_fit.fit_joint(model, x, y, iterations=100, rounds=2), model

    power_law, _ = fitted(rho_edge=0.3, m=3.0)
    flexible, model = fitted(knots=6)
    knots = torch.linspace(0, 1, 7, dtype=torch.float64)
    power_curve = 1 - (1 - power_law["shape"]["rho_edge"]) * knots ** power_law["shape"]["m"]
    gaps = {"power_law": float(np.abs(power_curve.numpy() - shouldered).max()),
            "flexible": float(np.abs(np.array(flexible["shape"]["levels"]) - shouldered).max())}
    levels = np.array(flexible["shape"]["levels"])
    assert np.all(np.diff(levels) <= 0) and levels.min() > 0, levels   # decreasing, and a probability
    assert gaps["flexible"] < gaps["power_law"], gaps
    assert flexible["train_loss"] <= power_law["train_loss"]
    error = {k: abs(r["constants"]["ell"] / truth["ell"] - 1) for k, r in
             (("power_law", power_law), ("flexible", flexible))}
    assert error["flexible"] <= error["power_law"] + 0.02, error


def test_with_geometry_carries_the_fitted_profile_onto_new_precipitates():
    """``grow_geometry`` rebuilds the model when it adds one; nothing fitted may be lost."""
    centres, radii = separated_precipitates(seed=17, count=5)
    model = joint_fit.JointDiffusionModel(centres, radii, c_eq=0.02, ell=2.0, xi=5.0, c_inf=0.1, width=0.4,
                                          knots=6, free_centres=True)
    with torch.no_grad():
        model.logit_ratios.add_(0.3)          # move the profile away from its starting power law
    grown = joint_fit.with_geometry(model, np.vstack([centres, [[1.0, 1.0, 1.0]]]), np.append(radii, 2.0))

    assert np.abs(grown.levels().detach().numpy() - model.levels().detach().numpy()).max() < 1e-9
    assert grown.constants() == model.constants()
    assert float(grown.width()) == 0.4 and grown.offsets is not None
    assert len(grown.radii()) == len(radii) + 1


def _growth_with_known_gain(monkeypatch, *, free_centres, added, penalty, gain, knots=None):
    """Separate the birth decision from optimization and peak-detection noise."""
    model = joint_fit.JointDiffusionModel(
        [[10.0, 10.0, 10.0]], [2.0], c_eq=0.02, ell=2.0, xi=5.0,
        c_inf=0.1, width=0.3, free_centres=free_centres, free_width=True, knots=knots,
    )
    x, y = np.zeros((256, 3)), np.zeros(256)
    losses = iter([1.0, 1.0 - gain / len(x)])
    monkeypatch.setattr(joint_fit, "fit_joint", lambda *args, **kwargs: {"train_loss": next(losses)})
    centres = np.column_stack([20.0 + 10.0 * np.arange(added), np.full(added, 10.0),
                               np.full(added, 10.0)])
    monkeypatch.setattr(joint_fit, "residual_peaks", lambda *args, **kwargs: (centres, np.full(added, 1.5)))
    result, record = joint_fit.grow_geometry(model, x, y, rounds=1, penalty=penalty)
    return model, result, record


@pytest.mark.parametrize("free_centres", [False, True])
@pytest.mark.parametrize("added", [1, 3])
@pytest.mark.parametrize("penalty", [None, 2.0])
@pytest.mark.parametrize("knots", [None, 6])
@pytest.mark.parametrize("excess_gain", [-0.25, 0.25])
def test_growth_charges_only_added_active_geometry_parameters(
        monkeypatch, free_centres, added, penalty, knots, excess_gain):
    """A free centre costs three extra coordinates; shared profile/width terms do not."""
    added_parameters = added * (4 if free_centres else 1)
    penalty_per_parameter = 0.5 * np.log(256) if penalty is None else penalty
    required_gain = penalty_per_parameter * added_parameters
    original, result, record = _growth_with_known_gain(
        monkeypatch, free_centres=free_centres, added=added, penalty=penalty,
        gain=required_gain + excess_gain, knots=knots,
    )

    decision = record["rounds"][0]
    kept = excess_gain > 0
    assert decision["added"] == added
    assert decision["added_parameters"] == added_parameters
    assert decision["penalty_nats"] == pytest.approx(required_gain)
    assert decision["gain_nats"] == pytest.approx(required_gain + excess_gain)
    assert decision["kept"] is kept
    assert (result is original) is (not kept)
    assert record["precipitates"] == 1 + (added if kept else 0)


@pytest.mark.parametrize("free_centres", [False, True])
@pytest.mark.parametrize("added", [1, 3])
def test_growth_rejects_a_gain_exactly_equal_to_its_parameter_penalty(monkeypatch, free_centres, added):
    """The criterion is strict; use binary-exact losses to exercise equality."""
    penalty = 2.0
    required_gain = penalty * added * (4 if free_centres else 1)
    original, result, record = _growth_with_known_gain(
        monkeypatch, free_centres=free_centres, added=added, penalty=penalty, gain=required_gain,
    )
    decision = record["rounds"][0]
    assert decision["gain_nats"] == decision["penalty_nats"] == required_gain
    assert decision["kept"] is False
    assert result is original
    assert record["precipitates"] == 1
