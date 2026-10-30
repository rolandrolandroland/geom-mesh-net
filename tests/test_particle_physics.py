"""Failure accounting and physical-domain checks for the shared particle fitter.

Optimisation is substituted in most tests so the checks exercise result selection
and validity without repeating the expensive joint-fit recovery experiments.
"""

import json
import math

import numpy as np
import pytest
import torch

from geom_mesh_net.fields import joint_fit, particle_physics as pp


CENTRES = np.array([[0.0, 0.0, 0.0]])
RADII = np.array([2.0])
X = np.array([[8.0, 0, 0], [9.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0]])
Y = np.array([0.0, 1.0, 0.0, 0.0])


def fitted_record(model, loss=0.4):
    return {"train_loss": loss, "round_losses": [loss], "constants": model.constants(),
            "radii": model.radii().detach().numpy(), "shape": model.shape(), "seconds": 0.0}


def test_no_particles_and_overlapping_initial_particles_are_recorded_without_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid initial geometry must not enter the optimiser")

    monkeypatch.setattr(joint_fit, "fit_joint", forbidden)
    model, empty = pp.fit_particle_physics([], [], X, Y)
    assert model is None and empty["status"] == "no_particles" and not empty["reliable"]
    model, overlapping = pp.fit_particle_physics([[0, 0, 0], [3, 0, 0]], [2, 2], X, Y)
    assert model is None and overlapping["status"] == "invalid_geometry"
    assert overlapping["geometry_before"]["overlap_pairs"] == 1
    assert overlapping["geometry_before"]["minimum_surface_gap"] == -1.0
    json.dumps([empty, overlapping], allow_nan=False)


def test_empty_observations_are_an_explicit_failure():
    model, record = pp.fit_particle_physics(CENTRES, RADII, np.empty((0, 3)), np.empty(0))
    assert model is None and record["status"] == "no_observations"


def test_initialisation_uses_observed_labels_and_fixed_fitting_configuration(monkeypatch):
    calls = []

    def fake_fit(model, x, y, iterations, rounds):
        calls.append((model.constants(), model.knots, model.offsets is not None, iterations, rounds))
        np.testing.assert_array_equal(x, X)
        np.testing.assert_array_equal(y, Y)
        return fitted_record(model)

    monkeypatch.setattr(joint_fit, "fit_joint", fake_fit)
    model, record = pp.fit_particle_physics(CENTRES, RADII, X, Y, iterations=17, rounds=2,
                                           starts=((None, None),), validation_points=X + 2)
    assert calls[0][0]["c_inf"] == pytest.approx(Y.mean())
    assert calls[0][0]["c_eq"] == pytest.approx(Y.mean() / 2)
    assert calls[0][0]["ell"] == pytest.approx(RADII.mean())
    assert calls[0][1:] == (6, True, 17, 2)
    assert model is not None and record["status"] == "ok" and record["reliable"]
    assert record["matrix_probability"]["source"] == "validation_positions"
    assert record["matrix_probability"]["matrix_points"] == len(X)
    json.dumps(record, allow_nan=False)


def test_start_selection_uses_training_loss_even_when_selected_field_is_invalid(monkeypatch):
    calls = []

    def fake_fit(model, x, y, **kwargs):
        calls.append(model)
        if len(calls) == 2:
            with torch.no_grad():
                model.log_c_eq.fill_(math.log(0.9))
                model.log_ell.fill_(math.log(4.0))
                model.log_xi.fill_(math.log(3.0))
            return fitted_record(model, loss=0.2)
        return fitted_record(model, loss=0.4)

    monkeypatch.setattr(joint_fit, "fit_joint", fake_fit)
    held_positions = np.array([[2.1, 0.0, 0.0]])
    model, record = pp.fit_particle_physics(CENTRES, RADII, X, Y,
                                           starts=((None, None), (1.0, 3.0)), validation_points=held_positions)
    assert model is calls[1] and record["selected_start"] == 1
    assert record["train_loss"] == 0.2
    assert record["matrix_probability"]["maximum"] > 1
    assert record["matrix_probability"]["outside_probability_range"] == 1
    assert not record["reliable"] and "invalid_matrix_probability" in record["flags"]
    # The legacy prediction helper would conceal this invalid matrix concentration.
    assert joint_fit.predict(model, held_positions, matrix_only=True)[0] < 1


def test_fitted_overlapping_spheres_are_retained_but_marked_invalid(monkeypatch):
    def fake_fit(model, *args, **kwargs):
        with torch.no_grad():
            model.offsets[1, 0] = -6.0
        return fitted_record(model)

    monkeypatch.setattr(joint_fit, "fit_joint", fake_fit)
    model, record = pp.fit_particle_physics([[0, 0, 0], [8, 0, 0]], [2, 2], X, Y,
                                           starts=((1.0, 3.0),), validation_points=X + 20)
    assert model is not None and record["geometry_before"]["separated"]
    assert not record["geometry_after"]["separated"]
    assert record["geometry_after"]["overlap_pairs"] == 1
    assert not record["reliable"] and "invalid_fitted_geometry" in record["flags"]


def test_bound_estimates_are_not_reported_as_reliable(monkeypatch):
    def fake_fit(model, *args, **kwargs):
        with torch.no_grad():
            model.log_xi.fill_(math.log(joint_fit.BOUNDS["xi"][1]))
            model.log_radii.fill_(math.log(joint_fit.RADIUS_BOUNDS[0]))
        return fitted_record(model)

    monkeypatch.setattr(joint_fit, "fit_joint", fake_fit)
    _, record = pp.fit_particle_physics(CENTRES, RADII, X, Y, starts=((1.0, 3.0),))
    assert record["at_bounds"] == ["xi"] and record["radius_at_bounds"] == [0]
    assert not record["reliable"]
    assert {"physical_constant_at_bound", "radius_at_bound"}.issubset(record["flags"])


def test_failed_and_nonfinite_starts_remain_in_json_safe_record(monkeypatch):
    calls = []

    def fake_fit(model, *args, **kwargs):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("singular coupling matrix")
        return fitted_record(model, loss=float("nan"))

    monkeypatch.setattr(joint_fit, "fit_joint", fake_fit)
    model, record = pp.fit_particle_physics(CENTRES, RADII, X, Y, starts=((1.0, 3.0), (2.0, 4.0)))
    assert model is None and record["status"] == "fit_failed"
    assert len(record["starts"]) == 2
    assert record["starts"][0]["error"] == "RuntimeError: singular coupling matrix"
    assert record["starts"][1]["status"] == "nonfinite" and record["starts"][1]["train_loss"] is None
    json.dumps(record, allow_nan=False)


def test_probability_check_does_not_apply_matrix_equation_inside_particles(monkeypatch):
    monkeypatch.setattr(joint_fit, "fit_joint", lambda model, *args, **kwargs: fitted_record(model))
    _, record = pp.fit_particle_physics(CENTRES, RADII, X, Y, starts=((1.0, 3.0),),
                                       validation_points=np.array([[0.0, 0, 0], [8.0, 0, 0]]))
    assert record["matrix_probability"]["supplied_points"] == 2
    assert record["matrix_probability"]["matrix_points"] == 1
    assert record["matrix_probability"]["passed"]


def test_missing_matrix_validation_points_cannot_silently_pass(monkeypatch):
    monkeypatch.setattr(joint_fit, "fit_joint", lambda model, *args, **kwargs: fitted_record(model))
    _, record = pp.fit_particle_physics(CENTRES, RADII, X, Y, starts=((1.0, 3.0),),
                                       validation_points=np.array([[0.0, 0, 0]]))
    assert not record["reliable"] and "no_matrix_check_points" in record["flags"]


def test_invalid_observation_shapes_and_values_are_rejected():
    with pytest.raises(ValueError, match="labels in"):
        pp.fit_particle_physics(CENTRES, RADII, X, [0, 1, float("nan"), 0])
    with pytest.raises(ValueError, match="centres must"):
        pp.fit_particle_physics([0, 0, 0], RADII, X, Y)
    with pytest.raises(ValueError, match="positive integers"):
        pp.fit_particle_physics(CENTRES, RADII, X, Y, iterations=0)


def test_short_real_fit_returns_finite_loss_and_serialisable_diagnostics():
    # A tiny optimiser integration check; parameter recovery belongs to test_joint_fit.
    _, record = pp.fit_particle_physics(CENTRES, RADII, X, Y, iterations=2, rounds=1,
                                       starts=((1.0, 3.0),))
    assert record["train_loss"] is not None and np.isfinite(record["train_loss"])
    assert len(record["starts"]) == 1
    json.dumps(record, allow_nan=False)
