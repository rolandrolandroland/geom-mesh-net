"""Tests for the misspecified matrix fields of Stage 5.2.

The inverse-square field must keep the boundary condition exactly while breaking the
diffusion equation; the shuffled-surface field must keep the equation exactly while breaking
the capillarity law; and the smoothed noise must have the mean and spread it was given.
Each property is checked against something the implementation does not use: surface
means by direct integration, a Laplacian by automatic differentiation, and sample
statistics.
"""

import numpy as np
import torch

from geom_mesh_net.fields import misspecified as mis
from geom_mesh_net.fields import physics
from tests.test_field_physics import matrix_points, separated_precipitates


def test_closed_form_sphere_mean_of_the_inverse_square_matches_quadrature():
    rng = np.random.default_rng(3)
    radius, distance, source_radius = 2.5, 7.0, 3.0
    directions = rng.normal(size=(400_000, 3))
    points = radius * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    squared = ((points - np.array([distance, 0.0, 0.0])) ** 2).sum(axis=1)
    sampled = np.mean(source_radius ** 2 / squared)
    closed = source_radius ** 2 * np.log((distance + radius) / (distance - radius)) / (2 * radius * distance)
    assert abs(sampled - closed) < 4 * np.std(source_radius ** 2 / squared) / np.sqrt(len(squared))


def test_inverse_square_field_meets_gibbs_thomson_on_every_surface_mean():
    centres, radii = separated_precipitates(seed=4, count=15)
    x = matrix_points(centres, radii)
    field, values = mis.inverse_square_to_matrix_mean(centres, radii, x, ell=3.0, supersaturation=4.0, matrix_mean=0.1)
    error = np.abs(field.surface_means(nodes=400) - field.surface_values)
    assert error.max() < 1e-10 * np.abs(field.amplitudes).max()
    assert abs(values.mean() - 0.1) < 1e-12
    assert abs(field.c_inf / field.c_eq - 4.0) < 1e-12
    assert np.allclose(field(x), values, rtol=0, atol=1e-14)


def test_an_inverse_square_profile_breaks_the_screened_equation_as_derived():
    centre, radius, xi = np.array([20.0, 20.0, 20.0]), 3.0, 4.0
    r = torch.linspace(3.5, 15.0, 40, dtype=torch.float64)
    x = torch.stack([centre[0] + r, torch.full_like(r, centre[1]), torch.full_like(r, centre[2])], dim=1).requires_grad_(True)
    c = radius ** 2 / ((x - torch.as_tensor(centre)) ** 2).sum(dim=1)
    gradient = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
    laplacian = sum(torch.autograd.grad(gradient[:, d].sum(), x, retain_graph=True)[0][:, d] for d in range(3))
    residual = (laplacian - c / xi ** 2).detach().numpy()
    rr = r.numpy()
    assert np.allclose(residual, radius ** 2 / rr ** 4 * (2 - rr ** 2 / xi ** 2), rtol=1e-10, atol=1e-14)
    assert np.abs(residual).max() > 1e-3  # nonzero except on the sphere r = sqrt(2) xi


def test_smoothed_noise_has_the_requested_moments_and_a_finite_correlation_length():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 60, size=(40_000, 3))
    field, values = mis.smoothed_noise_to_matrix(seed=11, length=3.0, matrix_points=x, mean=0.1, sd=0.02)
    assert abs(values.mean() - 0.1) < 1e-12 and abs(values.std() - 0.02) < 1e-12
    assert np.allclose(field(x), values, rtol=0, atol=1e-14)
    again, _ = mis.smoothed_noise_to_matrix(seed=11, length=3.0, matrix_points=x, mean=0.1, sd=0.02)
    assert np.allclose(again(x[:100]), values[:100], rtol=0, atol=1e-15)

    def correlation(shift):
        return np.corrcoef(field(x), field(x + np.array([shift, 0.0, 0.0])))[0, 1]

    assert correlation(1.0) > 0.9 and abs(correlation(25.0)) < 0.1


def test_the_misspecified_fields_differ_from_the_diffusion_field_on_the_same_geometry():
    centres, radii = separated_precipitates(seed=6, count=12)
    x = matrix_points(centres, radii)
    xi = physics.screening_length(radii, 40.0 ** 3)
    diffusion, d_values = physics.fit_to_matrix_mean(centres, radii, x, ell=3.0, supersaturation=4.0, matrix_mean=0.1,
                                                     volume=40.0 ** 3, xi=xi)
    _, s_values = mis.inverse_square_to_matrix_mean(centres, radii, x, ell=3.0, supersaturation=4.0, matrix_mean=0.1)
    assert np.corrcoef(d_values, s_values)[0, 1] < 0.99
    assert np.abs(d_values - s_values).max() > 0.1 * d_values.std()


def test_shuffled_surface_field_solves_the_equation_with_permuted_gibbs_thomson_means():
    centres, radii = separated_precipitates(seed=8, count=14)
    x = matrix_points(centres, radii)
    xi = physics.screening_length(radii, 40.0 ** 3)
    field, values = mis.shuffled_surface_to_matrix_mean(centres, radii, x, ell=3.0, supersaturation=4.0, matrix_mean=0.1,
                                                        volume=None, seed=2, xi=xi)
    assert abs(values.mean() - 0.1) < 1e-12 and np.allclose(field(x), values, rtol=0, atol=1e-14)
    assert not np.array_equal(field.permutation, np.arange(len(radii)))
    # Surface means by direct integration, through a diffusion field with the same amplitudes.
    same = physics.DiffusionField(centres, radii, xi, field.c_eq, field.ell, field.c_inf, field.amplitudes)
    means = physics.surface_means(same, nodes=400)
    assert np.abs(means - field.surface_values).max() < 1e-10 * np.abs(field.amplitudes).max()
    assert np.allclose(np.sort(field.surface_values), np.sort(physics.gibbs_thomson(radii, field.c_eq, field.ell)))
    assert same.pde_residual(x[:500]) < 1e-8
