"""Tests for the Stage 5 diffusion field.

Each property is checked against something the implementation does not use: the
textbook single-precipitate profile, a Laplacian taken by automatic differentiation,
surface means computed by direct integration rather than the mean-value identity the
solver relies on, and the closed-form screening length of equal spheres.
"""

import numpy as np
import pytest

from geom_mesh_net.fields import physics


def separated_precipitates(seed=0, count=12, box=40.0, gap=1.0, radius=None):
    rng = np.random.default_rng(seed)
    centres, radii = [], []
    while len(radii) < count:
        c, r = rng.uniform(6, box - 6, size=3), rng.uniform(2.0, 4.0) if radius is None else radius
        if all(np.linalg.norm(c - c2) >= r + r2 + gap for c2, r2 in zip(centres, radii)):
            centres.append(c)
            radii.append(r)
    return np.array(centres), np.array(radii)


def matrix_points(centres, radii, n=3000, seed=1, box=40.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, box, size=(n * 3, 3))
    outside = np.all(np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=-1) > radii[None, :] + 0.05, axis=1)
    return x[outside][:n]


def surface_means(field, nodes=400):
    """Mean of c over each precipitate's surface, by Gauss-Legendre integration in cos(angle).

    Precipitate j's source depends only on the angle between the surface point and the
    direction to j, so each term is a one-dimensional integral. The mean-value identity
    the solver uses is not involved.
    """
    mu, weights = np.polynomial.legendre.leggauss(nodes)
    means = np.full(len(field.radii), field.c_inf)
    for k, (ck, rk) in enumerate(zip(field.centres, field.radii)):
        for j, (cj, rj) in enumerate(zip(field.centres, field.radii)):
            if j == k:
                means[k] += field.amplitudes[j]  # its own source is exactly 1 on its surface
                continue
            d = np.linalg.norm(cj - ck)
            r = np.sqrt(rk ** 2 + d ** 2 - 2 * rk * d * mu)
            means[k] += field.amplitudes[j] * 0.5 * np.sum(weights * physics.source_kernel(r, rj, field.xi))
    return means


def test_a_single_precipitate_gives_the_textbook_profile():
    centre, radius, xi = np.array([[10.0, 10.0, 10.0]]), np.array([3.0]), 5.0
    field = physics.solve_field(centre, radius, xi, c_eq=0.02, ell=1.5, c_inf=0.05)
    r = np.linspace(3.0, 25.0, 50)
    points = centre + np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])
    c_k = 0.02 * np.exp(1.5 / 3.0)
    expected = 0.05 + (c_k - 0.05) * (3.0 / r) * np.exp(-(r - 3.0) / xi)
    assert np.allclose(field(points), expected, rtol=0, atol=1e-14)


def test_the_field_solves_the_screened_diffusion_equation():
    centres, radii = separated_precipitates()
    field = physics.solve_field(centres, radii, xi=4.0, c_eq=0.02, ell=2.0, c_inf=0.06)
    assert field.pde_residual(matrix_points(centres, radii, n=1500)) < 1e-10


def test_surface_means_equal_the_gibbs_thomson_values():
    centres, radii = separated_precipitates(gap=0.5)
    field = physics.solve_field(centres, radii, xi=5.0, c_eq=0.02, ell=2.0, c_inf=0.06)
    amplitude = np.abs(field.surface_values - field.c_inf).max()
    assert np.max(np.abs(surface_means(field) - field.surface_values)) < 1e-10 * max(amplitude, 1.0)


def test_adding_single_profiles_misses_the_boundary_condition():
    """The naive superposition the solve replaces: neighbours shift every surface mean."""
    centres, radii = separated_precipitates(gap=0.5)
    solved = physics.solve_field(centres, radii, xi=5.0, c_eq=0.02, ell=2.0, c_inf=0.06)
    naive = physics.DiffusionField(centres, radii, solved.xi, solved.c_eq, solved.ell, solved.c_inf,
                                   solved.surface_values - solved.c_inf)
    amplitude = np.median(np.abs(solved.surface_values - solved.c_inf))
    assert np.median(np.abs(surface_means(naive) - naive.surface_values)) > 0.1 * amplitude


def test_distant_precipitates_decouple():
    centres, radii = np.array([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0]]), np.array([3.0, 4.0])
    field = physics.solve_field(centres, radii, xi=2.0, c_eq=0.02, ell=2.0, c_inf=0.06)
    assert np.allclose(field.amplitudes, field.surface_values - field.c_inf, rtol=1e-12, atol=0)


def test_no_driving_force_leaves_a_uniform_matrix():
    centres, equal = separated_precipitates(radius=3.0)
    supersaturation = np.exp(2.0 / 3.0)  # every surface value equals c_inf
    field, values = physics.fit_to_matrix_mean(centres, equal, matrix_points(centres, equal), ell=2.0,
                                               supersaturation=supersaturation, matrix_mean=0.04, volume=40.0 ** 3)
    assert np.allclose(field.amplitudes, 0.0, atol=1e-15)
    assert np.allclose(values, 0.04, rtol=1e-12)


def test_screening_length_matches_equal_spheres():
    radius, count, volume = 3.0, 100, 60.0 ** 3
    phi = count * 4 / 3 * np.pi * radius ** 3 / volume
    assert physics.screening_length(np.full(count, radius), volume) == pytest.approx(radius / np.sqrt(3 * phi))


def test_fit_reproduces_the_requested_mean_and_supersaturation():
    centres, radii = separated_precipitates()
    points = matrix_points(centres, radii)
    field, values = physics.fit_to_matrix_mean(centres, radii, points, ell=2.0, supersaturation=3.0,
                                               matrix_mean=0.05, volume=40.0 ** 3)
    assert values.mean() == pytest.approx(0.05, rel=1e-12)
    assert field.supersaturation == pytest.approx(3.0, rel=1e-12)
    assert np.allclose(field(points), values, rtol=0, atol=1e-14)
    assert field.critical_radius == pytest.approx(2.0 / np.log(3.0))


def test_overlapping_precipitates_are_rejected():
    centres, radii = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), np.array([3.0, 3.0])
    with pytest.raises(ValueError, match="overlap"):
        physics.coupling_matrix(centres, radii, 4.0)


def test_a_field_survives_a_round_trip_through_a_dict():
    centres, radii = separated_precipitates(count=5)
    field = physics.solve_field(centres, radii, xi=4.0, c_eq=0.02, ell=2.0, c_inf=0.06)
    points = matrix_points(centres, radii, n=200)
    assert np.array_equal(physics.DiffusionField.from_dict(field.to_dict())(points), field(points))
