"""Endpoint attribution must distinguish a verified cusp from ordinary curvature."""

import json

import numpy as np
import pytest
import torch

from geom_mesh_net.fields import constrained_joint_fit as cj
from geom_mesh_net.fields import solver_diagnostics as sd


@pytest.fixture(autouse=True)
def limited_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    yield
    torch.set_num_threads(before)


def fixture(*, levels=(.9, .7, .1), centre=(5., 5., 5.), radius=2.00001):
    model = cj.ConstrainedDiffusionModel([centre], [radius], box=10., ell=0.,
                                       c_eq=.1, c_inf=.1, knots=2, levels=levels,
                                       candidate_ids=["sphere"])
    points = np.asarray(centre)[None, :] + np.array([[1., 0., 0.], [1.8, 0., 0.]])
    return model, points, np.array([1., .06])


def test_verified_cusp_has_actual_brackets_matching_independent_likelihood_differences():
    model, x, y = fixture()
    original = model.vector.clone()
    report = sd.diagnose_radius_knots(model, x, y, top_n=1, max_crossings=1,
                                      sample_indices=[91, 47])
    crossing = report["particles"][0]["crossings"][0]
    assert crossing["origins"] == [{"fitting_position": 0, "sample_index": 91,
                                    "knot_index": 1, "u": .5}]
    assert crossing["near_endpoint"] and crossing["radius"] == 2.
    for mode in ("fresh", "frozen"):
        evidence = crossing["assignment_modes"][mode]
        assert evidence["verified"] and evidence["resolved_jump"]
        assert evidence["nearby_coordinate_cusp"] and evidence["coordinate_cusp_minimum"]
        losses = []
        for point in evidence["points"]:
            independent = cj.restore_constrained_state(model.export_state())
            vector = independent.vector.numpy().copy()
            vector[3] = point["scaled_radius"]
            independent.set_vector(vector)
            raw = independent.raw_predict(x)
            losses.append(float(-(y * np.log(raw) + (1-y) * np.log1p(-raw)).mean()))
            assert point["valid"] and point["guard"]["likelihood_evaluated"]
            assert point["guard"]["geometry_feasible"] and point["guard"]["probability_valid"]
            assert point["mean_nll"] == pytest.approx(losses[-1], abs=1e-14)
        width = crossing["actual_offset_scaled"] / 2
        assert evidence["left_finite_difference"] == pytest.approx((losses[1]-losses[0])/width, abs=1e-9)
        assert evidence["right_finite_difference"] == pytest.approx((losses[3]-losses[2])/width, abs=1e-9)
    np.testing.assert_array_equal(model.vector.numpy(), original.numpy())
    assert report["joint_convergence_assessed"] is False
    assert report["diagnostic_evaluations"] == {
        "outside_fitting_budget": True, "total": 5, "likelihood_evaluated": 5, "probe_cache_hits": 4}
    json.dumps(report, allow_nan=False)


def test_equal_adjacent_slopes_do_not_falsely_attribute_a_smooth_knot():
    model, x, y = fixture(levels=(.9, .5, .1))
    report = sd.diagnose_radius_knots(model, x, y, top_n=1, max_crossings=1)
    evidence = report["particles"][0]["crossings"][0]["assignment_modes"]["fresh"]
    assert evidence["verified"]
    assert not evidence["resolved_jump"] and not evidence["coordinate_cusp_minimum"]


def test_distant_knot_is_not_attributed_to_the_endpoint():
    model, x, y = fixture(radius=2.1)
    report = sd.diagnose_radius_knots(model, x, y, top_n=1, max_crossings=1)
    crossing = report["particles"][0]["crossings"][0]
    assert not crossing["near_endpoint"]
    assert not crossing["assignment_modes"]["fresh"]["nearby_coordinate_cusp"]


def test_nearest_crossings_exclude_nonfinite_and_zero_values_and_keep_tied_samples():
    distances = np.array([[np.inf], [np.nan], [0.], [1.], [1.]])
    result = sd._radius_crossings(distances, np.zeros(5, dtype=int), 0, 2., 2,
                                 np.array([10, 20, 30, 40, 50]), 3)
    assert [crossing["radius"] for crossing in result] == [2., 1.]
    assert [origin["sample_index"] for origin in result[0]["origins"]] == [40, 50]
    assert sd._radius_crossings(distances, np.ones(5, dtype=int), 0, 2., 2, np.arange(5), 2) == []


def test_sample_indices_reproduce_crossings_after_input_permutation():
    model, x, y = fixture()
    original = sd.diagnose_radius_knots(model, x, y, top_n=1, max_crossings=1, sample_indices=[91, 47])
    permuted = sd.diagnose_radius_knots(model, x[::-1].copy(), y[::-1].copy(), top_n=1,
                                       max_crossings=1, sample_indices=[47, 91])
    a = original["particles"][0]["crossings"][0]
    b = permuted["particles"][0]["crossings"][0]
    assert a["origins"][0]["sample_index"] == b["origins"][0]["sample_index"] == 91
    assert a["origins"][0]["fitting_position"] == 0
    assert b["origins"][0]["fitting_position"] == 1
    assert a["radius"] == b["radius"]
    assert a["assignment_modes"]["fresh"]["left_derivative"] == pytest.approx(
        b["assignment_modes"]["fresh"]["left_derivative"], abs=1e-12)
    assert original["fitting_identity"] != permuted["fitting_identity"]


def test_bracket_reduces_to_avoid_adjacent_radius_crossing():
    model, _, _ = fixture()
    x = np.array([[6., 5., 5.], [6.000001, 5., 5.]])
    report = sd.diagnose_radius_knots(model, x, [1., 0.], top_n=1, max_crossings=2)
    crossings = report["particles"][0]["crossings"]
    assert len(crossings) == 2
    for crossing in crossings:
        offset = crossing["actual_offset_scaled"] * cj.GEOMETRY_SCALE
        assert offset < report["settings"]["side_offset_scaled"] * cj.GEOMETRY_SCALE
        assert offset == pytest.approx(crossing["nearest_other_crossing_distance"] / 4)
        assert all(abs(point["radius"] - crossing["radius"]) <= offset * (1 + 1e-8)
                   for point in crossing["assignment_modes"]["fresh"]["points"])


def test_guarded_invalid_side_is_recorded_and_never_confirms_a_cusp():
    model, x, y = fixture(centre=(2., 5., 5.), radius=1.99999)
    report = sd.diagnose_radius_knots(model, x, y, top_n=1, max_crossings=1)
    evidence = report["particles"][0]["crossings"][0]["assignment_modes"]["fresh"]
    assert not evidence["verified"] and not evidence["coordinate_cusp_minimum"]
    assert all(point["valid"] for point in evidence["points"][:2])
    for point in evidence["points"][2:]:
        assert not point["valid"] and point["mean_nll"] is None
        assert not point["guard"]["likelihood_evaluated"]
        assert not point["guard"]["geometry_feasible"]


def test_invalid_raw_probabilities_are_refused_before_diagnostic_likelihood():
    model = cj.ConstrainedDiffusionModel([[5., 5., 5.]], [2.], box=10., ell=3., c_eq=.9)
    with pytest.raises(cj.DomainError, match="raw probability"):
        sd.diagnose_radius_knots(model, [[7., 5., 5.]], [1.], top_n=1)


def test_exported_and_legacy_states_reconstruct_the_same_model():
    model, x, _ = fixture()
    exported = sd.model_from_fit_state(model.export_state())
    np.testing.assert_array_equal(exported.vector.numpy(), model.vector.numpy())
    legacy = dict(centres=model.centres().numpy().tolist(), radii=model.radii().numpy().tolist(),
                  constants=model.constants(), shape=model.shape())
    restored = sd.model_from_fit_state(legacy, configuration={"box": 10.}, candidate_ids=["sphere"])
    np.testing.assert_allclose(restored.raw_predict(x), model.raw_predict(x), atol=1e-14)
    assert restored.candidate_ids == ["sphere"]
    with pytest.raises(ValueError, match="original configuration"):
        sd.model_from_fit_state(legacy)


@pytest.mark.parametrize("options", [{"sample_indices": [1, 1]}, {"side_offset": 0},
                                     {"top_n": 1.5}, {"nearby_radius_window": -1}])
def test_invalid_diagnostic_configuration_is_rejected(options):
    model, x, y = fixture()
    with pytest.raises(ValueError):
        sd.diagnose_radius_knots(model, x, y, **options)
