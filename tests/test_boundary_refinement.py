"""Fixed physical samples and exact identity/fallback checks for E21 boundaries."""

import inspect
import json

import numpy as np
import pytest

from geom_mesh_net.fields.boundary_refinement import (
    CandidateCatalog, apply_proposals_with_fallback, propose_radii, sample_candidate_profiles,
)


def isolated_catalog():
    return CandidateCatalog([[5, 5, 5], [15, 15, 15]], [2, 2],
                            ("detected-a", "detected-b"), {"split_hash": "observed-fit-only"})


def transition(catalog, radii):
    def field(points):
        distance = np.linalg.norm(points[:, None] - catalog.centres[None], axis=-1)
        return .05 + .8 / (1 + np.exp(np.min((distance - radii) / .08, axis=1)))
    return field


def test_shared_sampling_identical_fields_and_stable_catalogue():
    catalog = isolated_catalog()
    samples = sample_candidate_profiles(catalog, box=20)
    calls = []
    def instrumented(points):
        calls.append(points.copy())
        return transition(catalog, np.array([2.16, 1.84]))(points)
    gaussian = propose_radii(catalog, samples, instrumented)
    fourier = propose_radii(catalog, samples, instrumented)
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(gaussian.radii, fourier.radii)
    np.testing.assert_allclose(gaussian.radii, [2.16, 1.84], atol=.04)
    assert gaussian.accepted.all()
    result = apply_proposals_with_fallback(catalog, gaussian)
    assert result.metadata["candidate_ids"] == ["detected-a", "detected-b"]
    np.testing.assert_array_equal(result.centres, catalog.centres)
    np.testing.assert_array_equal(catalog.radii, [2, 2])
    with pytest.raises(ValueError):
        catalog.radii[0] = 10
    json.dumps(result.to_dict(), allow_nan=False)


def test_every_neural_fallback_equals_gaussian_including_its_own_fallback():
    catalog = isolated_catalog()
    samples = sample_candidate_profiles(catalog, box=20)
    def only_first(points):
        return .05 + .8 / (1 + np.exp((np.linalg.norm(points - catalog.centres[0], axis=1) - 2.2) / .08))
    proposal_c = propose_radii(catalog, samples, only_first)
    assert proposal_c.accepted.tolist() == [True, False]
    arm_c = apply_proposals_with_fallback(catalog, proposal_c)
    flat = propose_radii(catalog, samples, lambda p: np.full(len(p), .2))
    assert not flat.accepted.any()
    arm_d = apply_proposals_with_fallback(catalog, flat, fallback=arm_c)
    np.testing.assert_array_equal(arm_d.radii, arm_c.radii)
    np.testing.assert_array_equal(arm_d.centres, arm_c.centres)
    assert arm_d.metadata["candidate_ids"] == arm_c.metadata["candidate_ids"]
    assert len(arm_d.radii) == 2
    assert arm_d.radii[1] == 2


def test_mask_is_common_and_excludes_box_and_neighbour_owned_samples():
    catalog = CandidateCatalog([[.1, 5, 5], [2.5, 5, 5]], [2, 2])
    samples = sample_candidate_profiles(catalog, box=10)
    assert (~samples.mask[0]).any()
    assert (~samples.mask[1]).any()
    # The same angular set contributes at every radius; changing membership
    # cannot manufacture a transition in an anisotropic but smooth field.
    assert np.all(samples.mask == samples.mask[:, :1, :])
    for k in range(2):
        points = samples.points[k][samples.mask[k]]
        assert np.all((points >= 0) & (points <= 10))
        assert np.all(np.argmin(np.linalg.norm(points[:, None] - catalog.centres, axis=-1), axis=1) == k)
    proposal = propose_radii(catalog, samples, transition(catalog, [2, 2]))
    assert "insufficient_angular_coverage" in proposal.diagnostics[0]["reasons"]


def test_competing_transitions_and_search_bound_are_rejected():
    catalog = CandidateCatalog([[5, 5, 5]], [2])
    samples = sample_candidate_profiles(catalog, box=10, n_radial=81)
    def multiple(points):
        r = np.linalg.norm(points - catalog.centres[0], axis=1)
        return .05 + .4 / (1 + np.exp((r - 1.65) / .025)) + .4 / (1 + np.exp((r - 2.35) / .025))
    proposal = propose_radii(catalog, samples, multiple)
    assert not proposal.accepted[0]
    assert "competing_transitions" in proposal.diagnostics[0]["reasons"]
    bound = propose_radii(catalog, samples, transition(catalog, [1.2]))
    assert not bound.accepted[0]
    assert "search_bound_contact" in bound.diagnostics[0]["reasons"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_probabilities_fall_back_without_changing_candidate_count(bad):
    catalog = isolated_catalog()
    samples = sample_candidate_profiles(catalog, box=20)
    result = propose_radii(catalog, samples, lambda p: np.full(len(p), bad))
    assert not result.accepted.any()
    assert all("invalid_field_probability" in d["reasons"] for d in result.diagnostics)
    geometry = apply_proposals_with_fallback(catalog, result)
    np.testing.assert_array_equal(geometry.radii, catalog.radii)
    json.dumps(result.to_dict(), allow_nan=False)


def test_other_catalogue_or_modified_fallback_is_refused():
    catalog = isolated_catalog()
    samples = sample_candidate_profiles(catalog, box=20)
    other = CandidateCatalog(catalog.centres, catalog.radii, ("changed-a", "changed-b"))
    with pytest.raises(ValueError, match="different candidate"):
        propose_radii(other, samples, lambda p: p[:, 0])
    proposal = propose_radii(catalog, samples, transition(catalog, [2, 2]))
    fallback = catalog.geometry()
    fallback.centres[0, 0] += .1
    with pytest.raises(ValueError, match="preserve"):
        apply_proposals_with_fallback(catalog, proposal, fallback=fallback)


def test_empty_catalogue_and_no_truth_or_held_labels_enter_interfaces():
    catalog = CandidateCatalog([], [])
    samples = sample_candidate_profiles(catalog, box=20)
    def unavailable(points):
        raise AssertionError("empty samples must not call field")
    proposal = propose_radii(catalog, samples, unavailable)
    assert len(apply_proposals_with_fallback(catalog, proposal).radii) == 0
    for function in (sample_candidate_profiles, propose_radii, apply_proposals_with_fallback):
        names = set(inspect.signature(function).parameters)
        assert not names.intersection({"truth", "held_y", "held_back_labels", "scoring", "guest"})


def test_quality_settings_and_field_shape_are_validated():
    catalog = isolated_catalog()
    samples = sample_candidate_profiles(catalog, box=20)
    with pytest.raises(ValueError, match="one probability"):
        propose_radii(catalog, samples, lambda p: np.zeros((len(p), 1)))
    with pytest.raises(ValueError, match="unknown"):
        propose_radii(catalog, samples, lambda p: p[:, 0], settings={"truth_threshold": .1})
