"""Small fresh simulations exercise seed separation, truth identity and leakage."""

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from experiments.reconstruction import e21_protocol as protocol
from experiments.reconstruction import e21_simulation as simulation
from experiments.reconstruction import stage2_field
from geom_mesh_net.fields import physics


@pytest.fixture(scope="module")
def small_design():
    design = protocol.load_design()
    design["simulation"].update(box=20.0, atom_count=6000, cr=[2.0, 2.2],
                                rb=[0.15, 0.2], volume_fraction=[0.04, 0.05],
                                min_radius=1.5, min_gap=0.5)
    return design


@pytest.fixture(scope="module")
def generated(small_design):
    spec = protocol.case_spec("development", 0, small_design)
    arrays, metadata = simulation.generate_case(spec, small_design)
    return spec, arrays, metadata


def test_repeated_case_is_exact_and_has_no_dependency_on_historical_files(small_design, generated, monkeypatch, tmp_path):
    spec, arrays, metadata = generated
    monkeypatch.chdir(tmp_path)
    repeated, repeated_metadata = simulation.generate_case(spec, small_design)
    assert metadata == repeated_metadata
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    assert not list(tmp_path.iterdir())


def test_full_spheres_and_fixed_count_labels_match_declared_truth(small_design, generated):
    _, arrays, metadata = generated
    centres, radii = arrays["centres"], arrays["radii"]
    assert np.all(centres - radii[:, None] >= 0)
    assert np.all(centres + radii[:, None] <= small_design["simulation"]["box"])
    gaps = cdist(centres, centres) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min() >= small_design["simulation"]["min_gap"]
    for index, counts in enumerate(metadata["interior_counts"]):
        inside = arrays["owner"] == index
        assert inside.sum() == counts["atoms"]
        assert arrays["y"][inside].sum() == counts["guests"]
        assert counts["guests"] == round(counts["atoms"] * metadata["parameters"]["rho_c"])
    field = physics.DiffusionField.from_dict(metadata["field"])
    q = field(arrays["x"][arrays["owner"] < 0])
    assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
    assert np.isclose(q.mean(), metadata["parameters"]["matrix_mean"])
    np.testing.assert_allclose([q.min(), q.max()], metadata["matrix_probability_range"])


def test_labels_seed_changes_only_marks_not_geometry_positions_or_outer_split(small_design, generated):
    spec, arrays, _ = generated
    alternative = deepcopy(spec)
    alternative["seeds"]["labels"] += 1
    changed, metadata = simulation.generate_case(alternative, small_design)
    assert np.any(changed["y"] != arrays["y"])
    for key in arrays.keys() - {"y"}:
        np.testing.assert_array_equal(changed[key], arrays[key])
    assert metadata["case"]["seeds"]["labels"] == alternative["seeds"]["labels"]


def test_thinning_seed_changes_observation_but_not_realisation(small_design, generated):
    spec, arrays, _ = generated
    alternative = deepcopy(spec)
    alternative["seeds"]["thinning"] += 1
    changed, _ = simulation.generate_case(alternative, small_design)
    for key in ("x", "y", "centres", "radii", "owner"):
        np.testing.assert_array_equal(changed[key], arrays[key])
    assert np.any(changed["removed"] != arrays["removed"])


def test_observation_splits_partition_every_atom_and_counts_are_exact(generated):
    _, arrays, metadata = generated
    indicators = np.stack([arrays[key] for key in ("fitting", "heldback", "removed")])
    assert indicators.dtype == bool
    np.testing.assert_array_equal(indicators.sum(axis=0), np.ones(len(arrays["x"])))
    for name in ("fitting", "heldback", "removed"):
        assert int(arrays[name].sum()) == metadata["counts"][name]
    assert metadata["counts"]["all"] == len(arrays["x"])


def test_serialized_case_is_write_once_and_round_trips(small_design, generated, tmp_path):
    spec, arrays, metadata = generated
    folder = tmp_path / "case"
    simulation.save_case(folder, arrays, metadata)
    loaded, truth = simulation.load_case(folder, spec, small_design)
    assert truth == metadata
    for name in arrays:
        np.testing.assert_array_equal(loaded[name], arrays[name])
    with pytest.raises(FileExistsError):
        simulation.save_case(folder, arrays, metadata)


@pytest.mark.parametrize("change", ["physical_metadata", "array", "split_fraction", "efficiency", "spec"])
def test_changed_truth_data_or_observation_design_refuses_cache(small_design, generated, tmp_path, change):
    spec, arrays, metadata = generated
    folder = tmp_path / "case"
    simulation.save_case(folder, arrays, metadata)
    settings = deepcopy(small_design)
    requested_spec = deepcopy(spec)
    if change == "physical_metadata":
        altered = deepcopy(metadata)
        altered["parameters"]["ell"] *= 1.2
        (folder / "metadata.json").write_text(json.dumps(altered))
    elif change == "array":
        altered = deepcopy(arrays)
        altered["y"][0] = 1 - altered["y"][0]
        with (folder / "data.npz").open("wb") as stream:
            np.savez_compressed(stream, **altered)
    elif change == "split_fraction":
        settings["held_back_fraction"] = 0.3
    elif change == "efficiency":
        settings["efficiency"] = 0.5
    elif change == "spec":
        requested_spec["seeds"]["positions"] += 1
    with pytest.raises(RuntimeError):
        simulation.load_case(folder, requested_spec, settings)


def test_neural_selection_cannot_see_heldback_removed_labels_or_truth(small_design, generated):
    spec, arrays, _ = generated
    settings = stage2_field.load_design()
    original, field_design = simulation.neural_fitting_data(arrays, spec, small_design, settings)
    altered = deepcopy(arrays)
    altered["y"][~altered["fitting"]] = 1 - altered["y"][~altered["fitting"]]
    altered["centres"] += 100
    altered["radii"] *= 10
    alternate, _ = simulation.neural_fitting_data(altered, spec, small_design, settings)
    for name in ("fit_x", "fit_y", "validation_x", "validation_y", "observed_x", "observed_y"):
        np.testing.assert_array_equal(getattr(original, name), getattr(alternate, name))
    assert original.split_hash == alternate.split_hash
    assert len(original.fit_y) + len(original.validation_y) == arrays["fitting"].sum()
    assert len(original.observed_y) == arrays["fitting"].sum()
    assert field_design["split"]["entropy"] == spec["seeds"]["network"]
    assert field_design["data"]["box_length"] == small_design["simulation"]["box"]
    assert settings["data"]["box_length"] == 60.0
