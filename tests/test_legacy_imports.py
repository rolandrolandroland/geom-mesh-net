"""Tests for the deprecated ``geom_mesh_net.core_functions`` names.

Old notebooks and the scripts in ``deprecated_code/`` still import the modules
under the names they had before the package was split into subpackages. Each old
name must resolve to the *same* module object as its new one, not to a copy:
clustersim draws from a module-level generator, and a copy would carry a second
generator whose draws silently differ from the first.
"""

import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

MOVED = {
    "clustersim": "geom_mesh_net.simulation.clustersim",
    "paper_spatial_features": "geom_mesh_net.statistics.paper_spatial_features",
    "spatial_stats_01": "geom_mesh_net.statistics.barcode",
    "paper_feature_experiments": "geom_mesh_net.statistics.feature_cache",
    "field_oracle": "geom_mesh_net.fields.oracle",
    "field_baselines": "geom_mesh_net.fields.baselines",
    "point_cloud_fields": "geom_mesh_net.fields.point_cloud",
    "voxelize_clusters": "geom_mesh_net.fields.density_grid",
    "compare_benchmarks": "geom_mesh_net.viz.benchmarks",
}


def fresh_import(name):
    """Import ``name`` as if for the first time, recording any warnings."""
    sys.modules.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    return module, caught


@pytest.mark.parametrize("old, new", MOVED.items())
def test_old_name_is_the_new_module_and_warns(old, new):
    module, caught = fresh_import(f"geom_mesh_net.core_functions.{old}")
    assert module is importlib.import_module(new)
    assert any(issubclass(w.category, DeprecationWarning) and new in str(w.message) for w in caught)


def test_from_import_form_resolves_to_the_new_module():
    sys.modules.pop("geom_mesh_net.core_functions.clustersim", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from geom_mesh_net.core_functions import clustersim as old
    from geom_mesh_net.simulation import clustersim as new

    assert old is new
    assert old.rng is new.rng


def test_data_loader_still_provides_the_dataset_and_every_model():
    module, _ = fresh_import("geom_mesh_net.core_functions.data_loader")
    from geom_mesh_net.neural import datasets, models

    assert module.LoadData is datasets.LoadData
    assert module.point_cloud_collate is datasets.point_cloud_collate
    for name in (
        "ContinuousNeuralField",
        "ContinuousNeuralField2",
        "ContinuousNeuralFieldspatstat_01",
        "ContinuousNeuralFieldGlobalFeatures",
        "ContinuousNeuralFieldFeatures",
    ):
        assert getattr(module, name) is getattr(models, name)


def test_the_library_never_imports_its_own_deprecated_names():
    """Run in a fresh interpreter, where any deprecated import would be an error."""
    new_modules = sorted(set(MOVED.values()) | {
        "geom_mesh_net.neural",
        "geom_mesh_net.simulation.parameters",
        "geom_mesh_net.statistics.presets",
        "geom_mesh_net.inference.calibration",
        "geom_mesh_net.inference.flow",
        "geom_mesh_net.paths",
    })
    result = subprocess.run(
        [sys.executable, "-W", "error:geom_mesh_net.core_functions:DeprecationWarning",
         "-c", "; ".join(f"import {name}" for name in new_modules)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
