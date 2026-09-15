"""Where the repository's data, experiments and documentation live.

Every script resolves its default inputs and outputs through this module, so the
results no longer depend on the directory a script is run from. The datasets are
large and regenerable, so each location can be overridden with an environment
variable:

| Variable | Default |
| --- | --- |
| ``GEOM_MESH_NET_ROOT`` | the repository containing this package |
| ``GEOM_MESH_NET_DATA`` | ``<root>/data`` (lattice cluster centres) |
| ``GEOM_MESH_NET_DATA_RANDOM_CENTRES`` | ``<root>/data_random_centres`` |
| ``GEOM_MESH_NET_DATA_SHARED_UPP`` | ``<root>/data_shared_upp`` |

The package is meant to be installed in editable mode (``pip install -e .``), so
the default root is the directory above this file.
"""

import os
from pathlib import Path


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


REPO_ROOT = _env_path("GEOM_MESH_NET_ROOT", Path(__file__).resolve().parents[1])

# Simulated datasets (gitignored; regenerable).
DATA_DIR = _env_path("GEOM_MESH_NET_DATA", REPO_ROOT / "data")
RANDOM_CENTRES_DIR = _env_path("GEOM_MESH_NET_DATA_RANDOM_CENTRES", REPO_ROOT / "data_random_centres")
SHARED_UPP_DIR = _env_path("GEOM_MESH_NET_DATA_SHARED_UPP", REPO_ROOT / "data_shared_upp")

# Experiment tracks.
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
INFERENCE_DIR = EXPERIMENTS_DIR / "inference"
RECONSTRUCTION_DIR = EXPERIMENTS_DIR / "reconstruction"
NEURAL_FIELD_DIR = EXPERIMENTS_DIR / "neural_field"

# Ground truth for data/, recovered by experiments/inference/recover_ground_truth.py.
GROUND_TRUTH_DIR = INFERENCE_DIR / "ground_truth"
THETA_PATH = GROUND_TRUTH_DIR / "theta.npy"

# Documentation.
DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"


def repo_relative(path):
    """``path`` relative to the repository root when it lies inside it, else unchanged.

    Scripts record their inputs in tracked JSON files. Recording them relative to
    the root keeps machine-specific absolute paths out of the repository.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
