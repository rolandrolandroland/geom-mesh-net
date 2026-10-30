"""The reconstruction benchmark: splits, strata, efficiencies and thinning masks.

Every stage of ``experiments/reconstruction/ROADMAP.md`` reads patterns and observations
through this module, so every method sees identical observed atoms and is scored
on identical removed atoms (ROADMAP section 5). The values below are the frozen
protocol. ``experiments/reconstruction/freeze_benchmark.py`` writes them, with checksums, to
``experiments/reconstruction/benchmark/``; changing any of them invalidates every stored
result.
"""

import hashlib
from pathlib import Path

import numpy as np

from geom_mesh_net import paths

DATA_DIR = paths.RANDOM_CENTRES_DIR   # the benchmark: random cluster centres
LATTICE_DATA_DIR = paths.DATA_DIR     # lattice centres, for Stage 3's comparison only
THETA_PATH = paths.THETA_PATH
BENCHMARK_DIR = paths.RECONSTRUCTION_DIR / "benchmark"
ORACLE_DIR = paths.RECONSTRUCTION_DIR / "results" / "oracle"

SPLITS = {
    "development": (0, 100),
    "training": (100, 800),
    "validation": (800, 900),
    "test": (900, 1000),
}
EFFICIENCIES = (0.1, 0.37, 0.8)
CR_BANDS = ((3.0, 6.0), (6.0, 10.0), (10.0, 15.0))
RHO_C_BANDS = ((0.2, 0.5), (0.5, 1.0))
THINNING_ENTROPY = 20260915  # fixed: changing it changes every mask
GUEST_MARKS = (2, 3)

# Headroom cell (ROADMAP section 5.5)
HEADROOM_MIN_OPEN = 0.15
HEADROOM_MIN_NATS = 0.01


def split_indices(name):
    lo, hi = SPLITS[name]
    return list(range(lo, hi))


def split_of(index):
    for name, (lo, hi) in SPLITS.items():
        if lo <= index < hi:
            return name
    raise ValueError(f"pattern index {index} is outside every split")


def band(value, bands):
    """Index of the half-open band containing value; the last band includes its upper edge."""
    for i, (lo, hi) in enumerate(bands):
        if lo <= value < hi or (i == len(bands) - 1 and value == hi):
            return i
    raise ValueError(f"{value} lies outside {bands}")


def load_theta(path=THETA_PATH):
    """(1000, 4) array of rho_c, rho_b, cr, rb."""
    return np.load(path)


def load_pattern(index, data_dir=DATA_DIR):
    """Atoms, guest labels and cluster geometry of one stored pattern."""
    d = np.load(Path(data_dir) / f"clust_pattern_{index}.npz", allow_pickle=True)
    coords, centres = d["coords"].item(), d["centers"].item()
    return {
        "coords": np.column_stack([coords[a] for a in ("x", "y", "z")]),
        "guest": np.isin(d["labels"], GUEST_MARKS),
        "centres": np.column_stack([np.asarray(centres[a], dtype=float) for a in ("x", "y", "z")]),
        "radii": np.asarray(d["radii"], dtype=float),
        "domain": d["domain"].item(),
    }


def thinning_mask(index, eta, n_atoms, entropy=THINNING_ENTROPY):
    """True for the atoms kept at efficiency ``eta``; the rest are the scored atoms.

    Seeded by (entropy, pattern, efficiency in per mille), so a mask never depends
    on which other patterns or efficiencies were drawn before it. Other datasets pass
    their own ``entropy`` so their masks are independent of the benchmark's.
    """
    per_mille = int(round(eta * 1000))
    if not np.isclose(per_mille / 1000, eta):
        raise ValueError(f"efficiency {eta} is not a whole number of per mille")
    rng = np.random.default_rng(np.random.SeedSequence([entropy, int(index), per_mille]))
    return rng.random(n_atoms) < eta


def mask_checksum(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()


def stage2_subset(theta, per_cell=6):
    """Lowest-index test patterns in each (cr band, rho_c band) cell (ROADMAP Stage 2)."""
    cells = {}
    for index in split_indices("test"):
        rho_c, _, cr, _ = theta[index]
        cells.setdefault((band(cr, CR_BANDS), band(rho_c, RHO_C_BANDS)), []).append(index)
    return {f"cr{c}_rhoc{r}": sorted(indices)[:per_cell] for (c, r), indices in sorted(cells.items())}


def has_headroom(l_const, l_method, l_oracle):
    """Whether a (pattern, efficiency) cell leaves room above ``l_method`` (ROADMAP section 5.5)."""
    gap = l_const - l_oracle
    remaining = l_method - l_oracle
    return gap > 0 and remaining >= HEADROOM_MIN_OPEN * gap and remaining >= HEADROOM_MIN_NATS


def oracle_path(index):
    return ORACLE_DIR / f"oracle_{index}.npz"


def load_oracle(index):
    """Cached per-atom guest probability and sphere count from Stage 0."""
    d = np.load(oracle_path(index))
    return d["p"].astype(np.float64), d["n_spheres"].astype(np.int32)
