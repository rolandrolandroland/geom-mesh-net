"""Freeze the reconstruction benchmark into reconstruction/benchmark/ (ROADMAP Stage 0).

Writes, all tracked:

- ``benchmark.json``: splits, efficiencies, strata, each pattern's stratum,
  the Stage 2 subset and the thinning entropy;
- ``mask_checksums.json``: SHA-256 of every thinning mask for the development,
  validation and test splits, so any later drift in how masks are drawn is caught;
- ``dataset_checksums.json`` and ``dataset_provenance.json``: copied from
  ``data_random_centres/``, so a regenerated dataset can be told apart from the
  one every result was scored on.

Usage
-----
    PYTHONPATH=. python reconstruction/freeze_benchmark.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from reconstruction import benchmark as bm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=bm.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=bm.BENCHMARK_DIR)
    args = parser.parse_args()

    theta = bm.load_theta()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atom_counts = set()
    for index in range(len(theta)):
        with np.load(args.data_dir / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
            atom_counts.add(len(d["labels"]))
    if len(atom_counts) != 1:
        raise SystemExit(f"patterns differ in atom count: {sorted(atom_counts)}")
    n_atoms = atom_counts.pop()

    strata = {
        str(i): {"cr_band": bm.band(theta[i, 2], bm.CR_BANDS), "rho_c_band": bm.band(theta[i, 0], bm.RHO_C_BANDS),
                 "split": bm.split_of(i)}
        for i in range(len(theta))
    }
    spec = {
        "data_dir": str(args.data_dir),
        "lattice_data_dir": str(bm.LATTICE_DATA_DIR),
        "theta": str(bm.THETA_PATH),
        "atoms_per_pattern": n_atoms,
        "splits": bm.SPLITS,
        "efficiencies": bm.EFFICIENCIES,
        "cr_bands": bm.CR_BANDS,
        "rho_c_bands": bm.RHO_C_BANDS,
        "thinning": "np.random.default_rng(SeedSequence([entropy, pattern, round(1000 * eta)])).random(n) < eta",
        "thinning_entropy": bm.THINNING_ENTROPY,
        "headroom": {"min_open_fraction": bm.HEADROOM_MIN_OPEN, "min_nats_per_atom": bm.HEADROOM_MIN_NATS},
        "stage2_subset": bm.stage2_subset(theta),
        "strata": strata,
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(spec, indent=1) + "\n")

    checksums = {}
    for split in ("development", "validation", "test"):
        for index in bm.split_indices(split):
            for eta in bm.EFFICIENCIES:
                checksums[f"{index}:{eta}"] = bm.mask_checksum(bm.thinning_mask(index, eta, n_atoms))
    (args.output_dir / "mask_checksums.json").write_text(json.dumps(checksums, indent=1) + "\n")

    for name in ("checksums.json", "provenance.json"):
        source = args.data_dir / name
        if source.exists():
            (args.output_dir / f"dataset_{name}").write_text(source.read_text())

    subset = spec["stage2_subset"]
    print(f"{n_atoms} atoms per pattern; {len(checksums)} mask checksums; "
          f"Stage 2 subset: {sum(len(v) for v in subset.values())} patterns "
          f"({', '.join(f'{k}: {len(v)}' for k, v in subset.items())})")


if __name__ == "__main__":
    main()
