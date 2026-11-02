"""Where do cluster centres sit, and how many clusters did each pattern receive?

Two findings of Stage 0 (``experiments/reconstruction/ROADMAP.md`` section 3.2), made
reproducible:

1. In ``data/``, cluster centres lie on a cubic lattice centred on the box. In
   ``data_random_centres/`` they do not.
2. For an odd number of lattice points per side, the lattice's outermost planes
   fall exactly on the edges of the window clustersim keeps. Floating point
   decides whether they survive, and some ``data/`` patterns lost that shell.
   The random-centre dataset draws the same parameters with oversampled random
   centres, so its cluster count is the one each parameter vector intends.

Writes ``experiments/reconstruction/pilot/results/lattice.json``: per-pattern counts, the
lattice test, the lost-shell patterns, and the centres of two example patterns
for plotting.

Usage
-----
    python -m experiments.reconstruction.pilot.check_lattice
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from geom_mesh_net import paths


def load(data_dir, index):
    with np.load(Path(data_dir) / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        centres = d["centers"].item()
        return np.column_stack([np.asarray(centres[a], dtype=float) for a in "xyz"]), np.asarray(d["radii"])


def lattice_status(centres):
    """'lattice' if every axis's distinct coordinates are whole multiples of one spacing,
    offset by half a spacing (or zero) from the box centre; 'untestable' with fewer than 2
    distinct values on some axis; otherwise 'random'."""
    axes = [np.unique(np.round(centres[:, a], 6)) for a in range(3)]
    if any(len(u) < 2 for u in axes):
        return "untestable"
    spacing = min(np.diff(u).min() for u in axes)
    for u in axes:
        offsets = (u - 30.0) / spacing
        if not (np.allclose(offsets, np.round(offsets), atol=1e-4)
                or np.allclose(offsets - 0.5, np.round(offsets - 0.5), atol=1e-4)):
            return "random"
    return "lattice"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lattice-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--random-dir", type=Path, default=paths.RANDOM_CENTRES_DIR)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "lattice.json")
    args = parser.parse_args()

    n_lattice, n_random, status_lattice, status_random = [], [], [], []
    for i in range(1000):
        c_l, r_l = load(args.lattice_dir, i)
        c_r, r_r = load(args.random_dir, i)
        n_lattice.append(len(r_l))
        n_random.append(len(r_r))
        status_lattice.append(lattice_status(c_l))
        status_random.append(lattice_status(c_r))
    n_lattice, n_random = np.array(n_lattice), np.array(n_random)

    lost_shell = []
    for i in range(1000):
        side = math.ceil(n_random[i] ** (1 / 3) - 1e-9)
        shortfall = n_random[i] - n_lattice[i]
        if side % 2 == 1 and n_lattice[i] == (side - 1) ** 3 and shortfall > max(2, 0.03 * n_random[i]):
            lost_shell.append(i)

    pcp_lattice = np.load(args.lattice_dir / "pattern_stats.npy")[:, 0]
    pcp_random = np.load(args.random_dir / "pattern_stats.npy")[:, 0]
    others = np.setdiff1d(np.arange(1000), lost_shell)
    examples = {}
    for i in (10, 99):
        examples[str(i)] = {"lattice": load(args.lattice_dir, i)[0].round(4).tolist(),
                            "random": load(args.random_dir, i)[0].round(4).tolist()}

    count = lambda xs, v: int(sum(x == v for x in xs))
    result = {
        "lattice_dataset": {"lattice": count(status_lattice, "lattice"), "untestable": count(status_lattice, "untestable"),
                            "random": count(status_lattice, "random")},
        "random_dataset": {"lattice": count(status_random, "lattice"), "untestable": count(status_random, "untestable"),
                           "random": count(status_random, "random")},
        "count_difference_lattice_minus_random": {str(k): int(v) for k, v in zip(*np.unique(n_lattice - n_random, return_counts=True))},
        "within_one": int(np.sum(np.abs(n_lattice - n_random) <= 1)),
        "lost_shell": {"indices": lost_shell, "n": len(lost_shell),
                       "by_side": {str(s): int(sum(math.ceil(n_random[i] ** (1 / 3) - 1e-9) == s for i in lost_shell)) for s in (3, 5, 7)},
                       "pcp_lost_shell_mean": float(pcp_lattice[lost_shell].mean()),
                       "pcp_other_mean": float(pcp_lattice[others].mean())},
        "n_clusters_lattice": n_lattice.tolist(),
        "n_clusters_random": n_random.tolist(),
        "pcp_lattice": pcp_lattice.round(5).tolist(),
        "pcp_random": pcp_random.round(5).tolist(),
        "example_centres": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("lattice_dataset", "random_dataset", "within_one")}, indent=1))
    print("lost shell:", {k: v for k, v in result["lost_shell"].items() if k != "indices"}, result["lost_shell"]["indices"])


if __name__ == "__main__":
    main()
