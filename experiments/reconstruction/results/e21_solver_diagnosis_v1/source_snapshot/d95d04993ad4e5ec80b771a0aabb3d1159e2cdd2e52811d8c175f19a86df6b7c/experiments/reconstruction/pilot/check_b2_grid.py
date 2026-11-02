"""Is B2's grid wide enough, or does cross-validation keep choosing its edge?

B2 smooths with a bandwidth proportional to the distance to the k-th nearest observed guest,
c * d_k(x). Stage 1 searched k in (4, 8, 16, 32, 64) and c in (0.25, 0.35, 0.5, 0.7, 1.0), and on
the 300 test cells 94 chose the corner (k = 64, c = 0.25) and 178 chose c = 0.25. A grid whose
edge is chosen that often may be holding B2 below what it can do, which would put Gate 2's bar
("beat the better of B1 and B2") too low (``experiments/reconstruction/ROADMAP.md``, Stage 1,
"Open before Stage 2").

This pilot widens the grid on **development** patterns and reports how often the choice still
lands on an edge, and what the widening is worth in cross-validated log loss. The Stage 1 numbers
are not revised: the grid fixed here is for Stage 2 onward.

Usage
-----
    python -m experiments.reconstruction.pilot.check_b2_grid
    python -m experiments.reconstruction.pilot.check_b2_grid --patterns 0,1,2,3 --k 4,16,64,256
"""

import argparse
import json
import time
from pathlib import Path

from experiments.reconstruction import benchmark as bm
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb

PATTERNS = tuple(range(8))
EFFICIENCIES = (0.37, 0.1)
WIDE_K = fb.ADAPTIVE_K_WIDE     # what this pilot settled on; the library keeps both grids
WIDE_C = fb.ADAPTIVE_C_WIDE


def edge_of(k, c, grid_k, grid_c):
    """Which edges of the grid this choice sits on."""
    edges = []
    if k == min(grid_k):
        edges.append("k low")
    if k == max(grid_k):
        edges.append("k high")
    if c == min(grid_c):
        edges.append("c low")
    if c == max(grid_c):
        edges.append("c high")
    return edges


def analyse(index, eta, grid_k, grid_c, folds, seed):
    pattern = bm.load_pattern(index)
    x, guest = pattern["coords"], pattern["guest"]
    mask = bm.thinning_mask(index, eta, len(x))
    started = time.perf_counter()
    wide = fb.fit_baselines(x[mask], guest[mask], folds=folds, seed=seed,
                            adaptive_k=grid_k, adaptive_c=grid_c)
    narrow_losses = {key: value for key, value in wide.cv_loss_adaptive.items()
                     if int(key.split("_")[0][1:]) in fb.ADAPTIVE_K
                     and float(key.split("_c")[1]) in fb.ADAPTIVE_C}
    narrow_key = min(narrow_losses, key=narrow_losses.get)
    narrow_k = int(narrow_key.split("_")[0][1:])
    narrow_c = float(narrow_key.split("_c")[1])
    best = wide.cv_loss_adaptive[f"k{wide.adaptive_k}_c{wide.adaptive_c}"]
    return {
        "pattern": index, "efficiency": eta,
        "wide": {"k": wide.adaptive_k, "c": wide.adaptive_c, "cv_loss": best,
                 "edges": edge_of(wide.adaptive_k, wide.adaptive_c, grid_k, grid_c)},
        "stage1_grid": {"k": narrow_k, "c": narrow_c, "cv_loss": narrow_losses[narrow_key],
                        "edges": edge_of(narrow_k, narrow_c, fb.ADAPTIVE_K, fb.ADAPTIVE_C)},
        "gain_nats": narrow_losses[narrow_key] - best,
        "b1": {"bandwidth": wide.bandwidth, "cv_loss": min(wide.cv_loss_fixed.values())},
        "b2_beats_b1_nats": min(wide.cv_loss_fixed.values()) - best,
        "cv_loss_adaptive": wide.cv_loss_adaptive,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", default=",".join(map(str, PATTERNS)))
    parser.add_argument("--efficiencies", default=",".join(map(str, EFFICIENCIES)))
    parser.add_argument("--k", default=",".join(map(str, WIDE_K)))
    parser.add_argument("--c", default=",".join(map(str, WIDE_C)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "b2_grid.json")
    args = parser.parse_args()
    patterns = [int(i) for i in args.patterns.split(",")]
    efficiencies = [float(e) for e in args.efficiencies.split(",")]
    grid_k = tuple(int(k) for k in args.k.split(","))
    grid_c = tuple(float(c) for c in args.c.split(","))

    rows, started = [], time.perf_counter()
    for index in patterns:
        for eta in efficiencies:
            row = analyse(index, eta, grid_k, grid_c, args.folds, args.seed)
            rows.append(row)
            wide, narrow = row["wide"], row["stage1_grid"]
            print(f"pattern {index} eta {eta}: Stage 1 grid chose k {narrow['k']:>3} c {narrow['c']:.2f}"
                  f"{' [' + ', '.join(narrow['edges']) + ']' if narrow['edges'] else ''}"
                  f" | widened chose k {wide['k']:>3} c {wide['c']:.2f}"
                  f"{' [' + ', '.join(wide['edges']) + ']' if wide['edges'] else ''}"
                  f" | worth {row['gain_nats']:.5f} nats, and {row['b2_beats_b1_nats']:+.5f} over B1"
                  f" [{row['seconds']} s]", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(
                {"design": {"patterns": patterns, "efficiencies": efficiencies, "k": list(grid_k),
                            "c": list(grid_c), "folds": args.folds, "seed": args.seed,
                            "stage1_k": list(fb.ADAPTIVE_K), "stage1_c": list(fb.ADAPTIVE_C)},
                 "per_cell": rows, "seconds": round(time.perf_counter() - started)}, indent=1) + "\n")
    on_edge = [r for r in rows if r["wide"]["edges"]]
    print(f"\n{len(on_edge)} of {len(rows)} cells still chose an edge of the widened grid")
    for row in on_edge:
        print(f"  pattern {row['pattern']} eta {row['efficiency']}: {', '.join(row['wide']['edges'])}")
    print(f"wrote {args.output} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
