"""Could any held-out comparison tell each misspecified matrix from the diffusion law?

Gate 5.2(iii) asks whether the Stage 5.2 method rejects its physics when the matrix does not
obey it (``experiments/reconstruction/ROADMAP.md``). A control is a fair test only if the
difference is detectable in principle. For development patterns, and each matrix field of
``generate_misspecified_patterns.py`` beside the diffusion field itself, this pilot fits the
screened diffusion family, with the true precipitate geometry and its constants free, to
the field's own probabilities at every matrix atom. That is the best the physics can do
with unlimited data. What remains is measured in nats:

- **signal**: the field's summed divergence from a constant matrix, over all matrix atoms;
- **gap**: its summed divergence from the best physics fit, over all matrix atoms.

The expected advantage of the true field on a held-out set scales as the gap times the
share of atoms held out. At efficiency 0.37 with 20% of observed matrix atoms held out,
about 0.074 of the gap; at 0.1, 0.02. A gap of a few nats cannot be detected by any rule.

Usage
-----
    python -m experiments.reconstruction.pilot.check_misspecification
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.reconstruction import generate_misspecified_patterns as gm
from geom_mesh_net import paths
from geom_mesh_net.fields import physics
from geom_mesh_net.neural import implicit

PATTERNS = tuple(range(8))
KINDS = ("diffusion",) + gm.KINDS
STARTS = ((None, None), (1.0, 3.0), (6.0, 12.0), (0.2, 6.0))   # (ell, xi); None: the true value
HELD_OUT_SHARE = {"0.37": 0.37 * 0.2, "0.1": 0.1 * 0.2}


def divergence(p, q):
    p, q = np.clip(p, 1e-12, 1 - 1e-12), np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def best_physics_fit(field, points, p):
    x = torch.as_tensor(points, dtype=torch.float64)
    target = torch.as_tensor(np.clip(p, 0.0, 1.0), dtype=torch.float64)
    best = None
    for ell0, xi0 in STARTS:
        model = implicit.ParametricDiffusionField(field.centres, field.radii, c_eq=field.c_eq,
                                                  ell=field.ell if ell0 is None else ell0,
                                                  xi=field.xi if xi0 is None else xi0, c_inf=field.c_inf)
        optimiser = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            loss = F.binary_cross_entropy(model(x).clamp(1e-9, 1 - 1e-9), target)
            loss.backward()
            return loss

        optimiser.step(closure)
        with torch.no_grad():
            gap = float(divergence(p, model(x).numpy()).sum())
        if best is None or gap < best[0]:
            best = (gap, model.constants())
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "misspecification.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows, started = [], time.perf_counter()
    for index in PATTERNS:
        with np.load(paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
            coords, labels = d["coords"].item(), d["labels"]
            truth = d["physics"].item()
        points = np.column_stack([coords[a] for a in "xyz"])[np.isin(labels, (0, 3))]
        field = physics.DiffusionField.from_dict(truth)
        row = {"pattern": index}
        for kind in KINDS:
            p = field(points) if kind == "diffusion" else gm.load(kind, index)[1]
            gap, constants = best_physics_fit(field, points, p)
            signal = float(divergence(p, np.full(len(p), p.mean())).sum())
            row[kind] = {"signal_nats": signal, "gap_nats": gap, "held_out_gap_nats": {k: v * gap for k, v in HELD_OUT_SHARE.items()},
                         "best_fit_constants": constants}
            print(f"pattern {index} {kind:17s} signal {signal:6.1f} nats, gap {gap:6.2f} nats "
                  f"(held-out at 0.37: {HELD_OUT_SHARE['0.37'] * gap:.2f})", {k: round(v, 3) for k, v in constants.items()},
                  flush=True)
        rows.append(row)
    summary = {kind: {"gap_nats": {"median": float(np.median([r[kind]["gap_nats"] for r in rows])),
                                   "min": float(min(r[kind]["gap_nats"] for r in rows)),
                                   "max": float(max(r[kind]["gap_nats"] for r in rows))},
                      "signal_nats_median": float(np.median([r[kind]["signal_nats"] for r in rows]))} for kind in KINDS}
    report = {"design": {"patterns": list(PATTERNS), "kinds": list(KINDS), "starts": STARTS, "held_out_share": HELD_OUT_SHARE,
                         "geometry": "true centres and radii"},
              "summary": summary, "per_pattern": rows, "seconds": round(time.perf_counter() - started)}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.output} in {report['seconds']} s")


if __name__ == "__main__":
    main()
