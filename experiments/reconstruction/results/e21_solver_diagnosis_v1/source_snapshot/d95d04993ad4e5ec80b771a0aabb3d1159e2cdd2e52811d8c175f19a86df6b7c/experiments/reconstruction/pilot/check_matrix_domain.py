"""How should Stage 5.2 find precipitates and choose the matrix atoms it fits?

Stage 5.2 fits the matrix field to observed matrix atoms, with precipitates found in the
data (``experiments/reconstruction/ROADMAP.md``, second Stage 5 correction). Both choices are
made from the same labels the fit then uses. For development patterns at efficiencies 0.37
and 0.1, with the B1 field at its cross-validated bandwidth:

- **Detection.** Precipitates are found by Otsu's threshold and by significance above the
  matrix level (``cluster_extraction.segment``). Each is matched to the true precipitates
  (precision, recall, median radius ratio), and overlapping spheres are counted before and
  after ``resolve_overlaps``.
- **Selection bias.** Observed atoms are admitted to the matrix by the conservative domain
  rule, once as a voxel lookup and once with each atom's own label left out
  (``cluster_extraction.matrix_atoms``). Among admitted atoms of the true matrix, the
  guest count is compared with the oracle's expectation as a z-score. A domain chosen from
  the labels without leaving them out selects on them, and the score falls far below zero.
- **Contamination.** How many admitted atoms lie inside true precipitates, and what share of
  the observed matrix is admitted.

Usage
-----
    python -m experiments.reconstruction.pilot.check_matrix_domain
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.reconstruction import benchmark as bm
from experiments.reconstruction import stage5_simulator
from experiments.reconstruction.generate_diffusion_patterns import THINNING_ENTROPY
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.fields import cluster_extraction as ce

PATTERNS = tuple(range(16))
EFFICIENCIES = (0.37, 0.1)
RESOLVE_GAP = 0.5


def admission(admitted, guest, p_star, inside, observed_matrix):
    matrix = admitted[~inside[admitted]]
    y, p = guest[matrix].astype(float), p_star[matrix]
    return {"bias_z": float((y.sum() - p.sum()) / np.sqrt(np.sum(p * (1 - p)))),
            "share_of_observed_matrix": float(len(matrix) / observed_matrix),
            "precipitate_atoms": int(np.sum(inside[admitted])),
            "precipitate_guests": int(np.sum(guest[admitted[inside[admitted]]]))}


def detection(centres, radii, true_centres, true_radii):
    rows, cols = ce.match_precipitates(true_centres, true_radii, centres)
    distance = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distance, np.inf)
    overlapping = int(np.sum(distance < radii[:, None] + radii[None, :]) // 2) if len(radii) else 0
    return {"found": int(len(radii)), "precision": len(rows) / max(len(radii), 1), "recall": len(rows) / len(true_radii),
            "radius_ratio_median": float(np.median(radii[cols] / true_radii[rows])) if len(rows) else None,
            "overlapping_pairs": overlapping}


def analyse(task):
    index, eta = task
    started = time.perf_counter()
    with np.load(paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        coords, labels = d["coords"].item(), d["labels"]
        truth = d["physics"].item()
    x = np.column_stack([coords[a] for a in "xyz"])
    guest = np.isin(labels, (2, 3))
    true_centres, true_radii = np.asarray(truth["centres"]), np.asarray(truth["radii"])
    p_star, inside = stage5_simulator.load_oracle(index)
    observed = np.flatnonzero(bm.thinning_mask(index, eta, len(x), entropy=THINNING_ENTROPY))
    observed_matrix = int(np.sum(~inside[observed]))
    bandwidth = fb.fit_baselines(x[observed], guest[observed], folds=5, seed=index).bandwidth
    guests, atoms = ce.smoothed_counts(x[observed], guest[observed], bandwidth)
    field = ce.guest_fraction(guests, atoms)
    squared = ce.squared_kernel_counts(x[observed], bandwidth)
    row = {"pattern": index, "efficiency": eta, "bandwidth": bandwidth, "true_precipitates": int(len(true_radii)),
           "methods": {}}
    for method in ("otsu", "significance"):
        seg = ce.segment(field, atoms, method=method, squared=squared)
        resolved = ce.resolve_overlaps(seg.centres, seg.radii, gap=RESOLVE_GAP)
        row["methods"][method] = {
            "detection": detection(seg.centres, seg.radii, true_centres, true_radii),
            "detection_resolved": detection(*resolved, true_centres, true_radii),
            "voxel_rule": admission(observed[seg.in_matrix(x[observed])], guest, p_star, inside, observed_matrix),
            "own_label_left_out": admission(observed[ce.matrix_atoms(x[observed], guest[observed], guests, atoms, bandwidth, seg)],
                                            guest, p_star, inside, observed_matrix),
        }
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def summarise(rows):
    summary = {}
    for eta in EFFICIENCIES:
        cells = [r for r in rows if r["efficiency"] == eta]
        summary[str(eta)] = {}
        for method in ("otsu", "significance"):
            m = [r["methods"][method] for r in cells]

            def stats(values):
                values = np.asarray([v for v in values if v is not None], dtype=float)
                return {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}

            summary[str(eta)][method] = {
                "precision": stats([c["detection"]["precision"] for c in m]),
                "recall": stats([c["detection"]["recall"] for c in m]),
                "radius_ratio": stats([c["detection"]["radius_ratio_median"] for c in m]),
                "patterns_with_overlaps": int(sum(c["detection"]["overlapping_pairs"] > 0 for c in m)),
                "voxel_rule_bias_z": stats([c["voxel_rule"]["bias_z"] for c in m]),
                "own_label_left_out_bias_z": {**stats([c["own_label_left_out"]["bias_z"] for c in m]),
                                              "mean": float(np.mean([c["own_label_left_out"]["bias_z"] for c in m])),
                                              "sd": float(np.std([c["own_label_left_out"]["bias_z"] for c in m]))},
                "admitted_share": stats([c["own_label_left_out"]["share_of_observed_matrix"] for c in m]),
                "precipitate_atoms_admitted": stats([c["own_label_left_out"]["precipitate_atoms"] for c in m]),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "matrix_domain.json")
    args = parser.parse_args()
    tasks = [(index, eta) for index in PATTERNS for eta in EFFICIENCIES]
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(analyse, tasks), key=lambda r: (-r["efficiency"], r["pattern"]))
    report = {"design": {"patterns": list(PATTERNS), "efficiencies": list(EFFICIENCIES), "resolve_gap": RESOLVE_GAP,
                         "thinning_entropy": THINNING_ENTROPY,
                         "significance": {"region_z": ce.REGION_Z, "seed_z": ce.SEED_Z, "exclusion_z": ce.EXCLUSION_Z}},
              "summary": summarise(rows), "per_cell": rows, "seconds": round(time.perf_counter() - started)}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {args.output} in {report['seconds']} s")


if __name__ == "__main__":
    main()
