"""Does the global "spatial barcode" carry information about the cluster parameters?

The July conditional neural field
(``deprecated_code/tester_scripts/train_network_spatstat_02.py``) appended
``spatial_stats_01.calculate_spatial_barcode`` to the coordinates, using
``LoadData``'s defaults ``barcode_source="thinned"`` and ``barcode_marks="all"``.
In this simulator atom positions are uniform by construction and only the labels
cluster, so a pair-distance histogram over *all* atoms should not vary with the
parameters at all. This measures that, against the same barcode on guest atoms.

Two numbers per barcode bin, over development patterns 0-99:
- Spearman correlation with ``cr`` and ``rho_c``;
- between-pattern standard deviation divided by the resampling standard
  deviation (the same thinned pattern, a second 500-atom subsample). A ratio
  near 1 means patterns differ no more than repeated draws of one pattern.

Recorded in ``reconstruction/ROADMAP.md`` sections 3.2 and 7.

Usage
-----
    PYTHONPATH=. python reconstruction/pilot/check_global_barcode.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from geom_mesh_net.core_functions import clustersim as csim
from geom_mesh_net.core_functions import spatial_stats_01 as spst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--theta", type=Path, default=Path("inference/ground_truth/theta.npy"))
    parser.add_argument("--retention", type=float, default=0.1, help="as in the July run")
    parser.add_argument("--output", type=Path, default=Path("reconstruction/pilot/results/global_barcode.json"))
    args = parser.parse_args()

    theta = np.load(args.theta)
    all_atoms, resampled, guests = [], [], []
    for k in range(100):
        d = np.load(args.data_dir / f"clust_pattern_{k}.npz", allow_pickle=True)
        tc, tl = csim.thin_cluster(d["coords"].item(), args.retention, labels=d["labels"], marks="all")
        all_atoms.append(spst.calculate_spatial_barcode(tc))
        resampled.append(spst.calculate_spatial_barcode(tc))
        g = np.isin(tl, (2, 3))
        guests.append(spst.calculate_spatial_barcode({a: v[g] for a, v in tc.items()}))
    A, R, G = map(np.array, (all_atoms, resampled, guests))
    cr, rho_c = theta[:100, 2], theta[:100, 0]

    def spearman(B, target):
        # The last bin is pinned at 1 by the max normalisation; correlation undefined.
        return [None if np.ptp(B[:, i]) == 0 else float(spearmanr(B[:, i], target)[0])
                for i in range(B.shape[1])]

    between, within = A.std(axis=0), (A - R).std(axis=0) / np.sqrt(2)
    result = dict(
        patterns=100, retention=args.retention,
        all_atoms=dict(spearman_cr=spearman(A, cr), spearman_rho_c=spearman(A, rho_c),
                       between_over_resampling_sd=[None if w == 0 else float(b / w)
                                                   for b, w in zip(between, within)]),
        guest_atoms=dict(spearman_cr=spearman(G, cr), spearman_rho_c=spearman(G, rho_c)),
        all_atom_barcodes_first_five=np.round(A[:5], 3).tolist(),
    )

    fmt = lambda xs: " ".join("  n/a" if x is None else f"{x:+.2f}" for x in xs)
    print("bins                              b0    b1    b2    b3    b4")
    print(f"all atoms   Spearman vs cr      {fmt(result['all_atoms']['spearman_cr'])}")
    print(f"all atoms   Spearman vs rho_c   {fmt(result['all_atoms']['spearman_rho_c'])}")
    print(f"guest atoms Spearman vs cr      {fmt(result['guest_atoms']['spearman_cr'])}")
    print(f"guest atoms Spearman vs rho_c   {fmt(result['guest_atoms']['spearman_rho_c'])}")
    print("all atoms   between / resampling sd  "
          + " ".join("  n/a" if x is None else f"{x:.2f}" for x in result["all_atoms"]["between_over_resampling_sd"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
