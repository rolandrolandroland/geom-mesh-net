"""Check density_grid.generate_density_grid (formerly voxelize_clusters) against clustersim.

``LoadData(target_source="simulation")`` uses this grid as its ground-truth
field, and the original reconstruction plan would have scored every model
against it. Three measurements, recorded in ``experiments/reconstruction/ROADMAP.md``
section 3.2:

1. Stored patterns, split by region. If the grid is a probability field, then
   among atoms where it says p, a fraction p are guests.
2. One isolated cluster, replayed. The grid rescales the linear weights to mean
   ``rho_c`` and clips them; clustersim calls ``rng.choice(replace=False, p=w)``,
   whose inclusion probabilities are not proportional to the weights.
3. How many stored patterns make ``generate_density_grid`` raise. A cluster
   whose bounding box holds no voxel centre (zero or tiny radius) leaves an
   empty array, and ``assign_clust_probs`` takes its maximum.

Uses development patterns only (indices below 100).

Usage
-----
    python -m experiments.reconstruction.pilot.check_density_grid
    python -m experiments.reconstruction.pilot.check_density_grid --patterns 60 \
        --output experiments/reconstruction/pilot/results/density_grid.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geom_mesh_net import paths
from geom_mesh_net.fields import density_grid as vc

GUEST_MARKS = (2, 3)
RHO_C_COLUMN, RHO_B_COLUMN = 4, 7  # LoadData's defaults into pattern_stats.npy


def load_pattern(data_dir, index):
    d = np.load(Path(data_dir) / f"clust_pattern_{index}.npz", allow_pickle=True)
    c, cen = d["coords"].item(), d["centers"].item()
    X = np.column_stack([c["x"], c["y"], c["z"]])
    C = np.column_stack([cen["x"], cen["y"], cen["z"]])
    return X, np.isin(d["labels"], GUEST_MARKS).astype(np.float64), d["radii"], C, cen


def usable_clusters(radii, C, resolution, extent=60.0):
    """Clusters generate_density_grid can process without raising."""
    axis = np.linspace(resolution / 2, extent - resolution / 2, int(extent / resolution))
    return np.array([all(((axis >= q - r) & (axis <= q + r)).any() for q in centre)
                     for r, centre in zip(radii, C)], dtype=bool)


def region_check(data_dir, n_patterns, resolution):
    stats = np.load(Path(data_dir) / "pattern_stats.npy")
    cols = {k: [] for k in ("y", "p_grid", "p_max", "p_union", "n_in", "rho_b")}
    withheld = 0
    for k in range(n_patterns):
        X, y, radii, C, cen = load_pattern(data_dir, k)
        rho_c, rho_b = stats[k, RHO_C_COLUMN], stats[k, RHO_B_COLUMN]

        # The target exactly as LoadData builds it, looked up at the nearest voxel.
        # Clusters that would make it raise are withheld from the grid only.
        ok = usable_clusters(radii, C, resolution)
        withheld += int((~ok).sum())
        _, _, _, grid = vc.generate_density_grid(
            grid_size=[60, 60, 60], cluster_centers={a: cen[a][ok] for a in "xyz"},
            radii=radii[ok], rho_c=rho_c, rho_b=rho_b, resolution=resolution,
            prob_function="Gaussian_decay", selection="sampled", overlap_prob="highest")
        idx = np.clip((X / resolution).astype(int), 0, grid.shape[0] - 1)

        # The same within-cluster formula at the atoms themselves, without the
        # rho_b floor and with overlaps as a union, as the simulator's labels are.
        n_in = np.zeros(len(X), dtype=np.int16)
        p_max, miss = np.zeros(len(X)), np.ones(len(X))
        tree = cKDTree(X)
        for r, centre in zip(radii, C):
            if r <= 0:
                continue
            ins = np.asarray(tree.query_ball_point(centre, r), dtype=int)
            if ins.size < 2:
                continue
            dist = np.linalg.norm(X[ins] - centre, axis=1)
            p_i = vc.assign_clust_probs(dist=dist, weighted_dist=dist.copy(),
                                        density_grid=np.zeros(ins.size), rho_c=rho_c,
                                        r_max=r, prob_function="Gaussian_decay",
                                        selection="sampled")
            n_in[ins] += 1
            p_max[ins] = np.maximum(p_max[ins], p_i)
            miss[ins] *= 1 - p_i
        for key, value in (("y", y), ("p_grid", grid[idx[:, 0], idx[:, 1], idx[:, 2]]),
                           ("p_max", p_max), ("p_union", np.where(n_in > 0, 1 - miss, rho_b)),
                           ("n_in", n_in), ("rho_b", np.full(len(X), rho_b))):
            cols[key].append(value)
    A = {k: np.concatenate(v) for k, v in cols.items()}

    one = A["n_in"] == 1
    rim = one & (A["p_max"] < A["rho_b"])
    regions = {
        "matrix (outside every sphere)": A["n_in"] == 0,
        "one sphere, within-cluster p < rho_b (rim)": rim,
        "one sphere, within-cluster p >= rho_b": one & ~rim,
        "two or more spheres (overlap)": A["n_in"] >= 2,
    }
    region_rows = [dict(region=name, atoms=int(m.sum()), share=float(m.mean()),
                        grid=float(A["p_grid"][m].mean()), observed=float(A["y"][m].mean()),
                        formula_at_atoms=float(A["p_union"][m].mean()))
                   for name, m in regions.items()]

    bins = np.array([0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001])
    b = np.digitize(A["p_grid"], bins) - 1
    reliability = []
    for i in range(len(bins) - 1):
        m = b == i
        if m.any():
            obs = A["y"][m].mean()
            reliability.append(dict(lo=float(bins[i]), hi=float(bins[i + 1]), atoms=int(m.sum()),
                                    grid=float(A["p_grid"][m].mean()), observed=float(obs),
                                    se=float(np.sqrt(max(obs * (1 - obs), 1e-12) / m.sum()))))
    return dict(patterns=n_patterns, atoms=int(len(A["y"])), resolution=resolution,
                clusters_withheld_from_grid=withheld, regions=region_rows, reliability=reliability)


def single_cluster_profiles(radius=8.0, replays=400, seed=0):
    rng = np.random.default_rng(seed)
    N = rng.poisson(4 / 3 * np.pi * radius ** 3)  # density 1, as in data/
    pts = rng.uniform(-radius, radius, size=(N * 3, 3))
    pts = pts[np.linalg.norm(pts, axis=1) <= radius][:N]
    dist = np.linalg.norm(pts, axis=1)
    u, w = dist / radius, 1 - dist / dist.max()
    u_edges = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    rows = []
    for rho_c in (0.3, 0.6, 0.9):
        n_sel = int(round(len(pts) * rho_c))
        hits = np.zeros(len(pts))
        for _ in range(replays):
            hits[rng.choice(len(pts), size=n_sel, replace=False, p=w / w.sum())] += 1
        incl = hits / replays
        approx = vc.assign_clust_probs(dist=dist, weighted_dist=dist.copy(),
                                       density_grid=np.zeros(len(pts)), rho_c=rho_c,
                                       r_max=radius + 1e-9)
        cells = [dict(u_lo=a, u_hi=b_, grid=float(approx[(u >= a) & (u < b_)].mean()),
                      simulator=float(incl[(u >= a) & (u < b_)].mean()))
                 for a, b_ in zip(u_edges[:-1], u_edges[1:])]
        rows.append(dict(rho_c=rho_c, rmse=float(np.sqrt(np.mean((approx - incl) ** 2))), bins=cells))
    return dict(radius=radius, atoms=int(len(pts)), replays=replays, profiles=rows)


def crash_count(data_dir, n_total=1000, resolutions=(0.5, 0.6, 1.0)):
    raising = {res: 0 for res in resolutions}
    zero_radius = 0
    for k in range(n_total):
        _, _, radii, C, _ = load_pattern(data_dir, k)
        zero_radius += int((radii <= 0).any())
        for res in resolutions:
            raising[res] += int(not usable_clusters(radii, C, res).all())
    return dict(patterns=n_total, with_zero_radius=zero_radius,
                raising_by_resolution={str(k): v for k, v in raising.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--patterns", type=int, default=60, help="development patterns 0..N-1")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "density_grid.json")
    args = parser.parse_args()
    if args.patterns > 100:
        parser.error("use development patterns only (at most 100)")

    t0 = time.time()
    result = dict(region_check=region_check(args.data_dir, args.patterns, args.resolution),
                  single_cluster=single_cluster_profiles(),
                  crashes=crash_count(args.data_dir))
    result["seconds"] = round(time.time() - t0, 1)

    rc = result["region_check"]
    print(f"{rc['patterns']} patterns, {rc['atoms']:,} atoms "
          f"({rc['clusters_withheld_from_grid']} zero/tiny clusters withheld from the grid)")
    for r in rc["regions"]:
        print(f"  {r['region']:<46} {100 * r['share']:6.2f}%  grid {r['grid']:.4f}  "
              f"observed {r['observed']:.4f}  formula at atoms {r['formula_at_atoms']:.4f}")
    print("Reliability of the grid:")
    for r in rc["reliability"]:
        print(f"  [{r['lo']:.2f},{r['hi']:.2f})  n={r['atoms']:>9,}  grid {r['grid']:.4f}  "
              f"observed {r['observed']:.4f}  (se {r['se']:.4f})")
    sc = result["single_cluster"]
    print(f"One cluster, radius {sc['radius']}, {sc['atoms']} atoms, {sc['replays']} replays (grid / simulator):")
    for p in sc["profiles"]:
        cells = "  ".join(f"{c['grid']:.3f}/{c['simulator']:.3f}" for c in p["bins"])
        print(f"  rho_c={p['rho_c']}: {cells}   RMSE {p['rmse']:.3f}")
    cc = result["crashes"]
    print(f"generate_density_grid raises on {cc['raising_by_resolution']} of {cc['patterns']} patterns "
          f"(by resolution); {cc['with_zero_radius']} patterns contain a zero radius")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {args.output} ({result['seconds']} s)")


if __name__ == "__main__":
    main()
