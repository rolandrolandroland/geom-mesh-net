"""Pilot: an oracle for the guest field, and how close kernel smoothing gets to it.

Development patterns only (indices below 100). Nothing here is a stage result;
it sizes the problem so that the gates in ``experiments/reconstruction/ROADMAP.md`` are set
from measurements rather than guesses.

1. Oracle. For every atom, p* = P(guest | atoms, cluster geometry, theta),
   by replaying clustersim's labelling on the stored atoms, centres and radii.
   Selections in different clusters are independent and ``np.maximum`` over
   labels makes an atom a guest if any cluster picks it, so overlaps combine as
   a union. Atoms outside every sphere are guests with probability rho_b
   exactly. Checked against the realised labels, an independent draw.

2. Headroom. At retention eta the observed atoms are fitted and the removed
   atoms scored by log loss. Constant, Nadaraya-Watson kernel smoothing with a
   Gaussian bandwidth chosen on observed atoms only, and the oracle. Because
   expected log loss is H(p*) + KL(p* || q), differences between methods on
   held-out atoms are differences in mean KL to the truth.

Recorded in ``experiments/reconstruction/ROADMAP.md`` section 4.

Usage
-----
    python -m experiments.reconstruction.pilot.headroom
    python -m experiments.reconstruction.pilot.headroom --replays 200 --workers 6
"""

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from geom_mesh_net import paths

ETAS = (0.1, 0.37, 0.8)                      # retention: stress test, then 37% and 80% detectors
BANDWIDTHS = (0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)
GRID, EXTENT, EPS = 0.5, 60.0, 1e-6


def stratified_development_patterns(theta, per_cell=2):
    """Lowest-index development patterns in each (cr band, rho_c band) cell."""
    chosen = []
    for lo, hi in ((3, 6), (6, 10), (10, 15.01)):
        for clo, chi in ((0.2, 0.5), (0.5, 1.01)):
            cell = [k for k in range(100) if lo <= theta[k, 2] < hi and clo <= theta[k, 0] < chi]
            chosen += cell[:per_cell]
    return chosen


def replay_oracle(X, centres, radii, rho_c, rho_b, replays, rng):
    tree = cKDTree(X)
    miss = np.ones(len(X))
    inside = np.zeros(len(X), dtype=bool)
    for r, c in zip(radii, centres):
        if r <= 0:
            continue
        ins = np.asarray(tree.query_ball_point(c, r), dtype=int)
        N = ins.size
        if N == 0:
            continue
        inside[ins] = True
        n = int(round(N * rho_c))
        if n >= N:
            pi = np.ones(N)
        elif n == 0:
            pi = np.zeros(N)
        else:
            d = np.linalg.norm(X[ins] - c, axis=1)
            w = 1 - d / d.max()
            w = w / w.sum()
            hits = np.zeros(N)
            for _ in range(replays):
                hits[rng.choice(N, size=n, replace=False, p=w)] += 1
            pi = hits / replays
        miss[ins] *= 1 - pi
    return np.where(inside, 1 - miss, rho_b), inside


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def smooth(Xo, yo, h):
    edges = np.linspace(0, EXTENT, int(round(EXTENT / GRID)) + 1)
    tot = np.histogramdd(Xo, bins=[edges] * 3)[0].astype(np.float32)
    gst = np.histogramdd(Xo[yo == 1], bins=[edges] * 3)[0].astype(np.float32)
    s = h / GRID
    return (gaussian_filter(tot, s, mode="constant", truncate=4.0),
            gaussian_filter(gst, s, mode="constant", truncate=4.0))


def nadaraya_watson(fields, Xq, fallback):
    """Binned Nadaraya-Watson guest probability, trilinearly interpolated at Xq."""
    tot, gst = fields
    coords = (Xq / GRID - 0.5).T
    t = map_coordinates(tot, coords, order=1, mode="nearest")
    g = map_coordinates(gst, coords, order=1, mode="nearest")
    out = np.full(len(Xq), fallback, dtype=np.float64)
    ok = t > 1e-8
    out[ok] = g[ok] / t[ok]
    return np.clip(out, 0.0, 1.0)


def run_pattern(k, data_dir, theta, replays):
    rng = np.random.default_rng(1000 + k)
    rho_c, rho_b, cr, rb = theta[k]
    d = np.load(Path(data_dir) / f"clust_pattern_{k}.npz", allow_pickle=True)
    c, cen = d["coords"].item(), d["centers"].item()
    X = np.column_stack([c["x"], c["y"], c["z"]])
    C = np.column_stack([cen["x"], cen["y"], cen["z"]])
    y = np.isin(d["labels"], (2, 3)).astype(np.float64)

    t0 = time.time()
    p_star, inside = replay_oracle(X, C, d["radii"], rho_c, rho_b, replays, rng)
    t_oracle = time.time() - t0
    # Replay frequencies of exactly 0 or 1 are Monte Carlo artefacts, inside spheres only.
    p_star = np.where(inside, np.clip(p_star, 0.5 / replays, 1 - 0.5 / replays), p_star)

    rows = []
    t0 = time.time()
    for eta in ETAS:
        keep = rng.random(len(X)) < eta
        Xo, yo = X[keep], y[keep]
        Xh, yh, ph, inh = X[~keep], y[~keep], p_star[~keep], inside[~keep]
        const = yo.mean()

        val = rng.random(len(Xo)) < 0.2  # bandwidth chosen on observed atoms only
        fallback = yo[~val].mean()
        val_loss = {h: logloss(yo[val], nadaraya_watson(smooth(Xo[~val], yo[~val], h), Xo[val], fallback))
                    for h in BANDWIDTHS}
        h_cv = min(val_loss, key=val_loss.get)
        preds = {h: nadaraya_watson(smooth(Xo, yo, h), Xh, const) for h in BANDWIDTHS}
        held = {h: logloss(yh, preds[h]) for h in BANDWIDTHS}
        h_best = min(held, key=held.get)

        L_const, L_oracle = logloss(yh, np.full(len(yh), const)), logloss(yh, ph)
        gap = L_const - L_oracle

        def excess(mask, p):
            return logloss(yh[mask], p[mask]) - logloss(yh[mask], ph[mask])

        rows.append(dict(
            eta=eta, n_observed=int(keep.sum()), h_cv=h_cv, h_best_in_hindsight=h_best,
            L_const=L_const, L_nw=held[h_cv], L_nw_best=held[h_best], L_oracle=L_oracle,
            gap=gap, closed=(L_const - held[h_cv]) / gap, closed_best=(L_const - held[h_best]) / gap,
            mse_nw=float(np.mean((preds[h_cv] - ph) ** 2)),
            excess_in_spheres=excess(inh, preds[h_cv]), excess_matrix=excess(~inh, preds[h_cv]),
            share_in_spheres=float(inh.mean()),
        ))
    return dict(pattern=k, n_clusters=int(len(d["radii"])), rho_c=float(rho_c), rho_b=float(rho_b),
                cr=float(cr), rb=float(rb), seconds_oracle=round(t_oracle, 1),
                seconds_smoothing=round(time.time() - t0, 1), rows=rows,
                calib_p=p_star[inside], calib_y=y[inside],
                matrix_rho_b=float(rho_b), matrix_observed=float(y[~inside].mean()))


def oracle_calibration(p, y):
    bins = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0001])
    b = np.digitize(p, bins) - 1
    table = []
    for i in range(len(bins) - 1):
        m = b == i
        if m.any():
            obs = y[m].mean()
            table.append(dict(lo=float(bins[i]), hi=float(bins[i + 1]), atoms=int(m.sum()),
                              oracle=float(p[m].mean()), observed=float(obs),
                              se=float(np.sqrt(max(obs * (1 - obs), 1e-12) / m.sum()))))
    # Logistic recalibration of the realised labels on logit p*: slope 1, intercept 0 if calibrated.
    z = np.log(p / (1 - p))
    A = np.column_stack([np.ones_like(z), z])
    beta = np.array([0.0, 1.0])
    for _ in range(30):
        mu = 1 / (1 + np.exp(-(A @ beta)))
        beta += np.linalg.solve(A.T @ (A * (mu * (1 - mu))[:, None]), A.T @ (y - mu))
    return dict(atoms=int(len(p)), intercept=float(beta[0]), slope=float(beta[1]), bins=table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--theta", type=Path, default=paths.THETA_PATH)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "headroom.json")
    args = parser.parse_args()

    theta = np.load(args.theta)
    patterns = stratified_development_patterns(theta)
    print("development patterns:", patterns, flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(partial(run_pattern, data_dir=args.data_dir, theta=theta, replays=args.replays),
                           patterns)
    calibration = oracle_calibration(np.concatenate([r.pop("calib_p") for r in results]),
                                     np.concatenate([r.pop("calib_y") for r in results]))

    print(f"oracle vs realised labels, {calibration['atoms']:,} atoms inside spheres: "
          f"slope {calibration['slope']:.3f}, intercept {calibration['intercept']:+.4f}")
    for c in calibration["bins"]:
        print(f"  p* [{c['lo']:.2f},{c['hi']:.2f})  n={c['atoms']:>8,}  oracle {c['oracle']:.4f}  "
              f"observed {c['observed']:.4f}  (se {c['se']:.4f})")
    print(f"\n{'pattern':>7} {'clusters':>8} {'cr':>5} {'rho_c':>5} {'eta':>5} {'gap':>8} "
          f"{'closed':>7} {'best h':>7} {'h':>4} {'excess in':>10} {'excess out':>10}")
    for r in results:
        for row in r["rows"]:
            print(f"{r['pattern']:>7} {r['n_clusters']:>8} {r['cr']:>5.1f} {r['rho_c']:>5.2f} {row['eta']:>5.2f} "
                  f"{row['gap']:>8.4f} {row['closed']:>7.3f} {row['closed_best']:>7.3f} {row['h_cv']:>4} "
                  f"{row['excess_in_spheres']:>10.4f} {row['excess_matrix']:>10.4f}")

    out = dict(patterns=patterns, replays=args.replays, etas=ETAS, bandwidths=BANDWIDTHS,
               oracle_calibration=calibration, results=results, seconds=round(time.time() - t0, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(f"written {args.output} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
