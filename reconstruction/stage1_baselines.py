"""Stage 1: classical baselines and the headroom map; decides Gate 1.

For every test pattern and efficiency (a *cell*):

- B0, B1 and B2 (``field_baselines``) are fitted to the observed atoms, with
  every hyperparameter chosen by five-fold cross-validation on those atoms;
- each method, and the Stage 0 oracle, is scored by log loss on the scored atoms,
  overall and by region (O1);
- expected calibration error (O2) and Brier excess are recorded;
- for the predictive check (O11), all atoms are relabelled from each method's
  field and the 14 global features compared with the realised pattern's.

Gate 1, stated before this ran (``reconstruction/ROADMAP.md``, Stage 1): at least
25% of the 300 test cells have headroom, meaning B1 leaves at least 15% of the
constant-to-oracle gap open and at least 0.01 nats per atom.

Usage
-----
    PYTHONPATH=. python reconstruction/stage1_baselines.py
    PYTHONPATH=. python reconstruction/stage1_baselines.py --split development --limit 2 \
        --output /tmp/stage1_dev.json
"""

import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geom_mesh_net.core_functions import field_baselines as fb
from geom_mesh_net.core_functions import field_oracle as fo
from geom_mesh_net.core_functions import paper_spatial_features as psf
from inference.extract_features import build_config
from reconstruction import benchmark as bm

METHODS = ("B0", "B1", "B2", "oracle")
REGIONS = ("rim", "core", "matrix", "interior")
FAMILIES = {"G": slice(0, 4), "F": slice(4, 6), "K": slice(6, 11), "cross-G": slice(11, 14)}
STAGE1_ENTROPY = 20260916


def cell_seed(index, eta, salt):
    return int(np.random.SeedSequence([STAGE1_ENTROPY, int(index), int(round(1000 * eta)), salt]).generate_state(1)[0])


def assign_regions(coords, centres, radii, inside):
    """O1: rim if |d - r| <= 2 for some cluster, else core if d/r < 0.7, else matrix if outside, else interior."""
    rim = np.zeros(len(coords), dtype=bool)
    core = np.zeros(len(coords), dtype=bool)
    tree = cKDTree(coords)
    for centre, radius in zip(centres, radii):
        if radius <= 0:
            continue
        idx = np.asarray(tree.query_ball_point(centre, radius + 2.0 + 1e-9), dtype=int)
        if idx.size == 0:
            continue
        d = fo.point_distances(coords[idx], centre)
        rim[idx[np.abs(d - radius) <= 2.0]] = True
        core[idx[d / radius < 0.7]] = True
    region = np.full(len(coords), REGIONS.index("interior"), dtype=np.int8)
    region[~inside] = REGIONS.index("matrix")
    region[core] = REGIONS.index("core")
    region[rim] = REGIONS.index("rim")
    return region


def expected_calibration_error(y, q, bins=20):
    """O2: 20 equal-width bins on q, weighted by share of atoms."""
    b = np.minimum((np.asarray(q) * bins).astype(int), bins - 1)
    counts = np.bincount(b, minlength=bins)
    used = counts > 0
    mean_y = np.bincount(b, weights=y, minlength=bins)[used] / counts[used]
    mean_q = np.bincount(b, weights=q, minlength=bins)[used] / counts[used]
    return float(np.sum(counts[used] / len(q) * np.abs(mean_y - mean_q)))


def global_features(coords, guest, domain, config):
    labels = np.where(guest, 2, 0).astype(np.int8)
    points = {"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]}
    return np.asarray(psf.calculate_global_paper_features(points, labels, domain, guest_marks=(2, 3),
                                                          config=config).values, dtype=float)


def run_pattern(index, predictive=True):
    started = time.perf_counter()
    theta = bm.load_theta()
    rho_c, rho_b, cr, rb = (float(v) for v in theta[index])
    pattern = bm.load_pattern(index)
    X, y = pattern["coords"], pattern["guest"]
    p_star, n_spheres = bm.load_oracle(index)
    region = assign_regions(X, pattern["centres"], pattern["radii"], n_spheres > 0)
    config = build_config(preset="stage1")
    true_features = global_features(X, y, pattern["domain"], config).tolist() if predictive else None

    cells = []
    for eta in bm.EFFICIENCIES:
        mask = bm.thinning_mask(index, eta, len(X))
        fit = fb.fit_baselines(X[mask], y[mask], folds=5, seed=cell_seed(index, eta, 0))
        q = fb.predict_baselines(fit, X[mask], y[mask], X)
        q["oracle"] = p_star
        scored = ~mask
        ys = y[scored].astype(float)
        loss = {m: fb.log_loss(ys, q[m][scored]) for m in METHODS}
        gap = loss["B0"] - loss["oracle"]

        region_scored = region[scored]
        excess_by_region, share = {m: {} for m in METHODS if m != "oracle"}, {}
        for r_id, name in enumerate(REGIONS):
            sel = region_scored == r_id
            share[name] = float(sel.mean())
            if not sel.any():
                continue
            l_oracle = fb.log_loss(ys[sel], p_star[scored][sel])
            for m in excess_by_region:
                excess_by_region[m][name] = fb.log_loss(ys[sel], q[m][scored][sel]) - l_oracle

        cell = dict(
            pattern=index, eta=eta, rho_c=rho_c, rho_b=rho_b, cr=cr, rb=rb,
            n_clusters=int(len(pattern["radii"])),
            cr_band=bm.band(cr, bm.CR_BANDS), rho_c_band=bm.band(rho_c, bm.RHO_C_BANDS),
            n_observed=int(mask.sum()), n_scored=int(scored.sum()),
            bandwidth=fit.bandwidth, adaptive_k=fit.adaptive_k, adaptive_c=fit.adaptive_c,
            cv_loss_fixed=fit.cv_loss_fixed, cv_loss_adaptive=fit.cv_loss_adaptive,
            loss=loss, gap=gap,
            gap_closed={m: (loss["B0"] - loss[m]) / gap for m in ("B1", "B2")},
            excess={m: loss[m] - loss["oracle"] for m in METHODS},
            headroom=bool(bm.has_headroom(loss["B0"], loss["B1"], loss["oracle"])),
            excess_by_region=excess_by_region, region_share=share,
            brier_excess={m: float(np.mean((q[m][scored] - p_star[scored]) ** 2)) for m in METHODS},
            ece={m: expected_calibration_error(ys, q[m][scored]) for m in METHODS},
        )
        if predictive:
            cell["relabelled_features"] = {}
            for m_id, m in enumerate(METHODS):
                rng = np.random.default_rng(cell_seed(index, eta, 1 + m_id))
                relabelled = rng.random(len(X)) < q[m]
                cell["relabelled_features"][m] = global_features(X, relabelled, pattern["domain"], config).tolist()
        cells.append(cell)
    return dict(pattern=index, true_features=true_features, cells=cells, seconds=round(time.perf_counter() - started, 1))


def bootstrap_median(values, rng, reps=2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan")] * 3
    boots = np.median(rng.choice(values, size=(reps, len(values)), replace=True), axis=1)
    return [float(np.median(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def summarise(results):
    rng = np.random.default_rng(0)
    cells = [c for r in results for c in r["cells"]]
    headroom = np.array([c["headroom"] for c in cells])
    summary = {"n_cells": len(cells), "headroom_fraction": float(headroom.mean())}

    groups = {}
    for key_name, key in (("cr_band", "cr_band"), ("rho_c_band", "rho_c_band")):
        for band_id in sorted({c[key] for c in cells}):
            for eta in bm.EFFICIENCIES:
                sel = [c for c in cells if c[key] == band_id and c["eta"] == eta]
                if not sel:
                    continue
                groups[f"{key_name}{band_id}_eta{eta}"] = dict(
                    n=len(sel),
                    headroom_fraction=float(np.mean([c["headroom"] for c in sel])),
                    gap=bootstrap_median([c["gap"] for c in sel], rng),
                    gap_closed_B1=bootstrap_median([c["gap_closed"]["B1"] for c in sel], rng),
                    gap_closed_B2=bootstrap_median([c["gap_closed"]["B2"] for c in sel], rng),
                    excess_B1=bootstrap_median([c["excess"]["B1"] for c in sel], rng),
                    excess_B1_by_region={r: float(np.median([c["excess_by_region"]["B1"][r] for c in sel
                                                             if r in c["excess_by_region"]["B1"]]))
                                         for r in REGIONS if any(r in c["excess_by_region"]["B1"] for c in sel)},
                    bandwidths={str(h): sum(c["bandwidth"] == h for c in sel) for h in fb.BANDWIDTHS},
                )
    summary["groups"] = groups

    b2_wins = np.array([c["loss"]["B2"] < c["loss"]["B1"] for c in cells])
    head_cells = [c for c in cells if c["headroom"]]
    summary["B2_vs_B1"] = dict(
        win_fraction_all=float(b2_wins.mean()),
        win_fraction_headroom=float(np.mean([c["loss"]["B2"] < c["loss"]["B1"] for c in head_cells])) if head_cells else None,
        median_share_of_B1_remaining_gap_closed_headroom=bootstrap_median(
            [(c["loss"]["B1"] - c["loss"]["B2"]) / (c["loss"]["B1"] - c["loss"]["oracle"]) for c in head_cells], rng)
        if head_cells else None,
        adaptive_choices={f"k{k}_c{cc}": sum(c["adaptive_k"] == k and c["adaptive_c"] == cc for c in cells)
                          for k in fb.ADAPTIVE_K for cc in fb.ADAPTIVE_C},
    )
    summary["ece_median"] = {m: float(np.median([c["ece"][m] for c in cells])) for m in METHODS}
    summary["brier_excess_median"] = {m: float(np.median([c["brier_excess"][m] for c in cells])) for m in METHODS}
    summary["region_share_median"] = {r: float(np.median([c["region_share"][r] for c in cells])) for r in REGIONS}

    if results and results[0]["true_features"] is not None:
        truth = {r["pattern"]: np.array(r["true_features"]) for r in results}
        sd = np.std(np.array(list(truth.values())), axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.nan)  # a feature constant across patterns has no scale
        z = {m: [] for m in METHODS}
        for c in cells:
            for m in METHODS:
                z[m].append(np.abs((np.array(c["relabelled_features"][m]) - truth[c["pattern"]]) / sd))
        summary["predictive_check_median_abs_z"] = {
            m: {fam: float(np.nanmedian(np.array(z[m])[:, s])) for fam, s in FAMILIES.items()} for m in METHODS
        }
        summary["feature_between_pattern_sd"] = sd.tolist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=list(bm.SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-predictive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reconstruction/results/stage1_headroom.json"))
    args = parser.parse_args()

    indices = bm.split_indices(args.split)[: args.limit]
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(partial(run_pattern, predictive=not args.no_predictive), indices)):
            results.append(result)
            print(f"  {i + 1}/{len(indices)} pattern {result['pattern']} ({result['seconds']} s)", flush=True)
    results.sort(key=lambda r: r["pattern"])
    summary = summarise(results)
    gate = {"condition": "headroom_fraction >= 0.25", "headroom_fraction": summary["headroom_fraction"],
            "passed": summary["headroom_fraction"] >= 0.25} if args.split == "test" and args.limit is None else None
    report = dict(split=args.split, patterns=indices, efficiencies=bm.EFFICIENCIES, bandwidths=fb.BANDWIDTHS,
                  adaptive_k=fb.ADAPTIVE_K, adaptive_c=fb.ADAPTIVE_C, regions=REGIONS, methods=METHODS,
                  gate1=gate, summary=summary, results=results, seconds=round(time.perf_counter() - started, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"gate1": gate, "headroom_fraction": summary["headroom_fraction"],
                      "B2_vs_B1": summary["B2_vs_B1"]}, indent=2))
    print(f"written {args.output} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
