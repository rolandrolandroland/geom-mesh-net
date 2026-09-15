"""Stage 0: build the replay oracle and decide Gate 0 (reconstruction/ROADMAP.md, Stage 0).

For each pattern, ``field_oracle.replay_oracle`` gives every atom's guest
probability. The realised labels are an independent draw from the same process,
so a correct oracle is calibrated against them. Gate 0, stated before this ran,
on the in-sphere atoms of development patterns 0-99 of the benchmark dataset:

- logistic recalibration of the labels on logit p*: slope in [0.99, 1.01] and
  |intercept| <= 0.01;
- every reliability bin holding at least 10,000 atoms lies within
  max(0.005, three standard errors) of its observed guest fraction;
- the matrix guest fraction lies within three standard errors of rho_b in at
  least 95 of the 100 patterns;
- thinning masks regenerate to their frozen checksums.

The test suite passing is the gate's fifth condition, checked separately.

The same checks are reported for the lattice dataset ``data/`` without gating.
Oracles for the benchmark's development, validation and test patterns are cached
to ``reconstruction/results/oracle/`` for later stages.

Usage
-----
    PYTHONPATH=. python reconstruction/stage0_oracle.py
    PYTHONPATH=. python reconstruction/stage0_oracle.py --workers 6 --replays 1000
"""

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from geom_mesh_net.core_functions import field_oracle as fo
from reconstruction import benchmark as bm

RELIABILITY_BINS = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0 + 1e-9])
SEED_OFFSET = 1_000_000  # oracle replay seeds: never equal to a thinning or simulation seed


def build(index, data_dir, replays, cache):
    theta = bm.load_theta()
    rho_c, rho_b = float(theta[index, 0]), float(theta[index, 1])
    pattern = bm.load_pattern(index, data_dir)
    started = time.perf_counter()
    oracle = fo.replay_oracle(pattern["coords"], pattern["centres"], pattern["radii"], rho_c, rho_b,
                              replays=replays, seed=SEED_OFFSET + index)
    seconds = time.perf_counter() - started
    if cache:
        bm.ORACLE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(bm.oracle_path(index), p=oracle.p.astype(np.float32),
                 n_spheres=np.minimum(oracle.n_spheres, 255).astype(np.uint8))

    inside, y = oracle.inside, pattern["guest"]
    matrix_n = int((~inside).sum())
    matrix_obs = float(y[~inside].mean()) if matrix_n else float("nan")
    matrix_se = np.sqrt(rho_b * (1 - rho_b) / max(matrix_n, 1))
    return dict(
        index=index, n_clusters=int(len(pattern["radii"])), seconds=round(seconds, 2),
        share_in_spheres=float(inside.mean()), in_p=oracle.p[inside].astype(np.float32),
        in_y=y[inside], matrix_n=matrix_n, matrix_observed=matrix_obs, rho_b=rho_b,
        matrix_within_3se=bool(abs(matrix_obs - rho_b) <= 3 * matrix_se),
    )


def logistic_recalibration(p, y):
    p = np.clip(p.astype(np.float64), 1e-9, 1 - 1e-9)
    z = np.log(p / (1 - p))
    A = np.column_stack([np.ones_like(z), z])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(A @ beta)))
        step = np.linalg.solve(A.T @ (A * (mu * (1 - mu))[:, None]), A.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(A @ beta)))
    cov = np.linalg.inv(A.T @ (A * (mu * (1 - mu))[:, None]))
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def assess(results):
    p = np.concatenate([r.pop("in_p") for r in results])
    y = np.concatenate([r.pop("in_y") for r in results]).astype(np.float64)
    intercept, slope, se_intercept, se_slope = logistic_recalibration(p, y)
    bins = []
    b = np.digitize(p, RELIABILITY_BINS) - 1
    for i in range(len(RELIABILITY_BINS) - 1):
        m = b == i
        n = int(m.sum())
        if n == 0:
            continue
        obs, pred = float(y[m].mean()), float(p[m].mean())
        se = float(np.sqrt(max(obs * (1 - obs), 1e-12) / n))
        tolerance = max(0.005, 3 * se)
        bins.append(dict(lo=float(RELIABILITY_BINS[i]), hi=float(min(RELIABILITY_BINS[i + 1], 1.0)), atoms=n,
                         oracle=pred, observed=obs, gap=obs - pred, tolerance=tolerance,
                         gated=n >= 10_000, within=abs(obs - pred) <= tolerance))
    matrix_ok = sum(r["matrix_within_3se"] for r in results)
    return dict(
        in_sphere_atoms=int(len(p)), intercept=intercept, slope=slope,
        intercept_se=se_intercept, slope_se=se_slope, reliability=bins,
        matrix_within_3se=int(matrix_ok), patterns=len(results),
        checks={
            "slope_in_[0.99,1.01]": 0.99 <= slope <= 1.01,
            "abs_intercept_le_0.01": abs(intercept) <= 0.01,
            "reliability_bins_within_tolerance": all(x["within"] for x in bins if x["gated"]),
            "matrix_within_3se_in_at_least_95": matrix_ok >= 95,
        },
    )


def masks_match_frozen():
    frozen = json.loads((bm.BENCHMARK_DIR / "mask_checksums.json").read_text())
    n_atoms = json.loads((bm.BENCHMARK_DIR / "benchmark.json").read_text())["atoms_per_pattern"]
    mismatches = [key for key, digest in frozen.items()
                  if bm.mask_checksum(bm.thinning_mask(int(key.split(":")[0]), float(key.split(":")[1]), n_atoms)) != digest]
    return len(frozen), mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("reconstruction/results/stage0_gate.json"))
    args = parser.parse_args()
    started = time.perf_counter()

    report = {"replays": args.replays, "datasets": {}}
    with Pool(args.workers) as pool:
        for name, data_dir, gated in (("benchmark", bm.DATA_DIR, True), ("lattice", bm.LATTICE_DATA_DIR, False)):
            dev = bm.split_indices("development")
            results = pool.map(partial(build, data_dir=data_dir, replays=args.replays, cache=gated), dev)
            summary = assess(results)
            summary["data_dir"] = str(data_dir)
            summary["gated"] = gated
            summary["per_pattern"] = results
            report["datasets"][name] = summary
            print(f"[{name}] slope {summary['slope']:.4f} ± {summary['slope_se']:.4f}, "
                  f"intercept {summary['intercept']:+.4f} ± {summary['intercept_se']:.4f}, "
                  f"matrix within 3 se in {summary['matrix_within_3se']}/100, "
                  f"{summary['in_sphere_atoms']:,} in-sphere atoms", flush=True)
            for x in summary["reliability"]:
                flag = "" if x["within"] or not x["gated"] else "  <-- outside tolerance"
                print(f"    p* [{x['lo']:.2f},{x['hi']:.2f})  n={x['atoms']:>9,}  oracle {x['oracle']:.4f}  "
                      f"observed {x['observed']:.4f}  gap {x['gap']:+.4f}  tol {x['tolerance']:.4f}{flag}")

        # Cache the benchmark's validation and test oracles for later stages.
        later = bm.split_indices("validation") + bm.split_indices("test")
        cached = pool.map(partial(build, data_dir=bm.DATA_DIR, replays=args.replays, cache=True), later)
        report["cached_validation_and_test"] = len(cached)

    total, mismatches = masks_match_frozen()
    report["masks"] = {"checked": total, "mismatches": mismatches[:20]}
    checks = dict(report["datasets"]["benchmark"]["checks"])
    checks["masks_match_frozen_checksums"] = not mismatches
    report["gate0"] = {"checks": checks, "passed_excluding_tests": all(checks.values())}
    report["seconds"] = round(time.perf_counter() - started, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["gate0"], indent=2))
    print(f"masks: {total} checked, {len(mismatches)} mismatched; written {args.output} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
