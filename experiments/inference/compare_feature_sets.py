"""Compare how recoverable each parameter is under different feature sets.

Used to settle three questions raised by checking this pipeline against rapt, the
reference implementation for Bennett, Proudian and Zimmerman (2023):

1. How much did the legacy port's fabricated K features cost? Compare legacy
   features against rapt-faithful ones on the same patterns.
2. rapt drops any pattern with a missing feature. Is dropping or masking better?
3. rapt shares one underlying point pattern across all training and test data.
   How much of the gap between this dataset and the paper is that design?

The screen is ridge regression over repeated random splits, the same instrument as
``screen_features.py``: a point-estimate lower bound, not a posterior, but cheap
enough to compare many variants. Zero-cluster patterns are excluded throughout;
they are structural outliers that dominate R^2 without saying anything about the
features.

``--paired`` adds a split-by-split comparison of the first two sets, which must
hold the same patterns (for example the same parameters simulated with and without
a shared point pattern). Each split trains and tests both sets on identical rows,
so the per-split difference isolates the feature set from the partition.

A subset caution. rapt's drop rule keeps a biased sample -- it discards patterns
whose K curve has no peak, which depends on cluster radius -- so an R^2 measured on
those rows is not comparable to one measured on every row. Each variant is scored
on its own rows and on the subset every variant shares.

Usage
-----
    python -m experiments.inference.compare_feature_sets \\
        --set legacy=experiments/inference/features/global_features.npz \\
        --set paper=experiments/inference/features/paper_features.npz
    python -m experiments.inference.compare_feature_sets --paired \\
        --set independent=experiments/inference/features/paper_features.npz \\
        --set shared=experiments/inference/features/shared_upp_paper_features.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from geom_mesh_net import paths
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES
from geom_mesh_net.statistics.paper_spatial_features import PAPER_FEATURE_NAMES
from geom_mesh_net.statistics.presets import MAY_BE_MISSING


DEFINED = [i for i, n in enumerate(PAPER_FEATURE_NAMES) if n not in MAY_BE_MISSING]
N_SPLITS = 25
PENALTY = 1e-3


def ridge_r2(features, targets, rows, n_splits=N_SPLITS, seed=0):
    """Mean and sd over random 80/20 splits of the best of linear and quadratic ridge.

    Missing values are imputed with the training-split median, so no information
    from held-out rows reaches the imputation.
    """
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    scores = {name: [] for name in PARAMETER_NAMES}
    for _ in range(n_splits):
        order = rng.permutation(rows)
        cut = int(round(0.8 * len(order)))
        train, test = order[:cut], order[cut:]
        X = features.astype(float).copy()
        medians = np.nanmedian(X[train], axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        missing = ~np.isfinite(X)
        X[missing] = np.take(medians, np.nonzero(missing)[1])
        mean, sd = X[train].mean(0), X[train].std(0)
        sd[sd == 0] = 1.0
        Z = (X - mean) / sd
        for position, name in enumerate(PARAMETER_NAMES):
            y = targets[:, position]
            best = -np.inf
            for design in (np.column_stack([np.ones(len(Z)), Z]),
                           np.column_stack([np.ones(len(Z)), Z, Z**2])):
                A = design[train]
                W = np.linalg.solve(A.T @ A + PENALTY * len(train) * np.eye(A.shape[1]),
                                    A.T @ y[train])
                pred = design[test] @ W
                total = ((y[test] - y[train].mean()) ** 2).sum()
                best = max(best, 1 - ((y[test] - pred) ** 2).sum() / total)
            scores[name].append(best)
    return {name: (float(np.mean(v)), float(np.std(v, ddof=1))) for name, v in scores.items()}


def paired_differences(first, second, theta, subsets, n_seeds=100):
    """Score two feature sets on identical splits and report the differences.

    Each seed draws two splits, used for both sets, so every difference compares
    the sets on the same training and test rows.
    """
    results = {}
    for name, (rows, columns) in subsets.items():
        a = np.array([[v[0] for v in ridge_r2(first[:, columns], theta, rows, 2, s).values()]
                      for s in range(n_seeds)])
        b = np.array([[v[0] for v in ridge_r2(second[:, columns], theta, rows, 2, s).values()]
                      for s in range(n_seeds)])
        results[name] = {
            "rows": len(rows),
            **{param: {"first": float(a[:, j].mean()), "second": float(b[:, j].mean()),
                       "difference": float((b - a)[:, j].mean()),
                       "second_better_fraction": float(np.mean(b[:, j] > a[:, j]))}
               for j, param in enumerate(PARAMETER_NAMES)},
        }
    return results


def load(path):
    with np.load(path, allow_pickle=False) as cached:
        return cached["values"].astype(float), cached["index"].astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", required=True,
                        help="label=path to a Stage 1 feature cache; repeatable")
    parser.add_argument("--theta", type=Path, default=paths.THETA_PATH)
    parser.add_argument("--descriptors", type=Path,
                        default=paths.GROUND_TRUTH_DIR / "descriptors.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--paired", action="store_true",
                        help="also compare the first two sets split by split")
    args = parser.parse_args()

    theta_all = np.load(args.theta)
    degenerate = np.load(args.descriptors)["degenerate"]

    sets = {}
    for spec in args.set:
        label, path = spec.split("=", 1)
        values, index = load(path)
        order = np.argsort(index)
        sets[label] = (values[order], index[order])

    common_index = sorted(set.intersection(*[set(i.tolist()) for _, i in sets.values()]))
    usable = [i for i in common_index if not degenerate[i]]

    variants = {}
    for label, (values, index) in sets.items():
        lookup = {p: r for r, p in enumerate(index)}
        rows_all = [lookup[p] for p in usable]
        X = values
        theta = theta_all[index]
        complete = [r for r in rows_all if np.all(np.isfinite(X[r]))]
        indicators = (~np.isfinite(X[:, [PAPER_FEATURE_NAMES.index(n) for n in MAY_BE_MISSING]])).astype(float)
        has_missing = bool(np.any(~np.isfinite(X[rows_all])))

        variants[f"{label}: 8 always-defined features"] = (X[:, DEFINED], theta, rows_all, index)
        if has_missing:
            variants[f"{label}: 14 features, drop incomplete (rapt rule)"] = (X, theta, complete, index)
            variants[f"{label}: 14 features, impute + missing flags"] = (
                np.column_stack([X, indicators]), theta, rows_all, index)
        else:
            variants[f"{label}: 14 features"] = (X, theta, rows_all, index)

    # The subset every variant can be scored on: patterns complete in every set.
    shared = set(usable)
    for label, (values, index) in sets.items():
        lookup = {p: r for r, p in enumerate(index)}
        shared &= {p for p in usable if np.all(np.isfinite(values[lookup[p]]))}
    shared = sorted(shared)

    print(f"{len(usable)} non-degenerate patterns common to all sets; "
          f"{len(shared)} complete in every set\n")
    header = f"{'variant':<52}{'rows':>6}" + "".join(f"{n:>15}" for n in PARAMETER_NAMES)
    results = {}
    for scope, rows_for in (("own rows", None), ("shared complete subset", shared)):
        print(f"--- R^2 on {scope} ---")
        print(header)
        for name, (X, theta, rows, index) in variants.items():
            if rows_for is not None:
                lookup = {p: r for r, p in enumerate(index)}
                rows = [lookup[p] for p in rows_for]
            scores = ridge_r2(X, theta, rows)
            results.setdefault(name, {})[scope] = {"rows": len(rows), "r2": scores}
            print(f"{name:<52}{len(rows):>6}" + "".join(
                f"{m:>9.3f}±{s:<5.3f}" for m, s in scores.values()))
        print()

    if args.paired:
        (label_a, (first, index_a)), (label_b, (second, index_b)) = list(sets.items())[:2]
        if not np.array_equal(index_a, index_b):
            raise SystemExit("--paired needs both sets to hold the same patterns")
        theta = theta_all[index_a]
        ok = ~degenerate[index_a]
        complete = np.all(np.isfinite(first), 1) & np.all(np.isfinite(second), 1)
        all_columns = list(range(first.shape[1]))
        subsets = {
            "8 always-defined features, all rows": (np.flatnonzero(ok), DEFINED),
            "14 features, complete in both": (np.flatnonzero(ok & complete), all_columns),
            "14 features, complete in both, cr < 6.5 (paper's radius range)":
                (np.flatnonzero(ok & complete & (theta[:, 2] < 6.5)), all_columns),
        }
        paired = paired_differences(first, second, theta, subsets)
        print(f"--- paired: {label_b} minus {label_a}, 200 identical splits ---")
        for name, entry in paired.items():
            print(f"{name}  ({entry['rows']} rows)")
            for param in PARAMETER_NAMES:
                e = entry[param]
                print(f"    {param:<6} {e['first']:.3f} -> {e['second']:.3f}  "
                      f"diff {e['difference']:+.3f}  {label_b} better in "
                      f"{e['second_better_fraction']:.0%} of splits")
        results["paired"] = {"first": label_a, "second": label_b, "subsets": paired}
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
