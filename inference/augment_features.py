"""Observation augmentation: several independent thinnings per pattern.

ROADMAP section 7 lists this as the free remedy for the sample-size limits that
Stages 2 through 4 all ran into. Instead of simulating new parameter draws, take
the patterns already on disk and observe each of them several times, by
independently thinning the point cloud.

Why this is legitimate and not merely duplicating rows: each thinning is a
genuine draw from p(s | theta). The parameter is unchanged but the observation is
not, so the flow sees how much the summary features move for a *fixed* physical
structure. That is exactly the variability a posterior needs in order to be wide
enough, and Stage 3 found the one miscalibrated parameter (``rho_b``) to be
mildly overconfident -- a posterior too narrow for its own observation noise.

It also models something physically real. An atom probe detects a fraction of the
atoms that reach it, so a measured point cloud already *is* a thinned realization
of the underlying structure. Training on thinned clouds is arguably closer to the
real observation process than training on the complete one.

The caveat that must be respected downstream: replicates of one pattern share a
theta, so they are not independent. Any split must keep all replicates of a
pattern on the same side, or the held-out set leaks and every calibration number
becomes meaningless. This file records ``pattern_index`` for exactly that
purpose.

Thinning is seeded per (pattern, replicate) rather than using the module-level
generator in ``clustersim``, so the cache is reproducible like everything else in
this pipeline.

Usage
-----
    PYTHONPATH=. python inference/augment_features.py --workers 7
    PYTHONPATH=. python inference/augment_features.py --replicates 2 --retention 0.7
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from geom_mesh_net.core_functions import paper_spatial_features as psf
from inference.extract_features import (
    GUEST_MARKS, build_config, discover_patterns, report_gate,
)


# A detector that records roughly half of what reaches it. Real atom probe
# efficiencies run about 37% to 80%, so this sits in the middle of the range.
DEFAULT_RETENTION = 0.5
DEFAULT_REPLICATES = 4
THINNING_SEED = 20260911


def thin(coords, labels, retention, seed_parts):
    """Independent Bernoulli thinning with a reproducible, collision-free seed."""
    rng = np.random.default_rng(list(seed_parts))
    n = len(labels)
    keep = rng.random(n) < retention
    return {axis: np.asarray(values)[keep] for axis, values in coords.items()}, labels[keep]


def extract_one_replicate(args):
    index, replicate, data_dir, config, retention = args
    path = Path(data_dir) / f"clust_pattern_{index}.npz"
    started = time.perf_counter()
    try:
        with np.load(path, allow_pickle=True) as handle:
            coords = handle["coords"].item()
            domain = handle["domain"].item()
            labels = handle["labels"]
        thinned_coords, thinned_labels = thin(
            coords, labels, retention, (THINNING_SEED, index, replicate)
        )
        result = psf.calculate_global_paper_features(
            thinned_coords, thinned_labels, domain,
            guest_marks=GUEST_MARKS, config=config,
        )
        return {
            "index": index,
            "replicate": replicate,
            "values": np.asarray(result.values, dtype=np.float64),
            "k_extrema_interior": np.asarray(result.k_extrema_interior, dtype=bool),
            "n_points": int(len(thinned_labels)),
            "seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return {
            "index": index,
            "replicate": replicate,
            "values": np.full(len(psf.PAPER_FEATURE_NAMES), np.nan),
            "k_extrema_interior": np.zeros(3, dtype=bool),
            "n_points": 0,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("inference/features/augmented_features.npz"),
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--retention", type=float, default=DEFAULT_RETENTION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    if not 0.0 < args.retention <= 1.0:
        raise SystemExit("--retention must be in (0, 1]")

    config = build_config()
    indices = discover_patterns(args.data_dir, args.limit)
    if not indices:
        raise SystemExit(f"no clust_pattern_*.npz found in {args.data_dir}")

    payload = [
        (index, replicate, str(args.data_dir), config, args.retention)
        for index in indices
        for replicate in range(args.replicates)
    ]
    print(
        f"Observation augmentation: {len(indices)} patterns x "
        f"{args.replicates} thinnings at {args.retention:.0%} retention "
        f"= {len(payload)} feature evaluations"
    )
    print(f"  null_model {config.null_model}, k_transform {config.k_transform}, "
          f"k_r_max {config.k_r_max}")
    print(f"  workers {args.workers}\n")

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(extract_one_replicate, payload, chunksize=4):
            results.append(result)
            if len(results) % 250 == 0 or len(results) == len(payload):
                rate = len(results) / max(time.perf_counter() - started, 1e-9)
                print(
                    f"  {len(results)}/{len(payload)}  {rate:.1f}/s  "
                    f"eta {(len(payload) - len(results)) / rate / 60:.1f} min",
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    results.sort(key=lambda r: (r["index"], r["replicate"]))
    values = np.stack([r["values"] for r in results])
    pattern_index = np.array([r["index"] for r in results], dtype=np.int32)
    replicate = np.array([r["replicate"] for r in results], dtype=np.int16)
    interior = np.stack([r["k_extrema_interior"] for r in results])
    n_points = np.array([r["n_points"] for r in results], dtype=np.int32)
    errors = [r["error"] for r in results]

    print(f"\n  wall clock {elapsed / 60:.1f} min")
    print(
        f"  thinned clouds hold {n_points[n_points > 0].mean():.0f} points on "
        f"average, against 216000 complete"
    )

    gate = report_gate(values, interior, pattern_index, errors)

    # How much do the features actually move between observations of the same
    # structure? That spread is the entire point of the exercise: it is the
    # observation noise the posterior has to be wide enough to cover.
    print("\n  Within-pattern spread of each feature, as a fraction of its")
    print("  between-pattern spread. Larger means the observation process")
    print("  contributes more of what the model sees.")
    finite = np.all(np.isfinite(values), axis=1)
    ratios = {}
    for position, name in enumerate(psf.PAPER_FEATURE_NAMES):
        column = values[finite, position]
        groups = pattern_index[finite]
        within = np.mean([
            column[groups == g].std()
            for g in np.unique(groups)
            if (groups == g).sum() > 1
        ])
        between = column.std()
        ratios[name] = float(within / between) if between > 0 else float("nan")
    for name, ratio in sorted(ratios.items(), key=lambda kv: -kv[1]):
        print(f"    {name:>16} {ratio:>6.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        values=values,
        pattern_index=pattern_index,
        replicate=replicate,
        k_extrema_interior=interior,
        n_points=n_points,
        feature_names=np.array(psf.PAPER_FEATURE_NAMES),
        errors=np.array(errors),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "replicates": args.replicates,
                "retention": args.retention,
                "thinning_seed": THINNING_SEED,
                "n_patterns": len(indices),
                "n_rows": len(values),
                "config": asdict(config),
                "wall_clock_seconds": round(elapsed, 2),
                "mean_points_retained": float(n_points[n_points > 0].mean()),
                "gate": gate,
                "within_over_between_sd": ratios,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
