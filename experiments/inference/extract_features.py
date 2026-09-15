"""Stage 1: extract global spatial-summary features for every pattern.

Computes the 14 Bennett et al. global features for each simulated pattern and
caches them in a single ``.npz`` alongside the configuration that produced them,
so that Stage 2 never touches a point cloud.

Configuration choices, all justified in ``experiments/inference/ROADMAP.md``:

``null_model="csr"`` (section 2.3)
    The closed-form CSR baseline rather than random relabeling. Posterior
    inference needs only a deterministic statistic of the point cloud, not a
    hypothesis test against a null, and the two give nearly identical features
    at a quarter of the cost.

``k_transform="sqrt"`` (section 8.2)
    The Bennett et al. definition. ``cube_root`` is available for a controlled
    comparison in Stage 4.

``k_r_max=40.0``, ``k_num_radii=801`` (section 8.3)
    The default of 10.0 leaves roughly two thirds of patterns with no interior
    K extremum, so the radius-valued features silently return a grid endpoint.
    At 40.0 all 16 spot-checked patterns reach an interior Rm, and raising the
    radius costs nothing in feature quality.

The gate: at least 95% of patterns must yield 14 finite features, and at least
80% must have a genuinely interior Rm. Failing either means the feature
definitions are still degenerate and Stage 2 would train on noise.

Usage
-----
    python -m experiments.inference.extract_features
    python -m experiments.inference.extract_features --limit 24 --workers 8
    python -m experiments.inference.extract_features --k-transform cube_root \\
        --output experiments/inference/features/global_cube_root.npz
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from geom_mesh_net import paths
from geom_mesh_net.statistics import paper_spatial_features as psf
from geom_mesh_net.statistics.presets import MAY_BE_MISSING, PRESETS, build_config


GUEST_MARKS = (2, 3)

GATE_MINIMUM_FINITE_FRACTION = 0.95
GATE_MINIMUM_INTERIOR_RM_FRACTION = 0.80


def extract_one(args):
    """Compute features for a single pattern. Runs in a worker process.

    Returns a plain dict so it pickles cheaply; the summary curves are large
    and deliberately not returned.
    """
    index, data_dir, config = args
    path = Path(data_dir) / f"clust_pattern_{index}.npz"
    started = time.perf_counter()
    try:
        with np.load(path, allow_pickle=True) as handle:
            coords = handle["coords"].item()
            domain = handle["domain"].item()
            labels = handle["labels"]
        result = psf.calculate_global_paper_features(
            coords,
            labels,
            domain,
            guest_marks=GUEST_MARKS,
            config=config,
        )
        return {
            "index": index,
            "values": np.asarray(result.values, dtype=np.float64),
            "k_extrema_interior": np.asarray(result.k_extrema_interior, dtype=bool),
            "seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        # A pattern that cannot produce features must be recorded rather than
        # dropped: silently omitting failures biases the training set.
        return {
            "index": index,
            "values": np.full(len(psf.PAPER_FEATURE_NAMES), np.nan),
            "k_extrema_interior": np.zeros(3, dtype=bool),
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def discover_patterns(data_dir, limit=None):
    indices = []
    index = 0
    while True:
        if not (data_dir / f"clust_pattern_{index}.npz").exists():
            break
        indices.append(index)
        index += 1
        if limit is not None and len(indices) >= limit:
            break
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument(
        "--output", type=Path, default=paths.INFERENCE_DIR / "features" / "global_features.npz"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel processes; the loop is embarrassingly parallel",
    )
    parser.add_argument(
        "--preset", choices=tuple(PRESETS), default="stage1",
        help="stage1 reproduces the committed results; paper matches rapt's walkthrough",
    )
    parser.add_argument("--k-transform", choices=psf.K_TRANSFORMS, default=None)
    parser.add_argument("--k-r-max", type=float, default=None)
    parser.add_argument("--null-model", choices=psf.NULL_MODELS, default=None)
    parser.add_argument(
        "--force", action="store_true", help="recompute even if the cache matches"
    )
    args = parser.parse_args()

    overrides = {}
    if args.k_transform:
        overrides["k_transform"] = args.k_transform
    if args.k_r_max is not None:
        overrides["k_r_max"] = args.k_r_max
    if args.null_model:
        overrides["null_model"] = args.null_model
    config = build_config(overrides, preset=args.preset)

    indices = discover_patterns(args.data_dir, args.limit)
    if not indices:
        raise SystemExit(f"no clust_pattern_*.npz found in {args.data_dir}")

    signature = json.dumps(
        {"config": asdict(config), "guest_marks": list(GUEST_MARKS),
         "n_patterns": len(indices)},
        sort_keys=True,
    )
    if args.output.exists() and not args.force:
        with np.load(args.output, allow_pickle=False) as cached:
            if str(cached["signature"]) == signature:
                print(f"cache at {args.output} already matches this configuration")
                print("pass --force to recompute")
                report_gate(
                    cached["values"], cached["k_extrema_interior"],
                    cached["index"], [str(e) for e in cached["errors"]],
                    config.feature_method,
                )
                return

    print(f"Stage 1: global features for {len(indices)} patterns")
    print(f"  null_model   {config.null_model}")
    print(f"  k_transform  {config.k_transform}")
    print(f"  k_r_max      {config.k_r_max}  ({config.k_num_radii} radii)")
    print(f"  workers      {args.workers}\n")

    payload = [(i, str(args.data_dir), config) for i in indices]
    started = time.perf_counter()
    results = []
    if args.workers <= 1:
        for item in payload:
            results.append(extract_one(item))
            _progress(len(results), len(payload), started)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(extract_one, payload, chunksize=4):
                results.append(result)
                _progress(len(results), len(payload), started)
    elapsed = time.perf_counter() - started
    print()

    results.sort(key=lambda r: r["index"])
    index_array = np.array([r["index"] for r in results], dtype=np.int32)
    values = np.stack([r["values"] for r in results])
    interior = np.stack([r["k_extrema_interior"] for r in results])
    seconds = np.array([r["seconds"] for r in results])
    errors = [r["error"] for r in results]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        index=index_array,
        values=values,
        k_extrema_interior=interior,
        feature_names=np.array(psf.PAPER_FEATURE_NAMES),
        seconds=seconds,
        errors=np.array(errors),
        signature=np.array(signature),
    )
    print(f"wrote {args.output}  values {values.shape}")
    print(
        f"  wall clock {elapsed / 60:.1f} min, "
        f"{seconds.sum() / len(seconds):.2f} s/pattern of CPU work, "
        f"speedup {seconds.sum() / elapsed:.1f}x"
    )

    metadata = {
        "stage": 1,
        "config": asdict(config),
        "guest_marks": list(GUEST_MARKS),
        "n_patterns": len(indices),
        "workers": args.workers,
        "wall_clock_seconds": round(elapsed, 2),
        "cpu_seconds": round(float(seconds.sum()), 2),
        "gate": report_gate(values, interior, index_array, errors, config.feature_method),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {metadata_path}")


def _progress(done, total, started):
    if done % 50 and done != total:
        return
    rate = done / max(time.perf_counter() - started, 1e-9)
    remaining = (total - done) / rate if rate else 0.0
    print(
        f"  {done}/{total}  {rate:.1f} patterns/s  "
        f"eta {remaining / 60:.1f} min",
        flush=True,
    )


def report_gate(values, interior, index_array, errors, feature_method="legacy_port"):
    """Evaluate and print the Stage 1 gate. Returns a dict for the metadata.

    Under the legacy port every feature is expected to be finite, and Rm must be
    a genuine interior extremum in at least 80% of patterns.

    Under rapt semantics a missing K feature is an honest NA rather than a
    failure, so the finiteness gate applies only to the features that are always
    defined. What matters instead is how many patterns survive rapt's own rule --
    drop any row with a missing feature -- and that is reported rather than
    gated, because whether to drop or mask those rows is a downstream decision.
    """
    values = np.asarray(values, dtype=float)
    interior = np.asarray(interior)
    total = len(values)
    names = psf.PAPER_FEATURE_NAMES
    defined = [i for i, n in enumerate(names) if n not in MAY_BE_MISSING]

    if feature_method == "rapt":
        finite_rows = np.all(np.isfinite(values[:, defined]), axis=1)
    else:
        finite_rows = np.all(np.isfinite(values), axis=1)
    finite_fraction = float(finite_rows.mean())
    complete_rows = np.all(np.isfinite(values), axis=1)
    complete_fraction = float(complete_rows.mean())
    interior_rm_fraction = float(interior[:, 0].mean())
    failures = [
        (int(index_array[i]), errors[i]) for i in range(total) if errors[i]
    ]

    label = "always-defined features finite" if feature_method == "rapt" else "all 14 features finite"
    print(f"\nStage 1 gate [{feature_method}]")
    print(
        f"  {label:<31}: {finite_rows.sum()}/{total} "
        f"({finite_fraction:.1%})  need >= {GATE_MINIMUM_FINITE_FRACTION:.0%}"
    )
    for position, name in enumerate(("Rm", "Rdm", "Rddm")):
        fraction = float(interior[:, position].mean())
        note = (
            f"  need >= {GATE_MINIMUM_INTERIOR_RM_FRACTION:.0%}"
            if name == "Rm" and feature_method != "rapt"
            else ""
        )
        print(
            f"  {name + ' defined':<31}: "
            f"{interior[:, position].sum()}/{total} ({fraction:.1%}){note}"
        )
    if feature_method == "rapt":
        print(
            f"  {'complete rows (rapt keeps)':<31}: {complete_rows.sum()}/{total} "
            f"({complete_fraction:.1%})"
        )
        for position, name in enumerate(names):
            missing = int(np.isnan(values[:, position]).sum())
            if missing:
                print(f"    {name:<14} missing in {missing}")
    if failures:
        print(f"  patterns that errored: {len(failures)}")
        for index, message in failures[:5]:
            print(f"    pattern {index}: {message}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")

    finite_pass = finite_fraction >= GATE_MINIMUM_FINITE_FRACTION
    if feature_method == "rapt":
        passed = finite_pass
    else:
        interior_pass = interior_rm_fraction >= GATE_MINIMUM_INTERIOR_RM_FRACTION
        passed = finite_pass and interior_pass
    print(f"\n  GATE {'PASSED' if passed else 'FAILED'}")
    if not finite_pass:
        print("    too many patterns produced non-finite features")
    if feature_method != "rapt" and not passed and finite_pass:
        print(
            "    too many Rm values are grid endpoints rather than "
            "measurements; raise k_r_max (ROADMAP section 8.3)"
        )

    return {
        "feature_method": feature_method,
        "finite_fraction": finite_fraction,
        "complete_row_fraction": complete_fraction,
        "interior_rm_fraction": interior_rm_fraction,
        "interior_rdm_fraction": float(interior[:, 1].mean()),
        "interior_rddm_fraction": float(interior[:, 2].mean()),
        "n_errored": len(failures),
        "passed": bool(passed),
    }


if __name__ == "__main__":
    main()
