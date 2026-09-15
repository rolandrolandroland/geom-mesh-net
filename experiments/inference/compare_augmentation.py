"""Does observation augmentation improve calibration?

Compares two models fitted on the same thinned observations, differing only in
how many observations per pattern the training set contains:

    control    one thinning per pattern
    augmented  every thinning per pattern

Both are scored on the *same* held-out observations, so the comparison isolates
the augmentation from the thinning itself. Comparing an augmented model against
the full-cloud Stage 1 features would confound the two.

Splitting is by ``pattern_index``, never by row. Replicates of one pattern share
a theta, so a row-wise split would put near-duplicates of a training example into
the held-out set and every calibration number would be optimistic. This is the
caveat ROADMAP section 7 flags, and it is the one thing here that would fail
silently.

Evaluation uses one replicate per held-out pattern. Using several would put
multiple ranks from the same theta into a simulation-based calibration histogram,
and those ranks are not independent.

Usage
-----
    python -m experiments.inference.compare_augmentation
    python -m experiments.inference.compare_augmentation --folds 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geom_mesh_net import paths
from geom_mesh_net.inference.calibration import (
    central_interval_coverage, ecdf_deviation, kolmogorov_band, sbc_ranks,
)
from geom_mesh_net.inference.flow import BoxFlow, fit
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES, PRIOR_HIGH, PRIOR_LOW


def load_augmented(path, theta_path):
    with np.load(path, allow_pickle=False) as cached:
        values = cached["values"].astype(np.float64)
        pattern_index = cached["pattern_index"]
        replicate = cached["replicate"]
        names = [str(n) for n in cached["feature_names"]]
    theta_all = np.load(theta_path)
    finite = np.all(np.isfinite(values), axis=1)
    if not finite.all():
        print(f"dropping {(~finite).sum()} rows with non-finite features")
    return (
        values[finite], pattern_index[finite], replicate[finite],
        theta_all, names,
    )


def run_fold(
    values, pattern_index, replicate, theta_all,
    train_patterns, held_out_patterns, use_all_replicates, seed, n_samples,
):
    """Fit on the training patterns, return posterior samples for held-out ones."""
    train_mask = np.isin(pattern_index, train_patterns)
    if not use_all_replicates:
        train_mask &= replicate == 0
    # One observation per held-out pattern keeps SBC ranks independent.
    evaluate_mask = np.isin(pattern_index, held_out_patterns) & (replicate == 0)

    # A slice of training *patterns* is held back for early stopping, so model
    # selection never touches the evaluation patterns either.
    stop_patterns = train_patterns[: max(40, len(train_patterns) // 10)]
    fit_patterns = train_patterns[len(stop_patterns):]
    fit_mask = train_mask & np.isin(pattern_index, fit_patterns)
    stop_mask = train_mask & np.isin(pattern_index, stop_patterns)

    mean = values[fit_mask].mean(axis=0)
    sd = values[fit_mask].std(axis=0)
    sd[sd == 0.0] = 1.0
    context = torch.tensor((values - mean) / sd, dtype=torch.float32)
    targets = torch.tensor(theta_all[pattern_index], dtype=torch.float32)

    torch.manual_seed(seed)
    model = BoxFlow(PRIOR_LOW, PRIOR_HIGH, context_dim=values.shape[1])
    fit(
        model,
        targets[fit_mask], context[fit_mask],
        targets[stop_mask], context[stop_mask],
        max_epochs=500, seed=seed, verbose=False,
    )
    model.eval()
    with torch.no_grad():
        samples = model.sample(
            context[evaluate_mask], n_samples=n_samples,
            generator=torch.Generator().manual_seed(seed + 5150),
        ).numpy()
        log_probs = model.log_prob(
            targets[evaluate_mask], context[evaluate_mask]
        ).numpy()
    return samples, log_probs, pattern_index[evaluate_mask], int(fit_mask.sum())


def cross_validate(
    values, pattern_index, replicate, theta_all,
    use_all_replicates, folds, seed, n_samples, label,
):
    patterns = np.unique(pattern_index)
    assignment = np.random.default_rng(seed).permutation(len(patterns)) % folds
    collected_samples, collected_patterns, collected_log_probs = [], [], []
    rows_used = 0

    for fold in range(folds):
        held_out = patterns[assignment == fold]
        train = patterns[assignment != fold]
        samples, log_probs, evaluated, n_rows = run_fold(
            values, pattern_index, replicate, theta_all,
            train, held_out, use_all_replicates, seed + fold, n_samples,
        )
        collected_samples.append(samples)
        collected_patterns.append(evaluated)
        collected_log_probs.append(log_probs)
        rows_used = n_rows
        print(f"    {label} fold {fold + 1}/{folds}: {n_rows} training rows", flush=True)

    samples = np.concatenate(collected_samples)
    evaluated = np.concatenate(collected_patterns)
    return samples, theta_all[evaluated], np.concatenate(collected_log_probs), rows_used


def summarize(samples, theta, log_probs, n_samples, rows_used, label):
    ranks = sbc_ranks(samples, theta)
    band = kolmogorov_band(len(theta))
    prior_sd = (PRIOR_HIGH - PRIOR_LOW) / np.sqrt(12.0)
    summary = {
        "training_rows_per_fold": rows_used,
        "median_log_prob": float(np.median(log_probs)),
        "kolmogorov_band": band,
        "parameters": {},
    }
    for position, name in enumerate(PARAMETER_NAMES):
        deviation = ecdf_deviation(ranks[:, position], n_samples)
        summary["parameters"][name] = {
            "max_ecdf_deviation": deviation,
            "within_band": bool(deviation <= band),
            "coverage_90": float(
                central_interval_coverage(samples, theta, 0.9)[position]
            ),
            "contraction": float(
                1.0 - samples[:, :, position].std(axis=1).mean() / prior_sd[position]
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=paths.INFERENCE_DIR / "features" / "augmented_features.npz",
    )
    parser.add_argument(
        "--theta", type=Path, default=paths.THETA_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=paths.INFERENCE_DIR / "posterior")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    values, pattern_index, replicate, theta_all, _ = load_augmented(
        args.features, args.theta
    )
    n_replicates = int(replicate.max()) + 1
    print(
        f"Augmentation comparison: {len(np.unique(pattern_index))} patterns x "
        f"{n_replicates} observations = {len(values)} rows\n"
    )

    started = time.perf_counter()
    outcomes = {}
    for label, use_all in (("control", False), ("augmented", True)):
        samples, theta, log_probs, rows = cross_validate(
            values, pattern_index, replicate, theta_all,
            use_all, args.folds, args.seed, args.samples, label,
        )
        outcomes[label] = summarize(
            samples, theta, log_probs, args.samples, rows, label
        )
        print()
    print(f"  {time.perf_counter() - started:.0f} s total\n")

    control, augmented = outcomes["control"], outcomes["augmented"]
    print("=" * 76)
    print(
        f"Training rows per fold: control {control['training_rows_per_fold']}, "
        f"augmented {augmented['training_rows_per_fold']} "
        f"({augmented['training_rows_per_fold'] / control['training_rows_per_fold']:.1f}x)"
    )
    print(
        f"Median held-out log-likelihood: control "
        f"{control['median_log_prob']:.3f}, augmented "
        f"{augmented['median_log_prob']:.3f} "
        f"({augmented['median_log_prob'] - control['median_log_prob']:+.3f})"
    )
    print(f"\nSBC deviation (band +/-{control['kolmogorov_band']:.4f}) and 90% coverage")
    header = (
        f"{'param':>7}{'ctrl dev':>10}{'aug dev':>10}{'change':>9}"
        f"{'ctrl cov':>10}{'aug cov':>9}{'ctrl contr':>12}{'aug contr':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in PARAMETER_NAMES:
        c, a = control["parameters"][name], augmented["parameters"][name]
        print(
            f"{name:>7}{c['max_ecdf_deviation']:>10.4f}"
            f"{a['max_ecdf_deviation']:>10.4f}"
            f"{a['max_ecdf_deviation'] - c['max_ecdf_deviation']:>+9.4f}"
            f"{c['coverage_90']:>10.3f}{a['coverage_90']:>9.3f}"
            f"{c['contraction']:>12.3f}{a['contraction']:>11.3f}"
        )

    improved = [
        name for name in PARAMETER_NAMES
        if augmented["parameters"][name]["max_ecdf_deviation"]
        < control["parameters"][name]["max_ecdf_deviation"]
    ]
    print(
        f"\n  SBC deviation improved for {len(improved)}/4 parameters"
        + (f": {', '.join(improved)}" if improved else "")
    )
    print(
        "\n  Read this against the within-pattern feature spread reported by\n"
        "  augment_features.py. If independent observations of one structure barely\n"
        "  move the features, the replicates are near-duplicates and augmentation\n"
        "  cannot add much -- and a posterior that is narrow is then narrow\n"
        "  correctly, meaning any miscalibration comes from somewhere else."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "augmentation_comparison.json"
    path.write_text(
        json.dumps(
            {
                "folds": args.folds,
                "posterior_samples": args.samples,
                "seed": args.seed,
                "n_replicates": n_replicates,
                "control": control,
                "augmented": augmented,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
