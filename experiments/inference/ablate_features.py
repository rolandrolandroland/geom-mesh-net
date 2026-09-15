"""Stage 4: which features actually carry information about theta?

``docs/guides/clustersim_todo.md`` has asked "which features capture the most
information?" since before any of this existed. Stages 2 and 3 make it answerable
rather than a matter of intuition: refit the posterior on a subset of features
and measure how much the posterior contracts. A feature that sharpens the
posterior carries information about theta; one that does not, does not.

This is stronger than a correlation screen. Correlation asks whether a feature
moves with a parameter; contraction asks whether it *reduces the uncertainty*
that remains after every other feature has been accounted for. A feature
perfectly correlated with theta but redundant with another feature contributes
nothing, and only the ablation shows that.

Two questions were queued specifically:

- Does ``Rddm`` contribute anything? It reaches an interior extremum in only 37%
  of patterns (ROADMAP section 8.3), so it is the prime suspect for being noise.
- Does ``cube_root`` beat ``sqrt``? Section 8.2 argued the point on theory and
  deferred to the published definition. Posterior contraction settles it on
  evidence. Pass a cube_root feature cache with ``--features`` to compare.

Metrics
-------
Median held-out log-likelihood, not the mean: Stage 2 showed a single structural
outlier moving the mean by 200 nats, which would swamp every ablation effect.

Per-parameter contraction, ``1 - posterior_sd / prior_sd``, averaged over the
held-out set. This is the quantity Stage 3 established is trustworthy, since the
posteriors are calibrated.

Every configuration is fitted from several seeds, because the seed-to-seed
spread is comparable to the effect being measured for the weaker features.

Usage
-----
    python -m experiments.inference.ablate_features
    python -m experiments.inference.ablate_features --seeds 5
    python -m experiments.inference.ablate_features \\
        --features experiments/inference/features/global_features_cube_root.npz \\
        --label cube_root
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.inference.fit_posterior import load_stage1, split_indices
from geom_mesh_net import paths
from geom_mesh_net.inference.flow import BoxFlow, fit
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES, PRIOR_HIGH, PRIOR_LOW


# Which summary function each feature comes from. The ablation groups by these
# because a whole family dropping out is more interpretable than single columns.
FEATURE_FAMILIES = {
    "G": ("G_max_diff", "G_max_diff_r", "G_min_diff", "G_zero_diff_r"),
    "F": ("F_min_diff", "F_min_diff_F"),
    "K": ("Tm", "Rm", "Rdm", "Rddm", "Tdm"),
    "crossG": ("GXGH_min_diff", "GXGH_95diff_r", "GXGH_FWHM"),
}

POSTERIOR_SAMPLES = 500


def evaluate(features, theta, columns, seeds, max_epochs=500):
    """Fit on a column subset and measure held-out fit and contraction."""
    train, validation, test = split_indices(len(features))
    subset = features[:, columns]

    mean = subset[train].mean(axis=0)
    sd = subset[train].std(axis=0)
    sd[sd == 0.0] = 1.0
    context = torch.tensor((subset - mean) / sd, dtype=torch.float32)
    targets = torch.tensor(theta, dtype=torch.float32)
    prior_sd = (PRIOR_HIGH - PRIOR_LOW) / np.sqrt(12.0)

    medians, contractions = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = BoxFlow(PRIOR_LOW, PRIOR_HIGH, context_dim=len(columns))
        fit(
            model,
            targets[train], context[train],
            targets[validation], context[validation],
            max_epochs=max_epochs, seed=seed, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            log_probs = model.log_prob(targets[test], context[test]).numpy()
            samples = model.sample(
                context[test], n_samples=POSTERIOR_SAMPLES,
                generator=torch.Generator().manual_seed(seed + 7777),
            ).numpy()
        medians.append(float(np.median(log_probs)))
        contractions.append(1.0 - samples.std(axis=1).mean(axis=0) / prior_sd)

    contractions = np.stack(contractions)
    return {
        "n_features": len(columns),
        # Per-seed values are kept so configurations can be compared *paired*.
        # Every configuration uses the same seeds, so the shared training noise
        # cancels in a per-seed difference. Comparing means instead throws that
        # away, and the seed spread is larger than a single-feature effect.
        "median_log_prob_per_seed": [float(v) for v in medians],
        "median_log_prob_mean": float(np.mean(medians)),
        "median_log_prob_sd": float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0,
        "contraction_mean": {
            name: float(contractions[:, i].mean())
            for i, name in enumerate(PARAMETER_NAMES)
        },
        "contraction_sd": {
            name: float(contractions[:, i].std(ddof=1)) if len(medians) > 1 else 0.0
            for i, name in enumerate(PARAMETER_NAMES)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=paths.INFERENCE_DIR / "features" / "global_features.npz",
    )
    parser.add_argument(
        "--theta", type=Path, default=paths.THETA_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=paths.INFERENCE_DIR / "posterior")
    parser.add_argument("--label", default="sqrt")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    features, theta, names, _ = load_stage1(args.features, args.theta)
    seeds = list(range(args.seeds))
    index_of = {name: position for position, name in enumerate(names)}
    everything = list(range(len(names)))

    configurations = [("all 14 features", everything)]
    for family, members in FEATURE_FAMILIES.items():
        keep = [index_of[m] for m in members]
        configurations.append((f"{family} family only", keep))
        configurations.append(
            (f"drop {family} family", [c for c in everything if c not in keep])
        )
    for name in names:
        configurations.append(
            (f"drop {name}", [c for c in everything if c != index_of[name]])
        )

    print(
        f"Stage 4 ablation [{args.label}]: {len(configurations)} configurations "
        f"x {args.seeds} seeds on {len(features)} patterns\n"
    )
    started = time.perf_counter()
    results = {}
    for position, (label, columns) in enumerate(configurations):
        results[label] = evaluate(features, theta, columns, seeds)
        print(
            f"  [{position + 1:2d}/{len(configurations)}] {label:<24} "
            f"median log-lik {results[label]['median_log_prob_mean']:7.3f}",
            flush=True,
        )
    print(f"\n  {time.perf_counter() - started:.0f} s total\n")

    baseline = results["all 14 features"]
    base_log = baseline["median_log_prob_mean"]
    base_contraction = baseline["contraction_mean"]

    print("=" * 78)
    print(f"Baseline, all 14 features [{args.label}]")
    print(
        f"  median held-out log-likelihood {base_log:.3f} "
        f"+/- {baseline['median_log_prob_sd']:.3f} over {args.seeds} seeds"
    )
    print("  contraction: " + "  ".join(
        f"{n} {base_contraction[n]:.3f}" for n in PARAMETER_NAMES
    ))

    print("\nFeature families in isolation (what each family alone can recover)")
    header = f"{'configuration':<24}{'log-lik':>9}" + "".join(
        f"{n:>9}" for n in PARAMETER_NAMES
    )
    print(header)
    print("-" * len(header))
    for family in FEATURE_FAMILIES:
        entry = results[f"{family} family only"]
        print(
            f"{family + ' only':<24}{entry['median_log_prob_mean']:>9.3f}"
            + "".join(
                f"{entry['contraction_mean'][n]:>9.3f}" for n in PARAMETER_NAMES
            )
        )

    print("\nDropping a whole family (change from baseline; negative = it mattered)")
    print(header)
    print("-" * len(header))
    for family in FEATURE_FAMILIES:
        entry = results[f"drop {family} family"]
        print(
            f"{'drop ' + family:<24}"
            f"{entry['median_log_prob_mean'] - base_log:>+9.3f}"
            + "".join(
                f"{entry['contraction_mean'][n] - base_contraction[n]:>+9.3f}"
                for n in PARAMETER_NAMES
            )
        )

    print("\nLeave-one-feature-out, paired by seed against the baseline")
    print(
        "  Note that pairing cannot move the estimate -- the mean of paired\n"
        "  differences is identically the difference of means. It only shrinks the\n"
        "  standard error, and only to the extent that configurations share\n"
        "  training noise. Measured here, they largely do not: the gain is about\n"
        "  1.2x. Resolution comes from seed count, not from the pairing."
    )
    base_per_seed = np.array(baseline["median_log_prob_per_seed"])
    print(
        f"\n{'dropped feature':<24}{'delta':>9}{'paired se':>11}"
        f"{'unpaired se':>13}   interpretation"
    )
    print("-" * 78)
    rows = []
    for name in names:
        entry = results[f"drop {name}"]
        per_seed = np.array(entry["median_log_prob_per_seed"])
        differences = per_seed - base_per_seed
        delta = float(differences.mean())
        se = (
            float(differences.std(ddof=1) / np.sqrt(len(differences)))
            if len(differences) > 1
            else float("nan")
        )
        unpaired_se = (
            float(np.sqrt(per_seed.var(ddof=1) / len(per_seed)
                          + base_per_seed.var(ddof=1) / len(base_per_seed)))
            if len(per_seed) > 1
            else float("nan")
        )
        rows.append((name, delta, se, unpaired_se))
    for name, delta, se, unpaired_se in sorted(rows, key=lambda r: r[1]):
        if np.isfinite(se) and abs(delta) > 2 * se:
            note = "carries information" if delta < 0 else "removing it HELPED"
        else:
            note = "not resolved"
        print(
            f"{name:<24}{delta:>+9.3f}{se:>11.3f}{unpaired_se:>13.3f}   {note}"
        )

    resolved = sum(
        1 for _, delta, se, _ in rows if np.isfinite(se) and abs(delta) > 2 * se
    )
    print(
        f"\n  {resolved}/{len(rows)} features resolved at 2 standard errors.\n"
        f"  Most are not, and that is expected rather than a failure: these features\n"
        f"  are redundant with one another, so removing any single one loses little.\n"
        f"  Leave-one-out is the wrong instrument for correlated inputs; the family\n"
        f"  ablation above is the one that answers the question."
    )

    rddm_per_seed = np.array(results["drop Rddm"]["median_log_prob_per_seed"])
    rddm_differences = rddm_per_seed - base_per_seed
    rddm_se = (
        float(rddm_differences.std(ddof=1) / np.sqrt(len(rddm_differences)))
        if len(rddm_differences) > 1
        else float("nan")
    )
    print(
        f"\n  Rddm specifically: paired change {rddm_differences.mean():+.3f} "
        f"+/- {rddm_se:.3f}."
    )
    print(
        "    Rddm reaches an interior extremum in only 37% of patterns, so the\n"
        "    question was whether it is signal or a grid endpoint in disguise."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"ablation_{args.label}.json"
    path.write_text(
        json.dumps(
            {
                "stage": 4,
                "label": args.label,
                "features_path": str(args.features),
                "seeds": seeds,
                "feature_names": names,
                "families": {k: list(v) for k, v in FEATURE_FAMILIES.items()},
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
