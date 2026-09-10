"""Screen the Stage 1 features for information about theta, before Stage 2.

A conditional flow is the deliverable, but it is worth knowing first whether the
features carry any information about each parameter at all. Ridge regression is a
cheap lower bound: it produces only a point estimate, with none of the posterior
shape or calibrated uncertainty that motivates the flow, but a parameter that
ridge cannot predict at all will not be rescued by a more flexible model. It
either has no signal in these 14 numbers or it has none in the point pattern.

This exists to make the Stage 3 predictions in ROADMAP section 5 falsifiable.
They were written down before any model was fitted; this is where they get
checked.

Usage
-----
    PYTHONPATH=. python inference/screen_features.py
    PYTHONPATH=. python inference/screen_features.py --output-json out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from inference.recover_ground_truth import PARAMETER_NAMES


RIDGE_PENALTY = 1e-3
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 0
# A single 200-pattern holdout gives an R^2 that moves by more than 0.1 between
# splits, which is larger than the effect being tested for. Repeated splits turn
# the estimate into a distribution so a near-zero R^2 can be distinguished from
# a small positive one instead of landing arbitrarily either side of a
# threshold.
N_SPLITS = 25

# Registered in ROADMAP section 5 before any model was fitted. rb is expected to
# be unidentifiable: it governs the spread of cluster radii, and with a median of
# 8 clusters per pattern there is almost no sample from which to estimate a
# spread.
PREDICTED_UNINFORMATIVE = ("rb",)
UNINFORMATIVE_R2_CEILING = 0.10


def ridge_fit(design, targets, penalty=RIDGE_PENALTY):
    scale = penalty * len(design)
    gram = design.T @ design + scale * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def r_squared(truth, prediction, baseline):
    """R^2 against predicting the training mean. Zero means no better."""
    total = ((truth - baseline) ** 2).sum()
    residual = ((truth - prediction) ** 2).sum()
    return 1.0 - residual / total if total > 0 else float("nan")


def screen_one_split(features, theta, order, n_test):
    test, train = order[:n_test], order[n_test:]

    mean = features[train].mean(axis=0)
    sd = features[train].std(axis=0)
    sd[sd == 0.0] = 1.0
    standardized = (features - mean) / sd

    def linear_design(block):
        return np.column_stack([np.ones(len(block)), block])

    def quadratic_design(block):
        return np.column_stack([np.ones(len(block)), block, block**2])

    rows = {}
    for position, name in enumerate(PARAMETER_NAMES):
        target = theta[:, position]
        baseline = target[train].mean()
        scores = {}
        for label, build in (
            ("linear", linear_design),
            ("quadratic", quadratic_design),
        ):
            weights = ridge_fit(build(standardized[train]), target[train])
            prediction = build(standardized[test]) @ weights
            scores[label] = float(r_squared(target[test], prediction, baseline))
            if label == "quadratic":
                scores["residual_sd"] = float((target[test] - prediction).std())
        scores["prior_sd"] = float(target[train].std())
        scores["best_r2"] = max(scores["linear"], scores["quadratic"])
        rows[name] = scores
    return rows


def screen(features, theta, seed=SPLIT_SEED, n_splits=N_SPLITS):
    """Repeat the holdout screen over independent splits.

    Returns per-parameter summaries carrying the mean and spread of each score
    across splits, so a verdict rests on the distribution rather than on one
    partition.
    """
    n_test = int(round(len(features) * HOLDOUT_FRACTION))
    rng = np.random.default_rng(seed)
    per_split = [
        screen_one_split(features, theta, rng.permutation(len(features)), n_test)
        for _ in range(n_splits)
    ]

    summary = {}
    for name in PARAMETER_NAMES:
        collected = {
            key: np.array([split[name][key] for split in per_split])
            for key in per_split[0][name]
        }
        best = np.maximum(collected["linear"], collected["quadratic"])
        summary[name] = {
            "linear_mean": float(collected["linear"].mean()),
            "quadratic_mean": float(collected["quadratic"].mean()),
            "best_r2_mean": float(best.mean()),
            "best_r2_sd": float(best.std(ddof=1)),
            "best_r2_min": float(best.min()),
            "best_r2_max": float(best.max()),
            "residual_sd_mean": float(collected["residual_sd"].mean()),
            "prior_sd_mean": float(collected["prior_sd"].mean()),
        }
    return summary, len(features) - n_test, n_test, n_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("inference/features/global_features.npz"),
    )
    parser.add_argument(
        "--theta", type=Path, default=Path("inference/ground_truth/theta.npy")
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=False) as cached:
        features = cached["values"].astype(np.float64)
        feature_names = [str(name) for name in cached["feature_names"]]
        interior = cached["k_extrema_interior"]
    theta = np.load(args.theta)[: len(features)]

    usable = np.all(np.isfinite(features), axis=1)
    if not usable.all():
        print(f"dropping {(~usable).sum()} patterns with non-finite features")
        features, theta, interior = features[usable], theta[usable], interior[usable]

    rows, n_train, n_test, n_splits = screen(features, theta)

    print(
        f"Feature screen: {len(features)} patterns, "
        f"{n_train} train / {n_test} test, {n_splits} random splits"
    )
    print("Ridge R^2 on held-out patterns. 0.0 = no better than the prior mean.\n")
    header = (
        f"{'param':>7} {'linear':>8} {'quadratic':>10} "
        f"{'best R^2 (mean +/- sd)':>24} {'range':>16} {'contraction':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, scores in rows.items():
        contraction = 1.0 - scores["residual_sd_mean"] / scores["prior_sd_mean"]
        spread = f"{scores['best_r2_mean']:+.3f} +/- {scores['best_r2_sd']:.3f}"
        span = f"{scores['best_r2_min']:+.2f}..{scores['best_r2_max']:+.2f}"
        print(
            f"{name:>7} {scores['linear_mean']:>8.3f} "
            f"{scores['quadratic_mean']:>10.3f} {spread:>24} {span:>16} "
            f"{contraction:>12.3f}"
        )

    print("\nWhich feature family carries each parameter (largest |correlation|):")
    for position, name in enumerate(PARAMETER_NAMES):
        correlations = np.array(
            [np.corrcoef(features[:, j], theta[:, position])[0, 1]
             for j in range(features.shape[1])]
        )
        top = np.argsort(-np.abs(correlations))[:3]
        detail = ", ".join(f"{feature_names[j]} {correlations[j]:+.2f}" for j in top)
        print(f"  {name:>7}: {detail}")

    print("\nRegistered prediction (ROADMAP section 5), checked here:")
    verdicts = {}
    for name in PREDICTED_UNINFORMATIVE:
        scores = rows[name]
        mean, sd = scores["best_r2_mean"], scores["best_r2_sd"]
        # Judge the distribution, not one split. The ceiling has to sit above the
        # mean by more than the split-to-split spread for the call to mean
        # anything.
        held = mean + sd <= UNINFORMATIVE_R2_CEILING
        marginal = mean <= UNINFORMATIVE_R2_CEILING < mean + sd
        verdicts[name] = bool(held)
        if held:
            status = "HELD"
        elif marginal:
            status = "MARGINAL (mean under the ceiling, spread crosses it)"
        else:
            status = "VIOLATED"
        print(
            f"  {name} predicted uninformative (R^2 <= "
            f"{UNINFORMATIVE_R2_CEILING:.2f})"
        )
        print(
            f"    measured R^2 {mean:+.3f} +/- {sd:.3f} over {n_splits} splits "
            f"-> {status}"
        )
        print(
            f"    residual sd {scores['residual_sd_mean']:.4f} against a prior sd "
            f"of {scores['prior_sd_mean']:.4f} "
            f"(contraction {1 - scores['residual_sd_mean'] / scores['prior_sd_mean']:.3f})"
        )

    if all(verdicts.values()):
        print(
            "\n  A calibrated posterior for these must return approximately the\n"
            "  prior. If Stage 3 yields a narrow one, the flow is overconfident\n"
            "  and simulation-based calibration must catch it."
        )
    else:
        print(
            "\n  Not cleanly held. Record the outcome in ROADMAP section 5 rather\n"
            "  than adjusting the ceiling -- it was registered in advance so that\n"
            "  it could fail. A small but nonzero R^2 still means Stage 3 should\n"
            "  produce a posterior only slightly narrower than the prior, and SBC\n"
            "  remains the arbiter of whether that narrowing is honest."
        )

    informative = [n for n, s in rows.items() if s["best_r2_mean"] > 0.5]
    print(
        f"\nStage 2 should be able to fit {len(informative)} of "
        f"{len(PARAMETER_NAMES)} parameters: {', '.join(informative)}"
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "n_patterns": int(len(features)),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_splits": n_splits,
                    "ridge_penalty": RIDGE_PENALTY,
                    "uninformative_r2_ceiling": UNINFORMATIVE_R2_CEILING,
                    "scores": rows,
                    "predicted_uninformative": list(PREDICTED_UNINFORMATIVE),
                    "predictions_held": verdicts,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
