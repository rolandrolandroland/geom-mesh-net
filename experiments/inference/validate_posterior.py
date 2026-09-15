"""Stage 3: simulation-based calibration and coverage. The decisive gate.

Stage 2 established that the flow beats the prior. That says it learned
something; it says nothing about whether its error bars are honest. A model can
beat the prior handsomely and still be systematically overconfident, and an
overconfident posterior is worse than no posterior, because it looks like an
answer.

Simulation-based calibration
----------------------------
For each held-out pattern, draw L samples from q(theta | s) and count how many
fall below the true theta. That count is the *rank* of the truth within the
posterior. If q is the true posterior, and theta was drawn from the prior and x
simulated from it -- which is exactly how scripts/generate_data.py built this dataset --
then those ranks are **uniform** on {0, ..., L} by construction.

That is the whole test, and its strength is that uniformity is not a threshold
anyone chose. Departures are diagnostic:

    peaked in the centre    posteriors too wide (underconfident)
    peaked at both edges    posteriors too narrow (OVERCONFIDENT)
    sloped                  biased

Why cross-validation
--------------------
Only data the flow never trained on is admissible, which in Stage 2 means 100
test patterns. A 100-point rank ECDF has a Kolmogorov band of about +/- 0.14,
wide enough to accept badly miscalibrated posteriors. Refitting over K folds
gives an out-of-fold posterior for every one of the 1,000 patterns and narrows
the band to about +/- 0.043, at a cost of a few seconds per fold.

Each fold is a different flow, so this measures the calibration of the
*procedure* rather than of one fitted model. That is the right target for a
method being proposed, and the single-model test-set result is reported
alongside as a consistency check.

Usage
-----
    python -m experiments.inference.validate_posterior
    python -m experiments.inference.validate_posterior --folds 5 --samples 499
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.inference.fit_posterior import load_stage1
from geom_mesh_net import paths
from geom_mesh_net.inference.calibration import (
    ECDF_ALPHA, central_interval_coverage, ecdf_deviation, kolmogorov_band,
    sbc_ranks, width_ratio,
)
from geom_mesh_net.inference.flow import BoxFlow, fit
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES, PRIOR_HIGH, PRIOR_LOW


COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)

# Pre-registered in ROADMAP section 6, Stage 3.
GATED_PARAMETERS = ("rho_c", "cr")
COVERAGE_TARGET_LEVEL = 0.9
COVERAGE_TOLERANCE = (0.85, 0.95)


def cross_validated_posteriors(
    features, theta, folds, n_samples, seed, verbose=True, ensemble=1
):
    """Out-of-fold posterior samples for every pattern.

    Every pattern is scored by a flow that never saw it. Within each fold a
    slice of the training portion is held back for early stopping, so the
    out-of-fold data is untouched by model selection as well as by fitting.

    With ``ensemble`` above one, each fold fits that many independently seeded
    flows and pools their draws in equal share, so the total number of draws per
    pattern is unchanged. That matters: SBC rank granularity depends on the draw
    count, so an ensemble drawing more samples than the single model it is
    compared against would produce an incomparable deviation. Section 8.7 records
    an earlier comparison that fell into exactly that trap.

    Ensembling widens the posterior by the disagreement between members, which is
    variance a single fit discards. Section 8.7 found that recovering it closes
    most of the overconfidence measured in Stage 3.
    """
    n = len(features)
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(n) % folds
    per_member = n_samples // ensemble
    total_draws = per_member * ensemble
    samples = np.zeros((n, total_draws, theta.shape[1]), dtype=np.float32)
    fold_log_probs = np.zeros(n, dtype=np.float64)

    for fold in range(folds):
        held_out = np.flatnonzero(assignment == fold)
        remaining = np.flatnonzero(assignment != fold)
        stop_size = max(50, len(remaining) // 10)
        stopping = remaining[:stop_size]
        training = remaining[stop_size:]

        mean = features[training].mean(axis=0)
        sd = features[training].std(axis=0)
        sd[sd == 0.0] = 1.0
        context = torch.tensor((features - mean) / sd, dtype=torch.float32)
        targets = torch.tensor(theta, dtype=torch.float32)

        member_draws, member_log_probs = [], []
        for member in range(ensemble):
            member_seed = seed + fold + 1000 * member
            torch.manual_seed(member_seed)
            model = BoxFlow(PRIOR_LOW, PRIOR_HIGH, context_dim=features.shape[1])
            fit(
                model,
                targets[training], context[training],
                targets[stopping], context[stopping],
                max_epochs=500, seed=member_seed, verbose=False,
            )
            model.eval()
            with torch.no_grad():
                member_draws.append(
                    model.sample(
                        context[held_out], n_samples=per_member,
                        generator=torch.Generator().manual_seed(
                            seed + 7000 + fold + 13 * member
                        ),
                    ).numpy()
                )
                member_log_probs.append(
                    model.log_prob(
                        targets[held_out], context[held_out]
                    ).numpy()
                )
        drawn = np.concatenate(member_draws, axis=1)
        # The pooled density is the mixture over members, so its log-density is
        # the log mean of theirs, not the mean of their logs.
        stacked = np.stack(member_log_probs)
        fold_log_probs[held_out] = (
            np.log(np.mean(np.exp(stacked - stacked.max(axis=0)), axis=0))
            + stacked.max(axis=0)
        )
        samples[held_out] = drawn
        if verbose:
            print(
                f"  fold {fold + 1}/{folds}: {len(training)} train, "
                f"{len(held_out)} held out, median out-of-fold log-density "
                f"{np.median(fold_log_probs[held_out]):7.3f}",
                flush=True,
            )
    return samples, fold_log_probs


# Added after Stage 3, motivated by section 8.7. Stage 3's gate let `cr` through
# at coverage 0.867 while its posterior was measurably too narrow, because
# coverage tolerates a narrow interval if the errors happen to be small. The
# width ratio measures over- and underconfidence directly. It is reported
# alongside the pre-registered criteria, which are left exactly as registered.
WIDTH_RATIO_TOLERANCE = (0.85, 1.15)


def report(samples, theta, label, n_samples, indent="  "):
    """Print SBC and coverage for one subset. Returns a dict of results."""
    ranks = sbc_ranks(samples, theta)
    band = kolmogorov_band(len(theta))
    results = {"n": int(len(theta)), "kolmogorov_band": band, "parameters": {}}

    print(f"\n{indent}Simulation-based calibration ({label}, n={len(theta)})")
    print(f"{indent}  uniform band at alpha={ECDF_ALPHA}: +/- {band:.4f}")
    print(
        f"{indent}{'param':>8} {'max ECDF dev':>13} {'verdict':>12} "
        f"{'mean rank':>11} {'(uniform = 0.500)':>18}"
    )
    for position, name in enumerate(PARAMETER_NAMES):
        deviation = ecdf_deviation(ranks[:, position], n_samples)
        normalized = (ranks[:, position] + 0.5) / (n_samples + 1)
        within = deviation <= band
        results["parameters"][name] = {
            "max_ecdf_deviation": deviation,
            "within_band": bool(within),
            "mean_normalized_rank": float(normalized.mean()),
        }
        print(
            f"{indent}{name:>8} {deviation:>13.4f} "
            f"{'UNIFORM' if within else 'DEPARTS':>12} "
            f"{normalized.mean():>11.3f}"
        )

    ratios = width_ratio(samples, theta, robust=True)
    fragile = width_ratio(samples, theta, robust=False)
    for position, name in enumerate(PARAMETER_NAMES):
        results["parameters"][name]["width_ratio"] = float(ratios[position])
        results["parameters"][name]["width_ratio_sd_based"] = float(
            fragile[position]
        )

    print(f"\n{indent}Credible-interval coverage")
    header = f"{indent}{'nominal':>8}" + "".join(f"{n:>10}" for n in PARAMETER_NAMES)
    print(header)
    coverage_table = {}
    for level in COVERAGE_LEVELS:
        empirical = central_interval_coverage(samples, theta, level)
        coverage_table[f"{level:.2f}"] = {
            name: float(empirical[i]) for i, name in enumerate(PARAMETER_NAMES)
        }
        print(
            f"{indent}{level:>8.0%}"
            + "".join(f"{value:>10.3f}" for value in empirical)
        )
    results["coverage"] = coverage_table

    print(f"\n{indent}Width ratio (residual spread / posterior spread; 1.0 honest)")
    print(f"{indent}{'':>14}" + "".join(f"{n:>10}" for n in PARAMETER_NAMES))
    print(
        f"{indent}{'robust':>14}"
        + "".join(f"{ratios[i]:>10.2f}" for i in range(len(PARAMETER_NAMES)))
    )
    print(
        f"{indent}{'sd-based':>14}"
        + "".join(f"{fragile[i]:>10.2f}" for i in range(len(PARAMETER_NAMES)))
    )
    gaps = [
        name for i, name in enumerate(PARAMETER_NAMES)
        if fragile[i] > 1.4 * max(ratios[i], 1e-9)
    ]
    if gaps:
        print(
            f"{indent}  sd-based far above robust for {', '.join(gaps)}: a few\n"
            f"{indent}  patterns carry very large errors. The robust row is the\n"
            f"{indent}  one to read for typical behaviour."
        )
    outside = [
        name for i, name in enumerate(PARAMETER_NAMES)
        if not WIDTH_RATIO_TOLERANCE[0] <= ratios[i] <= WIDTH_RATIO_TOLERANCE[1]
    ]
    if outside:
        print(
            f"{indent}  outside [{WIDTH_RATIO_TOLERANCE[0]}, "
            f"{WIDTH_RATIO_TOLERANCE[1]}]: {', '.join(outside)}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=paths.INFERENCE_DIR / "features" / "global_features.npz",
    )
    parser.add_argument(
        "--theta", type=Path, default=paths.THETA_PATH
    )
    parser.add_argument(
        "--descriptors", type=Path,
        default=paths.GROUND_TRUTH_DIR / "descriptors.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=paths.INFERENCE_DIR / "posterior")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ensemble", type=int, default=1,
        help="flows per fold; draws are split evenly so the total is unchanged",
    )
    args = parser.parse_args()

    features, theta, _, _ = load_stage1(args.features, args.theta)
    print(
        f"Stage 3: {args.folds}-fold cross-validated calibration over "
        f"{len(features)} patterns, {args.samples} posterior samples each"
    )
    if args.ensemble > 1:
        print(
            f"  ensemble of {args.ensemble} flows per fold, "
            f"{args.samples // args.ensemble} draws each"
        )
    print()
    started = time.perf_counter()
    samples, log_probs = cross_validated_posteriors(
        features, theta, args.folds, args.samples, args.seed,
        ensemble=args.ensemble,
    )
    print(f"\n  refitting took {time.perf_counter() - started:.1f} s")

    overall = report(samples, theta, "all patterns", args.samples)

    # Degenerate patterns have no clusters, so they carry no information about
    # cluster radius. Stage 2 showed one of them dominating a mean. Report both
    # ways so a structural outlier neither hides a real calibration failure nor
    # manufactures one.
    subset = None
    if args.descriptors.exists():
        with np.load(args.descriptors) as descriptors:
            degenerate = descriptors["degenerate"][: len(features)]
        if degenerate.any():
            keep = ~degenerate
            print(
                f"\n  {int(degenerate.sum())} zero-cluster patterns "
                f"({np.flatnonzero(degenerate).tolist()}); repeating without them"
            )
            subset = report(
                samples[keep], theta[keep], "excluding zero-cluster", args.samples
            )

    print("\n" + "=" * 72)
    print("Stage 3 gate (pre-registered, ROADMAP section 6)")
    print(
        f"  SBC rank ECDF within the uniform band, and {COVERAGE_TARGET_LEVEL:.0%} "
        f"coverage in [{COVERAGE_TOLERANCE[0]:.2f}, {COVERAGE_TOLERANCE[1]:.2f}],"
    )
    print(f"  for {' and '.join(GATED_PARAMETERS)}.\n")

    verdicts = {}
    for name in GATED_PARAMETERS:
        entry = overall["parameters"][name]
        level_key = f"{COVERAGE_TARGET_LEVEL:.2f}"
        empirical = overall["coverage"][level_key][name]
        sbc_ok = entry["within_band"]
        coverage_ok = COVERAGE_TOLERANCE[0] <= empirical <= COVERAGE_TOLERANCE[1]
        verdicts[name] = {
            "sbc_within_band": bool(sbc_ok),
            "coverage_90": empirical,
            "coverage_within_tolerance": bool(coverage_ok),
            "passed": bool(sbc_ok and coverage_ok),
        }
        print(
            f"  {name:>7}: SBC {'pass' if sbc_ok else 'FAIL'} "
            f"(dev {entry['max_ecdf_deviation']:.4f} vs band "
            f"{overall['kolmogorov_band']:.4f}), "
            f"coverage {empirical:.3f} {'pass' if coverage_ok else 'FAIL'}"
        )

    # rb is gated on calibration only, never on sharpness. An honest posterior
    # for an uninformative parameter is a wide one.
    rb = overall["parameters"].get("rb")
    if rb is not None:
        print(
            f"\n  {'rb':>7}: calibration-only (section 5). SBC "
            f"{'pass' if rb['within_band'] else 'FAIL'} "
            f"(dev {rb['max_ecdf_deviation']:.4f}), "
            f"coverage {overall['coverage']['0.90']['rb']:.3f}"
        )
        print(
            "           Width is not gated here: for a parameter the data do not\n"
            "           constrain, a wide posterior is the correct answer."
        )

    passed = all(v["passed"] for v in verdicts.values())
    print(f"\n  GATE {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print(
            "\n  Remedies in order (ROADMAP section 6, Stage 3): more simulations,\n"
            "  observation augmentation, a larger flow, then reconsider the summary\n"
            "  statistic. Do not proceed to Stage 4 with an uncalibrated posterior."
        )
    print("=" * 72)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "calibration.npz",
        ranks=sbc_ranks(samples, theta).astype(np.int32),
        out_of_fold_log_prob=log_probs,
        parameter_names=np.array(PARAMETER_NAMES),
        n_posterior_samples=args.samples,
    )
    (args.output_dir / "calibration.json").write_text(
        json.dumps(
            {
                "stage": 3,
                "folds": args.folds,
                "posterior_samples": args.samples,
                "ensemble": args.ensemble,
                "width_ratio_tolerance": list(WIDTH_RATIO_TOLERANCE),
                "seed": args.seed,
                "coverage_levels": list(COVERAGE_LEVELS),
                "gated_parameters": list(GATED_PARAMETERS),
                "all_patterns": overall,
                "excluding_degenerate": subset,
                "gate": {"parameters": verdicts, "passed": bool(passed)},
                "median_out_of_fold_log_prob": float(np.median(log_probs)),
            },
            indent=2,
        )
        + "\n"
    )
    plot_path = args.output_dir / "calibration.png"
    try:
        make_plot(samples, theta, args.samples, plot_path)
        print(f"\nwrote {plot_path}")
    except Exception as exc:  # noqa: BLE001 - a figure is not worth failing over
        print(f"\n(figure skipped: {exc})")
    print(f"wrote {args.output_dir}/calibration.json")


def make_plot(samples, theta, n_samples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sbc_ranks(samples, theta)
    band = kolmogorov_band(len(theta))
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))

    for position, name in enumerate(PARAMETER_NAMES):
        normalized = np.sort((ranks[:, position] + 0.5) / (n_samples + 1))
        empirical = np.arange(1, len(normalized) + 1) / len(normalized)

        top = axes[0, position]
        top.plot(normalized, empirical - normalized, lw=1.4, color="#1f77b4")
        top.axhline(0.0, color="grey", lw=0.8)
        top.fill_between([0, 1], -band, band, color="grey", alpha=0.2,
                         label=f"95% band +/-{band:.3f}")
        top.set_title(f"{name}: SBC rank ECDF difference")
        top.set_xlabel("normalized rank")
        top.set_ylabel("ECDF - uniform")
        top.set_ylim(-max(0.12, band * 2.5), max(0.12, band * 2.5))
        if position == 0:
            top.legend(fontsize=8)

        bottom = axes[1, position]
        levels = np.linspace(0.05, 0.99, 40)
        empirical_coverage = [
            central_interval_coverage(samples, theta, level)[position]
            for level in levels
        ]
        bottom.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="ideal")
        bottom.plot(levels, empirical_coverage, lw=1.6, color="#d62728")
        bottom.set_title(f"{name}: coverage")
        bottom.set_xlabel("nominal")
        bottom.set_ylabel("empirical")
        bottom.set_xlim(0, 1)
        bottom.set_ylim(0, 1)
        if position == 0:
            bottom.legend(fontsize=8)

    figure.suptitle(
        "Stage 3: simulation-based calibration and coverage "
        "(cross-validated, out-of-fold)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
