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
simulated from it -- which is exactly how data_factory.py built this dataset --
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
    PYTHONPATH=. python inference/validate_posterior.py
    PYTHONPATH=. python inference/validate_posterior.py --folds 5 --samples 499
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from inference.fit_posterior import PRIOR_HIGH, PRIOR_LOW, load_stage1
from inference.flow import BoxFlow, fit
from inference.recover_ground_truth import PARAMETER_NAMES


COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)

# Pre-registered in ROADMAP section 6, Stage 3.
GATED_PARAMETERS = ("rho_c", "cr")
COVERAGE_TARGET_LEVEL = 0.9
COVERAGE_TOLERANCE = (0.85, 0.95)
ECDF_ALPHA = 0.05


def kolmogorov_band(n, alpha=ECDF_ALPHA):
    """Simultaneous band on an ECDF, from the DKW inequality.

    P(sup|F_n - F| > eps) <= 2 exp(-2 n eps^2), so the band is
    sqrt(ln(2/alpha) / (2n)). It is a *simultaneous* bound, so a single
    excursion anywhere is already evidence against uniformity.
    """
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def sbc_ranks(samples, truth):
    """Rank of each true value within its posterior samples.

    samples: (n, L, d).  truth: (n, d).  Returns (n, d) in {0, ..., L}.
    """
    return (samples < truth[:, None, :]).sum(axis=1)


def ecdf_deviation(ranks, n_samples):
    """Maximum absolute deviation of the rank ECDF from uniform."""
    uniform = np.sort((ranks + 0.5) / (n_samples + 1))
    n = len(uniform)
    empirical = np.arange(1, n + 1) / n
    below = np.abs(empirical - uniform).max()
    above = np.abs(uniform - np.arange(0, n) / n).max()
    return float(max(below, above))


def central_interval_coverage(samples, truth, level):
    """Fraction of patterns whose truth lies in the central `level` interval."""
    lower = np.quantile(samples, (1 - level) / 2, axis=1)
    upper = np.quantile(samples, 1 - (1 - level) / 2, axis=1)
    return ((truth >= lower) & (truth <= upper)).mean(axis=0)


def cross_validated_posteriors(features, theta, folds, n_samples, seed, verbose=True):
    """Out-of-fold posterior samples for every pattern.

    Every pattern is scored by a flow that never saw it. Within each fold a
    slice of the training portion is held back for early stopping, so the
    out-of-fold data is untouched by model selection as well as by fitting.
    """
    n = len(features)
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(n) % folds
    samples = np.zeros((n, n_samples, theta.shape[1]), dtype=np.float32)
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

        torch.manual_seed(seed + fold)
        model = BoxFlow(PRIOR_LOW, PRIOR_HIGH, context_dim=features.shape[1])
        fit(
            model,
            targets[training], context[training],
            targets[stopping], context[stopping],
            max_epochs=500, seed=seed + fold, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            drawn = model.sample(
                context[held_out], n_samples=n_samples,
                generator=torch.Generator().manual_seed(seed + 1000 + fold),
            ).numpy()
            fold_log_probs[held_out] = model.log_prob(
                targets[held_out], context[held_out]
            ).numpy()
        samples[held_out] = drawn
        if verbose:
            print(
                f"  fold {fold + 1}/{folds}: {len(training)} train, "
                f"{len(held_out)} held out, median out-of-fold log-density "
                f"{np.median(fold_log_probs[held_out]):7.3f}",
                flush=True,
            )
    return samples, fold_log_probs


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
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("inference/features/global_features.npz"),
    )
    parser.add_argument(
        "--theta", type=Path, default=Path("inference/ground_truth/theta.npy")
    )
    parser.add_argument(
        "--descriptors", type=Path,
        default=Path("inference/ground_truth/descriptors.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inference/posterior"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features, theta, _, _ = load_stage1(args.features, args.theta)
    print(
        f"Stage 3: {args.folds}-fold cross-validated calibration over "
        f"{len(features)} patterns, {args.samples} posterior samples each\n"
    )
    started = time.perf_counter()
    samples, log_probs = cross_validated_posteriors(
        features, theta, args.folds, args.samples, args.seed
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
