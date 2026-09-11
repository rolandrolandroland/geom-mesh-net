"""Why is rho_b miscalibrated?

Stage 3 found `rho_b` outside the uniform band (deviation 0.0475 against 0.0429)
with 90% coverage at 0.847. Small, but systematic, and not caused by the
zero-cluster patterns. Before spending disk on more simulations, it is worth
knowing *where* the bias comes from, because that determines whether more data
could fix it at all.

Three hypotheses, each with a distinct signature:

  shrinkage      The posterior is pulled toward the prior centre, so it
                 over-predicts small rho_b and under-predicts large rho_b. A
                 regression of posterior mean on truth has slope below one, and
                 the SBC rank trends upward with the true value.

  confounding    The posterior borrows uncertainty from another parameter. The
                 rank then correlates with a parameter it should be independent
                 of. Stage 3's rank array already hints at this: rho_b's rank
                 correlates -0.136 with `rb`.

  saturation     The features that carry rho_b stop responding at one end of its
                 range, so the posterior cannot track it there. The signature is
                 a nonlinear feature-parameter relationship plus locally heavy
                 rank tails.

These are not exclusive, and the point is to find which dominates. Unlike more
simulations, none of these is fixed by sample size if it is structural.

Usage
-----
    PYTHONPATH=. python inference/diagnose_rho_b.py
    PYTHONPATH=. python inference/diagnose_rho_b.py --parameter rho_c
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from inference.fit_posterior import PRIOR_HIGH, PRIOR_LOW, load_stage1
from inference.flow import BoxFlow, fit
from inference.recover_ground_truth import PARAMETER_NAMES
from inference.validate_posterior import sbc_ranks


def out_of_fold_posteriors(features, theta, folds, n_samples, seed):
    """Refit per fold and keep the posterior samples, not only the ranks."""
    n = len(features)
    assignment = np.random.default_rng(seed).permutation(n) % folds
    samples = np.zeros((n, n_samples, theta.shape[1]), dtype=np.float32)

    for fold in range(folds):
        held_out = np.flatnonzero(assignment == fold)
        remaining = np.flatnonzero(assignment != fold)
        stopping, training = remaining[:100], remaining[100:]

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
            samples[held_out] = model.sample(
                context[held_out], n_samples=n_samples,
                generator=torch.Generator().manual_seed(seed + 4242 + fold),
            ).numpy()
        print(f"  fold {fold + 1}/{folds}", flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", default="rho_b")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--features", type=Path,
        default=Path("inference/features/global_features.npz"),
    )
    args = parser.parse_args()

    position = PARAMETER_NAMES.index(args.parameter)
    features, theta, names, _ = load_stage1(
        args.features, Path("inference/ground_truth/theta.npy")
    )
    print(f"Refitting {args.folds} folds to recover posterior samples\n")
    samples = out_of_fold_posteriors(
        features, theta, args.folds, args.samples, args.seed
    )

    truth = theta[:, position]
    posterior_mean = samples[:, :, position].mean(axis=1)
    posterior_sd = samples[:, :, position].std(axis=1)
    residual = posterior_mean - truth
    ranks = sbc_ranks(samples, theta)
    normalized_rank = (ranks[:, position] + 0.5) / (args.samples + 1)

    print(f"\n{'=' * 72}\nDiagnosis for {args.parameter}\n{'=' * 72}")
    print(f"  mean normalized rank {normalized_rank.mean():.4f} (uniform = 0.5)")
    print(f"  mean residual        {residual.mean():+.5f}")
    print(f"  residual sd          {residual.std():.5f}")
    print(f"  mean posterior sd    {posterior_sd.mean():.5f}")
    ratio = residual.std() / posterior_sd.mean()
    print(f"  residual sd / posterior sd = {ratio:.3f}")
    print(
        "    Above 1 means the posterior is narrower than its own errors, which\n"
        "    is overconfidence. Near 1 is honest."
    )

    # Hypothesis 1: shrinkage toward the prior centre.
    slope, intercept = np.polyfit(truth, posterior_mean, 1)
    print(f"\n  HYPOTHESIS 1, shrinkage")
    print(f"    regression of posterior mean on truth: slope {slope:.3f}")
    print(
        "    A slope below 1 means the posterior systematically under-reacts to\n"
        "    the truth, pulling estimates toward the middle of the prior."
    )
    corr_rank_truth = np.corrcoef(normalized_rank, truth)[0, 1]
    print(f"    corr(rank, true {args.parameter}) = {corr_rank_truth:+.3f}")
    print(
        "    Shrinkage also makes the rank rise with the truth, since large\n"
        "    values end up above their posterior and small ones below."
    )
    verdict_1 = slope < 0.9 and corr_rank_truth > 0.1
    print(f"    -> {'SUPPORTED' if verdict_1 else 'not supported'}")

    # Hypothesis 2: another parameter contaminating this one.
    print(f"\n  HYPOTHESIS 2, confounding with another parameter")
    print("    The rank must be independent of every parameter, including its own.")
    worst_name, worst_value = None, 0.0
    for other, name in enumerate(PARAMETER_NAMES):
        correlation = np.corrcoef(normalized_rank, theta[:, other])[0, 1]
        residual_corr = np.corrcoef(residual, theta[:, other])[0, 1]
        print(
            f"    {name:>7}: corr(rank) {correlation:+.3f}   "
            f"corr(residual) {residual_corr:+.3f}"
        )
        if abs(correlation) > abs(worst_value):
            worst_name, worst_value = name, correlation
    verdict_2 = abs(worst_value) > 0.1
    print(
        f"    -> {'SUPPORTED' if verdict_2 else 'not supported'}"
        f" (largest: {worst_name} at {worst_value:+.3f})"
    )

    # Hypothesis 3: the carrying features stop responding at one end.
    print(f"\n  HYPOTHESIS 3, feature saturation")
    correlations = np.array(
        [abs(np.corrcoef(features[:, j], truth)[0, 1]) for j in range(features.shape[1])]
    )
    carriers = np.argsort(-correlations)[:3]
    print(f"    carried by: " + ", ".join(
        f"{names[j]} ({correlations[j]:.2f})" for j in carriers
    ))
    print(f"\n    {'quintile of ' + args.parameter:>26}{'n':>5}"
          f"{'mean rank':>11}{'resid sd':>10}{'post sd':>10}{'ratio':>8}")
    edges = np.quantile(truth, np.linspace(0, 1, 6))
    ratios = []
    for i in range(5):
        mask = (truth >= edges[i]) & (truth <= edges[i + 1])
        local_ratio = residual[mask].std() / posterior_sd[mask].mean()
        ratios.append(local_ratio)
        print(
            f"    {edges[i]:>11.4f}-{edges[i+1]:<10.4f}{mask.sum():>5}"
            f"{normalized_rank[mask].mean():>11.4f}{residual[mask].std():>10.5f}"
            f"{posterior_sd[mask].mean():>10.5f}{local_ratio:>8.2f}"
        )
    print(
        "\n    A ratio far above 1 in some quintiles and near 1 in others means the\n"
        "    posterior width does not adapt to where the features are informative."
    )
    verdict_3 = max(ratios) / min(ratios) > 1.5
    print(
        f"    -> {'SUPPORTED' if verdict_3 else 'not supported'} "
        f"(ratio spread {min(ratios):.2f} to {max(ratios):.2f})"
    )

    print(f"\n{'=' * 72}")
    supported = [
        name for name, ok in (
            ("shrinkage", verdict_1),
            ("confounding", verdict_2),
            ("feature saturation", verdict_3),
        ) if ok
    ]
    if supported:
        print("  Supported: " + ", ".join(supported))
    else:
        print("  No single hypothesis dominates; the departure is diffuse.")
    print(
        "\n  More simulations help a diffuse or small-sample departure. They do not\n"
        "  help a structural one -- shrinkage from an over-regularized fit,\n"
        "  confounding the flow cannot resolve, or features that saturate."
    )


if __name__ == "__main__":
    main()
