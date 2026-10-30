"""Stage 2: fit q(theta | features), the amortized posterior.

Trains the conditional flow in ``geom_mesh_net/inference/flow.py`` by maximum
likelihood on the simulated (theta, features) pairs. Because those pairs are
drawn from the joint p(theta) p(x | theta), the minimizer of the training loss is
the true posterior p(theta | s) -- Bayesian inference recast as conditional
density estimation. See ``experiments/inference/ROADMAP.md`` section 3.

The gate is that held-out log-likelihood beats the uniform prior. A flow that
cannot beat the prior has learned nothing, which would be a real and reportable
negative result about the 14 features rather than a bug.

What this stage does *not* establish is whether the posterior is calibrated. A
model can beat the prior handsomely and still be systematically overconfident.
That is Stage 3, and it is the gate that matters.

Stage 3 found that an ensemble of five independently seeded flows is what makes
the coverage claim correct (ROADMAP section 8.8), so that is what this stage now
fits and saves: ``flow.pt`` holds every member, and the posterior is their equal
mixture. Sampling splits the draws evenly between members, because rank
granularity depends on the number of draws and an ensemble drawing more than the
model it is compared against is not comparable. ``--ensemble 1`` reproduces the
single flow the first Stage 2 run reported.

Splitting is by pattern index. If observation augmentation is added later
(ROADMAP section 7), replicates of one pattern must stay on the same side of the
split or the held-out set leaks and every calibration number becomes invalid.

Usage
-----
    python -m experiments.inference.fit_posterior
    python -m experiments.inference.fit_posterior --seed 1 --n-layers 6
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geom_mesh_net import paths
from geom_mesh_net.inference.flow import BoxFlow, fit
from geom_mesh_net.simulation.parameters import PARAMETER_NAMES, PRIOR_HIGH, PRIOR_LOW


N_VALIDATION = 100
N_TEST = 100
SPLIT_SEED = 42


ENSEMBLE = 5          # the configuration Stage 3 validated; see ROADMAP section 8.8


def mixture_log_prob(models, targets, context):
    """Log density of the equal mixture of the members, which is the ensemble's posterior."""
    with torch.no_grad():
        stacked = torch.stack([model.log_prob(targets, context) for model in models])
    return (torch.logsumexp(stacked, dim=0) - np.log(len(models))).numpy()


def mixture_sample(models, context, n_samples, seed):
    """Equal draws from each member, so the total does not depend on the ensemble size."""
    per_member = max(1, n_samples // len(models))
    with torch.no_grad():
        draws = [model.sample(context, n_samples=per_member,
                              generator=torch.Generator().manual_seed(seed + position))
                 for position, model in enumerate(models)]
    return torch.cat(draws, dim=1).numpy()


def split_indices(n, n_val=N_VALIDATION, n_test=N_TEST, seed=SPLIT_SEED):
    """Disjoint train/validation/test partition over pattern indices."""
    order = np.random.default_rng(seed).permutation(n)
    test = np.sort(order[:n_test])
    validation = np.sort(order[n_test : n_test + n_val])
    train = np.sort(order[n_test + n_val :])
    assert len(np.intersect1d(train, validation)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(validation, test)) == 0
    return train, validation, test


def load_stage1(features_path, theta_path):
    with np.load(features_path, allow_pickle=False) as cached:
        features = cached["values"].astype(np.float64)
        feature_names = [str(name) for name in cached["feature_names"]]
        interior = cached["k_extrema_interior"]
    theta = np.load(theta_path)[: len(features)]

    finite = np.all(np.isfinite(features), axis=1)
    if not finite.all():
        print(f"dropping {(~finite).sum()} patterns with non-finite features")
    return features[finite], theta[finite], feature_names, interior[finite]


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=ENSEMBLE,
                        help="independently seeded flows to fit and save; 1 is the original single flow")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--posterior-samples", type=int, default=2000)
    args = parser.parse_args()

    features, theta, feature_names, interior = load_stage1(args.features, args.theta)
    train, validation, test = split_indices(len(features))

    # Standardize the context using training statistics only. Using all of it
    # would leak held-out information into the input scaling.
    mean = features[train].mean(axis=0)
    sd = features[train].std(axis=0)
    sd[sd == 0.0] = 1.0
    context = torch.tensor((features - mean) / sd, dtype=torch.float32)
    targets = torch.tensor(theta, dtype=torch.float32)

    print(f"Stage 2: fitting q(theta | features) on {len(features)} patterns")
    print(f"  train {len(train)}  validation {len(validation)}  test {len(test)}")
    print(f"  flow: {args.n_layers} layers, {args.hidden} hidden, seed {args.seed}, "
          f"{args.ensemble} member{'s' if args.ensemble > 1 else ''}\n")

    models, histories, member_validation = [], [], []
    started = time.perf_counter()
    for member in range(args.ensemble):
        seed = args.seed + member
        torch.manual_seed(seed)
        model = BoxFlow(
            PRIOR_LOW, PRIOR_HIGH, context_dim=features.shape[1],
            n_layers=args.n_layers, hidden=args.hidden,
        )
        history, member_best = fit(
            model,
            targets[train], context[train],
            targets[validation], context[validation],
            max_epochs=args.max_epochs, seed=seed,
        )
        model.eval()
        models.append(model)
        histories.append(history)
        member_validation.append(member_best)
        if args.ensemble > 1:
            print(f"  member {member + 1}/{args.ensemble} (seed {seed}): "
                  f"validation log-likelihood {member_best:8.3f} nats over {len(history)} epochs")
    elapsed = time.perf_counter() - started

    prior_log_prob = models[0].log_prior().item()
    # The gate is on the mixture, because the mixture is what is saved and reported.
    best_val = float(mixture_log_prob(models, targets[validation], context[validation]).mean())
    test_log_probs = mixture_log_prob(models, targets[test], context[test])
    test_mean = float(test_log_probs.mean())
    test_median = float(np.median(test_log_probs))

    print(f"\n  trained in {elapsed:.1f} s over {sum(len(h) for h in histories)} epochs")
    print("\nStage 2 gate (pre-registered, ROADMAP section 6):")
    print("  validation log-likelihood must beat the uniform prior.")
    print(f"    uniform prior log density : {prior_log_prob:8.3f} nats")
    print(f"    validation log-likelihood : {best_val:8.3f} nats")
    passed = best_val > prior_log_prob
    print(f"    gain                      : {best_val - prior_log_prob:+8.3f} nats")
    print(f"\n  GATE {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print(
            "    The features carry no usable information about theta. That is a\n"
            "    real negative result about the 14 features, not necessarily a bug."
        )

    # Validation was used for early stopping, so it is optimistic. The test set
    # is the honest number, and it is reported whether or not it agrees.
    print("\nHeld-out test set (never used for stopping or selection):")
    print(f"    mean log-likelihood       : {test_mean:8.3f} nats")
    print(f"    median log-likelihood     : {test_median:8.3f} nats")
    below = int((test_log_probs < prior_log_prob).sum())
    print(f"    patterns below the prior  : {below}/{len(test)}")

    # The mean of a log-density is not robust: one pattern assigned a very small
    # density dominates it. Report the gap explicitly rather than quietly
    # preferring whichever statistic looks better.
    if test_mean < prior_log_prob <= test_median:
        worst_position = int(np.argmin(test_log_probs))
        worst_pattern = int(test[worst_position])
        print(
            f"\n    NOTE: mean and median disagree. The mean is dominated by "
            f"pattern\n    {worst_pattern} at {test_log_probs[worst_position]:.1f} "
            f"nats. Excluding it, the mean is "
            f"{float(np.delete(test_log_probs, worst_position).mean()):.3f}."
        )
        print(
            "    A log-density that small means the posterior was confidently wrong\n"
            "    there, which is real and must not be averaged away. Stage 3 exists\n"
            "    to quantify exactly this."
        )

    degenerate_path = paths.GROUND_TRUTH_DIR / "descriptors.npz"
    if degenerate_path.exists():
        with np.load(degenerate_path) as descriptors:
            degenerate = descriptors["degenerate"][: len(features)]
        in_test = np.intersect1d(test, np.flatnonzero(degenerate))
        if len(in_test):
            positions = [int(np.where(test == index)[0][0]) for index in in_test]
            print(
                f"\n    Zero-cluster patterns in the test set: {in_test.tolist()}, "
                f"log-density {[round(float(test_log_probs[p]), 1) for p in positions]}"
            )
            print(
                "    A pattern with no clusters carries no information about the\n"
                "    cluster radius, so its posterior should widen toward the prior.\n"
                "    With only 2 such patterns in 1000 the flow has no way to learn\n"
                "    that, and is confidently wrong instead. See ROADMAP section 7."
            )
    test_log_prob = test_mean

    # Posterior samples on the test set, for Stage 3 to calibrate.
    samples = mixture_sample(models, context[test], args.posterior_samples, args.seed + 1000)

    print("\nPer-parameter marginals on the test set:")
    print(
        f"{'param':>7} {'prior sd':>9} {'post sd':>9} {'contraction':>12} "
        f"{'bias':>9} {'|z| mean':>9}"
    )
    print("-" * 60)
    marginals = {}
    for position, name in enumerate(PARAMETER_NAMES):
        prior_sd = (PRIOR_HIGH[position] - PRIOR_LOW[position]) / np.sqrt(12.0)
        posterior_sd = samples[:, :, position].std(axis=1)
        posterior_mean = samples[:, :, position].mean(axis=1)
        truth = theta[test, position]
        contraction = 1.0 - posterior_sd.mean() / prior_sd
        bias = (posterior_mean - truth).mean()
        # Standardized error: near 1 if the spread honestly reflects the error.
        z = np.abs(posterior_mean - truth) / np.maximum(posterior_sd, 1e-12)
        marginals[name] = {
            "prior_sd": float(prior_sd),
            "posterior_sd_mean": float(posterior_sd.mean()),
            "contraction": float(contraction),
            "bias": float(bias),
            "abs_z_mean": float(z.mean()),
        }
        print(
            f"{name:>7} {prior_sd:>9.4f} {posterior_sd.mean():>9.4f} "
            f"{contraction:>12.3f} {bias:>+9.4f} {z.mean():>9.2f}"
        )
    print(
        "\n  |z| mean near 0.8 is what a calibrated Gaussian-ish posterior gives.\n"
        "  Much above that means overconfident, much below means too wide.\n"
        "  Stage 3 replaces this rough check with SBC and coverage."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "members": [member.state_dict() for member in models],
            "seeds": [args.seed + position for position in range(len(models))],
            "state_dict": models[0].state_dict(),   # the first member, for readers that expect one flow
            "prior_low": PRIOR_LOW, "prior_high": PRIOR_HIGH,
            "context_mean": mean, "context_sd": sd,
            "n_layers": args.n_layers, "hidden": args.hidden,
            "context_dim": features.shape[1],
        },
        args.output_dir / "flow.pt",
    )
    np.savez_compressed(
        args.output_dir / "test_posterior.npz",
        samples=samples.astype(np.float32),
        theta_true=theta[test].astype(np.float32),
        test_index=test,
        train_index=train,
        validation_index=validation,
        parameter_names=np.array(PARAMETER_NAMES),
    )
    metadata = {
        "stage": 2,
        "seed": args.seed,
        "ensemble": args.ensemble,
        "member_validation_log_prob": [float(v) for v in member_validation],
        "n_patterns": int(len(features)),
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "n_layers": args.n_layers,
        "hidden": args.hidden,
        "epochs_run": [len(h) for h in histories],
        "train_seconds": round(elapsed, 2),
        "prior_log_prob": prior_log_prob,
        "validation_log_prob": best_val,
        "test_log_prob_mean": test_mean,
        "test_log_prob_median": test_median,
        "test_below_prior": below,
        "validation_gain_over_prior_nats": best_val - prior_log_prob,
        "gate_passed": bool(passed),
        "gate_criterion": "validation log-likelihood > uniform prior",
        "marginals": marginals,
        "feature_names": feature_names,
    }
    (args.output_dir / "fit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    with open(args.output_dir / "training_history.csv", "w") as handle:
        handle.write("member,epoch,train_log_prob,val_log_prob\n")
        for member, rows in enumerate(histories):
            for row in rows:
                handle.write(
                    f"{member},{row['epoch']},{row['train_log_prob']:.6f},"
                    f"{row['val_log_prob']:.6f}\n"
                )
    print(f"\nwrote {args.output_dir}/")


if __name__ == "__main__":
    main()
