"""Stage 0: recover and persist the simulator's ground-truth parameters.

``scripts/generate_data.py`` varies four parameters but writes only two of them
to ``data/pattern_stats.npy``. The mean cluster radius ``cr`` and the radius
spread ``rb`` were never saved.

They are recoverable because the factory seeds ``np.random.default_rng(42)``
and draws in a fixed order:

    rho_c_vec = rng.uniform(low=pcp*2, high=1,       size=n_sims)
    rho_b_vec = rng.uniform(low=0,     high=pcp*0.5, size=n_sims)
    cr_vec    = rng.uniform(low=3,     high=15,      size=n_sims)
    rb_vec    = rng.uniform(low=0,     high=0.5,     size=n_sims)

Replaying that sequence reproduces ``rho_c`` and ``rho_b`` exactly, which
certifies the replay and therefore certifies ``cr`` and ``rb`` too. This script
performs that check and then writes the parameters to disk so that nothing
downstream ever depends on the replay again.

That matters because the guarantee is fragile: any edit to the order, count or
distribution of those four draws silently invalidates it, with no error.

Usage
-----
    python -m experiments.inference.recover_ground_truth
    python -m experiments.inference.recover_ground_truth --limit 50 --no-descriptors

See ``experiments/inference/ROADMAP.md`` sections 2.2 and 6 (Stage 0).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geom_mesh_net import paths
from geom_mesh_net.simulation.parameters import (
    FACTORY_N_SIMS, FACTORY_PCP, FACTORY_SEED, PARAMETER_NAMES, replay_factory_draws,
)

GUEST_MARKS = (2, 3)

# Column layout of data/pattern_stats.npy, from scripts/generate_data.py.
STATS_COLUMNS = {
    "pcp_measured": 0,
    "pcp_true": 1,
    "pcp_percent_error": 2,
    "rho_c_measured": 3,
    "rho_c_true": 4,
    "rho_c_percent_error": 5,
    "rho_b_measured": 6,
    "rho_b_true": 7,
    "rho_b_percent_error": 8,
}


def verify_replay(theta, stats):
    """Check replayed values against the two parameters the factory did save.

    Returns a dict of diagnostics. The gate is exact equality: these are the
    same float64 draws from the same seeded generator, so any nonzero
    difference means the draw sequence has changed and the replay is void.
    """
    n = len(stats)
    checks = {}
    for name in ("rho_c", "rho_b"):
        saved = stats[:, STATS_COLUMNS[f"{name}_true"]]
        replayed = theta[name][:n]
        max_abs_diff = float(np.abs(saved - replayed).max())
        checks[name] = {
            "max_abs_diff": max_abs_diff,
            "exact": max_abs_diff == 0.0,
        }
    saved_pcp = stats[:, STATS_COLUMNS["pcp_true"]]
    checks["pcp"] = {
        "constant": bool(np.all(saved_pcp == saved_pcp[0])),
        "value": float(saved_pcp[0]),
    }
    return checks


def pattern_descriptors(data_dir, index):
    """Realized quantities for one pattern, read without loading coordinates.

    ``.npz`` members are decompressed individually, so reading `radii` and
    `labels` does not pay for the 216,000 x 3 coordinate array.
    """
    with np.load(data_dir / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        radii = np.asarray(d["radii"], dtype=float)
        labels = d["labels"]
    n_points = int(len(labels))
    guest_count = int(np.isin(labels, GUEST_MARKS).sum())
    return {
        "n_clusters": int(len(radii)),
        "radius_mean": float(radii.mean()) if len(radii) else np.nan,
        "radius_sd": float(radii.std()) if len(radii) else np.nan,
        "radius_min": float(radii.min()) if len(radii) else np.nan,
        "radius_max": float(radii.max()) if len(radii) else np.nan,
        "n_points": n_points,
        "guest_count": guest_count,
        "guest_fraction": guest_count / n_points if n_points else np.nan,
        # A pattern with no clusters is a legitimate draw from the prior
        # predictive, not a bug, but it must be handled explicitly downstream
        # rather than dropped silently -- dropping biases the training set.
        "degenerate": bool(len(radii) == 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=paths.GROUND_TRUTH_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only describe the first N patterns (theta is always full length)",
    )
    parser.add_argument(
        "--no-descriptors",
        action="store_true",
        help="skip reading .npz files; write theta and the verification only",
    )
    args = parser.parse_args()

    stats_path = args.data_dir / "pattern_stats.npy"
    if not stats_path.exists():
        raise SystemExit(f"missing {stats_path}")
    stats = np.load(stats_path)

    theta = replay_factory_draws()
    checks = verify_replay(theta, stats)

    print("Stage 0 gate: does the RNG replay reproduce the saved parameters?")
    gate_passed = True
    for name in ("rho_c", "rho_b"):
        c = checks[name]
        status = "PASS" if c["exact"] else "FAIL"
        gate_passed &= c["exact"]
        print(f"  {name:6s} max abs diff = {c['max_abs_diff']:.3e}   {status}")
    print(
        f"  pcp    constant = {checks['pcp']['constant']} "
        f"at {checks['pcp']['value']}  (not a parameter; excluded from theta)"
    )

    if not gate_passed:
        raise SystemExit(
            "\nGATE FAILED. The replayed draws do not match pattern_stats.npy, so "
            "scripts/generate_data.py's RNG usage has changed and cr/rb cannot be "
            "trusted. "
            "Regenerate the dataset, writing all four parameters this time."
        )
    print("\nGate passed: cr and rb from the same replay are exact.\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta_matrix = np.column_stack([theta[n] for n in PARAMETER_NAMES])
    np.save(args.output_dir / "theta.npy", theta_matrix)
    print(f"wrote theta.npy  shape {theta_matrix.shape}  columns {PARAMETER_NAMES}")

    descriptors = None
    describe_seconds = 0.0
    if not args.no_descriptors:
        count = args.limit or FACTORY_N_SIMS
        start = time.perf_counter()
        rows = []
        for i in range(count):
            path = args.data_dir / f"clust_pattern_{i}.npz"
            if not path.exists():
                print(f"  stopping at index {i}: {path} not found")
                break
            rows.append(pattern_descriptors(args.data_dir, i))
            if (i + 1) % 100 == 0:
                print(f"  described {i + 1}/{count}")
        describe_seconds = time.perf_counter() - start
        keys = list(rows[0])
        descriptors = {k: np.array([r[k] for r in rows]) for k in keys}
        np.savez(args.output_dir / "descriptors.npz", **descriptors)
        print(
            f"wrote descriptors.npz  {len(rows)} patterns  "
            f"({describe_seconds:.1f} s)"
        )

        deg = int(descriptors["degenerate"].sum())
        ncl = descriptors["n_clusters"]
        gf = descriptors["guest_fraction"]
        print(f"\n  degenerate (zero-cluster) patterns: {deg}/{len(rows)}")
        print(
            f"  n_clusters: min {ncl.min()} median {int(np.median(ncl))} "
            f"max {ncl.max()}"
        )
        print(f"  guest fraction: {gf.mean():.4f} +/- {gf.std():.4f} "
              f"(target pcp = {FACTORY_PCP})")

        ok = ~descriptors["degenerate"]
        cr = theta["cr"][: len(rows)][ok]
        print("\n  Identifiability expectations (Stage 3 predictions):")
        print(
            f"    corr(cr, log n_clusters)  = "
            f"{np.corrcoef(cr, np.log(ncl[ok]))[0, 1]:+.3f}  -> cr should be sharp"
        )
        print(
            f"    corr(rb, realized r mean) = "
            f"{np.corrcoef(theta['rb'][:len(rows)][ok], descriptors['radius_mean'][ok])[0, 1]:+.3f}"
            f"  -> rb should stay near its prior"
        )

    provenance = {
        "generated_by": "experiments/inference/recover_ground_truth.py",
        "source": "scripts/generate_data.py",
        "method": "replay of np.random.default_rng(42) in the factory's draw order",
        "factory_seed": FACTORY_SEED,
        "factory_n_sims": FACTORY_N_SIMS,
        "factory_pcp_constant": FACTORY_PCP,
        "parameter_names": list(PARAMETER_NAMES),
        "priors": {
            "rho_c": [FACTORY_PCP * 2, 1.0],
            "rho_b": [0.0, FACTORY_PCP * 0.5],
            "cr": [3.0, 15.0],
            "rb": [0.0, 0.5],
        },
        "verification": checks,
        "patterns_described": 0 if descriptors is None else int(
            len(descriptors["n_clusters"])
        ),
        "describe_seconds": round(describe_seconds, 2),
        "warning": (
            "cr and rb are not stored in data/pattern_stats.npy. They exist only "
            "here. Any change to the RNG draws in scripts/generate_data.py "
            "invalidates the "
            "replay silently, so treat these files as the authoritative record."
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"\nwrote provenance.json -> {args.output_dir}")


if __name__ == "__main__":
    main()
