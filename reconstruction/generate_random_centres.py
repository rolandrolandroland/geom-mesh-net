"""Regenerate the patterns of data/ with randomly placed cluster centres.

In ``data/`` the cluster centres lie exactly on a cubic lattice centred on the
box, its spacing fixed by the parameters (``reconstruction/ROADMAP.md`` section
3.2). A model trained across patterns can learn where clusters are allowed to
be, which no real material would teach it. rapt's walkthrough, which produced
the published results, drew centres from a Poisson pattern.

This script reuses every parameter vector of ``data/``, read from
``inference/ground_truth/theta.npy``, and every other setting of
``data_factory.py``. It changes one thing: the overlying pattern is uniform
random points instead of a lattice. clustersim shrinks that pattern to a target
density of centres and keeps a random subset of the size it needs.
``OPP_OVERSAMPLE`` makes the pattern eight times denser than the target, so the
window is almost never short of centres. The cluster count then follows theta
exactly as it does for ``data/``.

``estimate_cluster_volume`` draws from NumPy's global generator, so it is seeded
per pattern here. ``data/`` was not seeded that way, so a cluster count can still
differ from ``data/`` by one where the volume estimate sits on a rounding
boundary. The provenance records how often that happens.

Generation is serial: clustersim draws from a module-level generator seeded at
import, so parallel workers would all start from the same state.

Usage
-----
    PYTHONPATH=. python reconstruction/generate_random_centres.py
    PYTHONPATH=. python reconstruction/generate_random_centres.py --limit 20 --output-dir /tmp/rc
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from geom_mesh_net.core_functions import clustersim as csim

# Mirrors data_factory.py so that centre placement is the only change.
SIDE = 60
OPP_SIDE = 30
INTENSITY = 1
PCP = 0.1
OPP_OVERSAMPLE = 8


def on_lattice(centres):
    """True when every axis's distinct centre coordinates are evenly spaced."""
    for axis in "xyz":
        values = np.unique(np.round(centres[axis], 6))
        if len(values) < 3:
            return None  # too few distinct coordinates to tell
        steps = np.diff(values)
        if not np.allclose(steps, steps[0], atol=1e-6):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data_random_centres"))
    parser.add_argument("--theta", type=Path, default=Path("inference/ground_truth/theta.npy"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data"),
                        help="lattice dataset whose cluster counts are compared")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir.resolve() in {Path("data").resolve(), Path("data_shared_upp").resolve()}:
        raise SystemExit(f"refusing to write into {args.output_dir}, which holds another dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    theta = np.load(args.theta)
    count = len(theta) if args.limit is None else min(args.limit, len(theta))
    domain = {axis: np.array([0.0, float(SIDE)]) for axis in "xyz"}
    domain_opp = {axis: np.array([0.0, float(OPP_SIDE)]) for axis in "xyz"}

    stats = np.full((count, 11), np.nan)
    checksums, count_differences, lattice_like, failures = {}, [], 0, []
    started = time.perf_counter()
    for index in range(count):
        rho_c, rho_b, cr, rb = theta[index]
        np.random.seed(index)  # estimate_cluster_volume's Monte Carlo draws
        points, labels = csim.gen_rand_points(intensity=INTENSITY, dim1=SIDE, dim2=SIDE, dim3=SIDE)
        upp = csim.PointPattern3(points, domain=domain, labels=labels)
        opp_points, opp_labels = csim.gen_rand_points(intensity=INTENSITY, dim1=OPP_SIDE, dim2=OPP_SIDE, dim3=OPP_SIDE)
        opp = csim.PointPattern3(opp_points, domain=domain_opp, labels=opp_labels)
        try:
            pattern, radii, centers = csim.clustersim(
                opp=opp, upp=upp, pcp=PCP, rho_c=rho_c, rho_b=rho_b, cr=cr, rb=rb,
                cut="buffered", buffer_factor=1, selection="sampled",
                prob_function="Gaussian_decay", opp_oversample=OPP_OVERSAMPLE,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            failures.append((index, f"{type(exc).__name__}: {exc}"))
            continue

        path = args.output_dir / f"clust_pattern_{index}.npz"
        np.savez(path, coords=pattern.coords, domain=pattern.domain, labels=pattern.labels,
                 radii=radii, centers=centers)
        checksums[index] = hashlib.sha256(path.read_bytes()).hexdigest()
        lattice_like += int(on_lattice(centers) is True)

        reference = args.reference_dir / f"clust_pattern_{index}.npz"
        if reference.exists():
            n_reference = len(np.load(reference, allow_pickle=True)["radii"])
            if n_reference != len(radii):
                count_differences.append((index, n_reference, int(len(radii))))

        labels = pattern.labels
        in_guest, in_host = (labels == 2).sum(), (labels == 1).sum()
        bg_guest, bg_host = (labels == 3).sum(), (labels == 0).sum()
        stats[index] = [
            np.isin(labels, (2, 3)).sum() / pattern.n_points, PCP, np.nan,
            in_guest / max(in_guest + in_host, 1), rho_c, np.nan,
            bg_guest / max(bg_guest + bg_host, 1), rho_b, np.nan,
            cr, rb,
        ]
        if (index + 1) % 50 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"  {index + 1}/{count}  {rate:.2f} patterns/s", flush=True)

    np.save(args.output_dir / "pattern_stats", stats)
    provenance = {
        "generated_by": "reconstruction/generate_random_centres.py",
        "design": "same parameter vectors and settings as data/; overlying pattern uniform random, not a lattice",
        "theta_source": str(args.theta),
        "opp_oversample": OPP_OVERSAMPLE,
        "global_numpy_seed": "np.random.seed(index) before each pattern",
        "n_patterns": count,
        "n_failed": len(failures),
        "failures": failures[:20],
        "patterns_with_lattice_centres": lattice_like,
        "cluster_count_differences_from_reference": {
            "reference_dir": str(args.reference_dir),
            "n_different": len(count_differences),
            "examples_index_reference_new": count_differences[:20],
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output_dir / "checksums.json").write_text(json.dumps(checksums, indent=1) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
