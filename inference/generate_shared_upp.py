"""Simulate patterns that all share one underlying point pattern.

rapt's walkthrough, which produced Bennett, Proudian and Zimmerman (2023), calls
``clustersim(pattern, pattern, ...)`` with the same Poisson pattern for every
training and test pattern, and computes the relabeling expectation on that same
pattern. Only the labels vary between patterns. That removes point-position
noise from the features entirely.

``data/`` was generated differently: every pattern has its own independently
drawn points. The paper reports R^2 of roughly 0.71 for the radius dispersity
where this dataset reaches about 0.1, and the shared pattern is a candidate
explanation that needs no appeal to the model or the features.

This script tests it with a paired design. It reuses the exact parameter vector
of every pattern in ``data/`` -- read from ``inference/ground_truth/theta.npy`` --
so that the only difference between the two datasets is whether the underlying
points are shared. Any change in recoverability is attributable to that.

Generation is serial on purpose. ``clustersim`` draws from a module-level
generator seeded at import, so parallel workers would each start from the same
state and produce correlated clusterings.

Usage
-----
    PYTHONPATH=. python inference/generate_shared_upp.py
    PYTHONPATH=. python inference/generate_shared_upp.py --limit 50 --output-dir /tmp/shared
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geom_mesh_net.core_functions import clustersim as csim


# Mirrors data_factory.py so that point sharing is the only change.
SIDE = 60
OPP_SIDE = 30
INTENSITY = 1
PCP = 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_shared_upp"))
    parser.add_argument("--theta", type=Path, default=Path("inference/ground_truth/theta.npy"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir.resolve() == Path("data").resolve():
        raise SystemExit("refusing to write into data/, which holds the original dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    theta = np.load(args.theta)
    count = len(theta) if args.limit is None else min(args.limit, len(theta))

    domain = {axis: np.array([0.0, float(SIDE)]) for axis in "xyz"}
    domain_opp = {axis: np.array([0.0, float(OPP_SIDE)]) for axis in "xyz"}

    # Drawn once. PointPattern3 copies its arrays, so each pattern below starts
    # from an untouched copy of the same points.
    shared_points, shared_labels = csim.gen_rand_points(
        intensity=INTENSITY, dim1=SIDE, dim2=SIDE, dim3=SIDE
    )
    opp_points, opp_labels = csim.gen_uniform_points(
        intensity=INTENSITY, dim1=OPP_SIDE, dim2=OPP_SIDE, dim3=OPP_SIDE
    )

    stats = np.full((count, 11), np.nan)
    started = time.perf_counter()
    failures = []
    for index in range(count):
        rho_c, rho_b, cr, rb = theta[index]
        upp = csim.PointPattern3(shared_points, domain=domain, labels=shared_labels)
        opp = csim.PointPattern3(opp_points, domain=domain_opp, labels=opp_labels)
        try:
            pattern, radii, centers = csim.clustersim(
                opp=opp, upp=upp, pcp=PCP, rho_c=rho_c, rho_b=rho_b, cr=cr, rb=rb,
                cut="buffered", buffer_factor=1, selection="sampled",
                prob_function="Gaussian_decay",
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            failures.append((index, f"{type(exc).__name__}: {exc}"))
            continue

        np.savez(
            args.output_dir / f"clust_pattern_{index}",
            coords=pattern.coords, domain=pattern.domain, labels=pattern.labels,
            radii=radii, centers=centers,
        )
        labels = pattern.labels
        guests = np.isin(labels, (2, 3)).sum()
        in_cluster_guest, in_cluster_host = (labels == 2).sum(), (labels == 1).sum()
        background_guest, background_host = (labels == 3).sum(), (labels == 0).sum()
        stats[index] = [
            guests / pattern.n_points, PCP, np.nan,
            in_cluster_guest / max(in_cluster_guest + in_cluster_host, 1), rho_c, np.nan,
            background_guest / max(background_guest + background_host, 1), rho_b, np.nan,
            cr, rb,
        ]
        if (index + 1) % 100 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"  {index + 1}/{count}  {rate:.2f} patterns/s", flush=True)

    np.save(args.output_dir / "pattern_stats", stats)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "generated_by": "inference/generate_shared_upp.py",
        "design": "same parameter vectors as data/, one shared underlying point pattern",
        "theta_source": str(args.theta),
        "n_patterns": count,
        "n_failed": len(failures),
        "failures": failures[:20],
        "shared_point_count": int(len(shared_labels)),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2) + "\n")
    print(f"wrote {count - len(failures)} patterns to {args.output_dir} "
          f"({len(failures)} failed) in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
