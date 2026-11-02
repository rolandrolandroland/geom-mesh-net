"""Stage 5.1: verify the diffusion-field simulator and decide Gate 5.1.

Gate 5.1, as amended on 2026-09-16 (``experiments/reconstruction/ROADMAP.md``, Stage 5
correction), on the 50 development patterns of ``data_diffusion/``:

1. in every pattern, the screened diffusion equation's residual, by automatic
   differentiation in float64 at 2,000 random matrix atoms, is below 1e-8 relative to
   xi^-2 max |c - c_inf|;
2. in every pattern, fewer than 0.1% of matrix atoms need c(x) clipped to [0, 1];
3. in every pattern, each precipitate's surface mean, by direct integration, lies within
   1e-6 of its Gibbs-Thomson value, relative to the depletion amplitude;
4. inside precipitates, pooled over the patterns, the replay oracle passes Gate 0's
   conditions: logistic recalibration slope in [0.99, 1.01], |intercept| <= 0.01, and
   every reliability bin of at least 10,000 atoms within max(0.005, 3 s.e.);
5. in the matrix, in at least 48 of 50 patterns, the realised guest fraction lies within
   three standard errors of the mean of c(x), and the log-likelihood gain of c(x) over
   that constant lies within three standard errors of its expectation.

Condition 5 is what shows the labels follow the field. The gain is
sum_j [y_j log(c_j / c) + (1 - y_j) log((1 - c_j) / (1 - c))] over matrix atoms, for c the
mean of c(x). Under the field its expectation is the summed Kullback-Leibler divergence
and its variance sum_j c_j (1 - c_j) [log(c_j / c) - log((1 - c_j) / (1 - c))]^2. Labels
drawn from a uniform matrix would fall short of the expectation by twice the divergence.

The oracle of every pattern, development and test, is cached for Stage 5.2 to
``results/oracle_diffusion/``: the replay oracle inside precipitates, c(x) in the matrix.
Test patterns are only cached; nothing about their labels enters the gate.

Usage
-----
    python -m experiments.reconstruction.stage5_simulator
"""

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.reconstruction.stage0_oracle import logistic_recalibration, reliability_table
from geom_mesh_net import paths
from geom_mesh_net.fields import oracle as fo
from geom_mesh_net.fields import physics

ORACLE_DIR = paths.RECONSTRUCTION_DIR / "results" / "oracle_diffusion"
SEED_OFFSET = 2_000_000       # replay seeds, distinct from Stage 0's and from every simulation seed
PDE_POINTS = 2000
THRESHOLDS = {"pde_residual": 1e-8, "clipped_fraction": 1e-3, "surface_mean_error": 1e-6,
              "matrix_patterns_within_3se": 48}


def matrix_check(c, y):
    """Do matrix labels ``y`` follow the field values ``c``? Condition 5 of Gate 5.1.

    Two z-scores against their expectations under c: the guest fraction, and the
    log-likelihood gain of c over its own mean. Labels drawn from c give roughly standard
    normal scores; labels drawn from a uniform matrix with the same mean give a gain far
    below its expectation.
    """
    c, y = np.asarray(c, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mean_c, n = float(c.mean()), len(c)
    fraction_z = float((y.mean() - mean_c) / (np.sqrt(np.sum(c * (1 - c))) / n))
    guest_term, host_term = np.log(c / mean_c), np.log((1 - c) / (1 - mean_c))
    gain = float(np.sum(y * guest_term + (1 - y) * host_term))
    expected = float(np.sum(c * guest_term + (1 - c) * host_term))
    sd = float(np.sqrt(np.sum(c * (1 - c) * (guest_term - host_term) ** 2)))
    gain_z = (gain - expected) / sd
    # The score that labels drawn from a uniform matrix at the same mean would get on average:
    # how strongly this check can reject a matrix without the field.
    uniform_expected = float(np.sum(mean_c * guest_term + (1 - mean_c) * host_term))
    return {"matrix_observed": float(y.mean()), "matrix_expected": mean_c, "matrix_fraction_z": fraction_z,
            "log_likelihood_gain": gain, "log_likelihood_gain_expected": expected, "log_likelihood_gain_sd": sd,
            "log_likelihood_gain_z": gain_z, "uniform_labels_expected_gain_z": (uniform_expected - expected) / sd,
            "matrix_within_3se": bool(abs(fraction_z) <= 3 and abs(gain_z) <= 3)}


def field_slice(data_dir, index=0, z=30.0, step=0.5, shell_radii=np.arange(0.0, 15.01, 0.5)):
    """What the field looks like, for the walkthrough: a plane through one development pattern,
    and shell-averaged profiles around its largest and smallest precipitates."""
    with np.load(Path(data_dir) / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        field = physics.DiffusionField.from_dict(d["physics"].item())
    grid = np.arange(0.0, 60.0 + 1e-9, step)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    distance = np.sqrt(((points[:, None, :] - field.centres[None, :, :]) ** 2).sum(axis=-1))
    values = field(points)
    values[(distance <= field.radii[None, :]).any(axis=1)] = np.nan
    cut = np.abs(field.centres[:, 2] - z) < field.radii
    circles = [{"x": float(cx), "y": float(cy), "r": float(np.sqrt(r ** 2 - (cz - z) ** 2)), "surface_value": float(s)}
               for (cx, cy, cz), r, s in zip(field.centres[cut], field.radii[cut], field.surface_values[cut])]
    unit = np.random.default_rng(0).normal(size=(4000, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    profiles = []
    for k in (int(np.argmax(field.radii)), int(np.argmin(field.radii))):
        means = []
        for extra in shell_radii:
            shell = field.centres[k] + (field.radii[k] + extra) * unit
            outside = np.all(np.sqrt(((shell[:, None, :] - field.centres[None, :, :]) ** 2).sum(-1)) > field.radii, axis=1)
            outside[:] = outside | (extra == 0)
            means.append(float(field(shell[outside]).mean()))
        profiles.append({"radius": float(field.radii[k]), "surface_value": float(field.surface_values[k]),
                         "distance_from_surface": shell_radii.tolist(), "shell_mean": means})
    return {"pattern": index, "z": z, "step": step, "grid": grid.tolist(),
            "values": [[None if np.isnan(v) else round(float(v), 5) for v in row] for row in values.reshape(gx.shape)],
            "circles": circles, "profiles": profiles, "c_inf": field.c_inf, "c_eq": field.c_eq, "ell": field.ell,
            "xi": field.xi, "critical_radius": float(field.critical_radius)}


def oracle_path(index):
    return ORACLE_DIR / f"oracle_{index}.npz"


def load_oracle(index):
    """Cached per-atom guest probability and precipitate membership of a Stage 5 pattern."""
    with np.load(oracle_path(index)) as d:
        return d["p"].astype(np.float64), d["inside"]


def build(index, data_dir, replays, gate):
    started = time.perf_counter()
    with np.load(Path(data_dir) / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        coords, labels, centres = d["coords"].item(), d["labels"], d["centers"].item()
        radii = np.asarray(d["radii"], dtype=float)
        field = physics.DiffusionField.from_dict(d["physics"].item())
        parameters = d["parameters"].item()
    x = np.column_stack([coords[a] for a in "xyz"])
    guest = np.isin(labels, (2, 3))

    oracle = fo.replay_oracle(coords, centres, radii, parameters["rho_c"], parameters["matrix_mean"],
                              replays=replays, seed=SEED_OFFSET + index)
    inside = oracle.inside
    c = field(x[~inside])
    p = oracle.p.copy()
    p[~inside] = c
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(oracle_path(index), p=p.astype(np.float32), inside=inside)

    row = {"pattern": index, "precipitates": int(len(radii)),
           # Precipitates are separated, so an atom is inside a sphere exactly when clustersim labelled it 1 or 2.
           "membership_mismatches": int((inside != np.isin(labels, (1, 2))).sum())}
    if gate:
        rng = np.random.default_rng(index)
        matrix_x = x[~inside]
        row["pde_residual"] = field.pde_residual(matrix_x[rng.choice(len(matrix_x), size=PDE_POINTS, replace=False)])
        row["clipped_fraction"] = float(np.mean((c < 0) | (c > 1)))
        amplitude = float(np.median(np.abs(field.surface_values - field.c_inf)))
        row["surface_mean_error"] = float(np.max(np.abs(physics.surface_means(field) - field.surface_values)) / amplitude)

        row.update(matrix_check(c, guest[~inside]))
        row["in_p"] = oracle.p[inside].astype(np.float32)
        row["in_y"] = guest[inside]
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=paths.DIFFUSION_DIR)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "results" / "stage5_simulator.json")
    parser.add_argument("--slice-output", type=Path,
                        default=paths.RECONSTRUCTION_DIR / "results" / "stage5_field_slice.json")
    args = parser.parse_args()
    started = time.perf_counter()

    records = json.loads((args.data_dir / "patterns.json").read_text())
    development = [r["pattern"] for r in records if r["split"] == "development"]
    test = [r["pattern"] for r in records if r["split"] == "test"]
    with Pool(args.workers) as pool:
        rows = pool.map(partial(build, data_dir=args.data_dir, replays=args.replays, gate=True), development)
        cached = pool.map(partial(build, data_dir=args.data_dir, replays=args.replays, gate=False), test)

    p = np.concatenate([r.pop("in_p") for r in rows])
    y = np.concatenate([r.pop("in_y") for r in rows]).astype(np.float64)
    intercept, slope, intercept_se, slope_se = logistic_recalibration(p, y)
    bins = reliability_table(p, y)
    matrix_ok = sum(r["matrix_within_3se"] for r in rows)
    worst = {key: max(r[key] for r in rows) for key in ("pde_residual", "clipped_fraction", "surface_mean_error")}
    checks = {
        "1_pde_residual_below_1e-8_in_every_pattern": worst["pde_residual"] < THRESHOLDS["pde_residual"],
        "2_clipped_below_0.1pct_in_every_pattern": worst["clipped_fraction"] < THRESHOLDS["clipped_fraction"],
        "3_surface_means_within_1e-6_in_every_pattern": worst["surface_mean_error"] < THRESHOLDS["surface_mean_error"],
        "4a_in_sphere_slope_in_[0.99,1.01]": 0.99 <= slope <= 1.01,
        "4b_in_sphere_abs_intercept_le_0.01": abs(intercept) <= 0.01,
        "4c_in_sphere_reliability_bins_within_tolerance": all(b["within"] for b in bins if b["gated"]),
        "5_matrix_within_3se_in_at_least_48": matrix_ok >= THRESHOLDS["matrix_patterns_within_3se"],
    }
    report = {
        "gate": {"checks": checks, "passed": all(checks.values())},
        "development_patterns": len(rows),
        "worst": worst,
        "in_sphere": {"atoms": int(len(p)), "slope": slope, "slope_se": slope_se, "intercept": intercept,
                      "intercept_se": intercept_se, "reliability": bins},
        "matrix": {
            "patterns_within_3se": int(matrix_ok),
            "fraction_z_range": [min(r["matrix_fraction_z"] for r in rows), max(r["matrix_fraction_z"] for r in rows)],
            "log_likelihood_gain_z_range": [min(r["log_likelihood_gain_z"] for r in rows),
                                            max(r["log_likelihood_gain_z"] for r in rows)],
            "log_likelihood_gain_median": float(np.median([r["log_likelihood_gain"] for r in rows])),
            "log_likelihood_gain_expected_median": float(np.median([r["log_likelihood_gain_expected"] for r in rows])),
            "uniform_labels_expected_gain_z_range": [min(r["uniform_labels_expected_gain_z"] for r in rows),
                                                     max(r["uniform_labels_expected_gain_z"] for r in rows)],
        },
        "consistency": {"membership_mismatches": int(sum(r["membership_mismatches"] for r in rows + cached))},
        "replays": args.replays,
        "cached_oracles": {"development": len(rows), "test": len(cached),
                           "directory": paths.repo_relative(ORACLE_DIR)},
        "per_pattern": rows,
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    # The development pattern with the largest share of dissolving precipitates, so the
    # figure shows enrichment beside depletion.
    by_dissolving = min((r for r in records if r["split"] == "development"),
                        key=lambda r: r["depleting_precipitates"] / r["precipitates"])
    args.slice_output.write_text(json.dumps(field_slice(args.data_dir, index=by_dissolving["pattern"])) + "\n")

    print(f"in-sphere: {len(p):,} atoms, slope {slope:.4f} ± {slope_se:.4f}, intercept {intercept:+.4f} ± {intercept_se:.4f}")
    for b in bins:
        flag = "" if b["within"] or not b["gated"] else "  <-- outside tolerance"
        print(f"    p* [{b['lo']:.2f},{b['hi']:.2f})  n={b['atoms']:>8,}  oracle {b['oracle']:.4f}  "
              f"observed {b['observed']:.4f}  gap {b['gap']:+.4f}  tol {b['tolerance']:.4f}{'' if b['gated'] else ' (not gated)'}{flag}")
    print(f"matrix: {matrix_ok}/{len(rows)} within 3 s.e.; fraction z {report['matrix']['fraction_z_range']}, "
          f"gain z {report['matrix']['log_likelihood_gain_z_range']}")
    print(f"worst: {worst}")
    print(f"membership mismatches over {len(rows) + len(cached)} patterns: {report['consistency']['membership_mismatches']}")
    print(json.dumps(report["gate"], indent=2))
    print(f"wrote {args.output} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
