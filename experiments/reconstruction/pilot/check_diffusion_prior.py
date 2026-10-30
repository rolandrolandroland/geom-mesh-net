"""Can the Stage 5 physics be seen, and its constants recovered, in the development patterns?

For each development pattern of ``data_diffusion/``, with the field rebuilt from the
constants stored beside it:

- **Signal.** The expected log-likelihood, in nats, by which the true matrix field beats
  a constant matrix on the observed matrix atoms at each efficiency. A field no method
  could tell from a constant cannot support any physics claim.
- **Identifiability.** The Cramer-Rao bound on the relative error of the capillary
  length, screening length and c_eq, for a fit to the observed matrix labels that knows
  the true precipitate geometry. No method can do better, and the Stage 5 network, which
  must segment the geometry itself, will do worse.
- **Consistency.** The screened diffusion equation's residual by automatic
  differentiation at random matrix atoms; each precipitate's mean surface value, by
  direct integration, against its Gibbs-Thomson value; and how far single surface points
  depart from that mean.

These set the development prior (open definition O8) before any test pattern is made.

Usage
-----
    python -m experiments.reconstruction.pilot.check_diffusion_prior
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from geom_mesh_net import paths
from geom_mesh_net.fields import physics

EFFICIENCIES = (0.37, 0.1)
STEP = 1e-3                 # relative step for derivatives in log parameters
PDE_POINTS = 2000
SURFACE_POINTS = 64
QUADRATURE_NODES = 200


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)])


def analyse(path):
    with np.load(path, allow_pickle=True) as d:
        coords, labels = d["coords"].item(), d["labels"]
        physics_values, parameters = d["physics"].item(), d["parameters"].item()
    return analyse_pattern(int(Path(path).stem.split("_")[-1]), coords, labels, physics_values, parameters)


def analyse_pattern(index, coords, labels, physics_values, parameters):
    """The measurements for one pattern, from its arrays and stored physical constants."""
    field = physics.DiffusionField.from_dict(physics_values)
    x = np.column_stack([coords[a] for a in "xyz"])[np.isin(labels, (0, 3))]
    c = field(x)
    row = {"pattern": index, "precipitates": int(len(field.radii)), "xi": field.xi, "ell": field.ell,
           "c_eq": field.c_eq, "c_inf": field.c_inf, "matrix_mean": parameters["matrix_mean"],
           "rb": parameters["rb"], "mean_error": float(abs(c.mean() - parameters["matrix_mean"]))}

    # Fields at perturbed constants; amplitudes are re-solved for each.
    def amplitudes(xi, c_eq, ell, c_inf):
        return physics.solve_field(field.centres, field.radii, xi, c_eq, ell, c_inf).amplitudes

    up, down = np.exp(STEP), np.exp(-STEP)
    base = (field.xi, field.c_eq, field.ell, field.c_inf)
    same_xi = np.column_stack([
        amplitudes(base[0], base[1], base[2] * up, base[3]), amplitudes(base[0], base[1], base[2] * down, base[3]),
        amplitudes(base[0], base[1] * up, base[2], base[3]), amplitudes(base[0], base[1] * down, base[2], base[3]),
        amplitudes(base[0], base[1], base[2], base[3] * up), amplitudes(base[0], base[1], base[2], base[3] * down),
    ])
    s = physics.kernel_sum(x, field.centres, field.radii, field.xi, same_xi)
    xi_up = base[3] + physics.kernel_sum(x, field.centres, field.radii, field.xi * up,
                                         amplitudes(field.xi * up, base[1], base[2], base[3]))
    xi_down = base[3] + physics.kernel_sum(x, field.centres, field.radii, field.xi * down,
                                           amplitudes(field.xi * down, base[1], base[2], base[3]))
    jacobian = np.column_stack([
        (s[:, 0] - s[:, 1]) / (2 * STEP),
        (xi_up - xi_down) / (2 * STEP),
        (s[:, 2] - s[:, 3]) / (2 * STEP),
        (base[3] * up + s[:, 4] - base[3] * down - s[:, 5]) / (2 * STEP),
    ])
    information = (jacobian / (c * (1 - c))[:, None]).T @ jacobian
    mean = c.mean()
    divergence = float(np.sum(c * np.log(c / mean) + (1 - c) * np.log((1 - c) / (1 - mean))))
    for eta in EFFICIENCIES:
        covariance = np.linalg.inv(eta * information)
        row[f"signal_nats_eta{eta}"] = eta * divergence
        row[f"crb_rel_ell_eta{eta}"] = float(np.sqrt(covariance[0, 0]))
        row[f"crb_rel_xi_eta{eta}"] = float(np.sqrt(covariance[1, 1]))
        row[f"crb_rel_c_eq_eta{eta}"] = float(np.sqrt(covariance[2, 2]))

    rng = np.random.default_rng(index)
    row["pde_residual"] = field.pde_residual(x[rng.choice(len(x), size=PDE_POINTS, replace=False)])
    amplitude = float(np.median(np.abs(field.surface_values - field.c_inf)))
    row["surface_mean_error_rel"] = float(np.max(np.abs(physics.surface_means(field, QUADRATURE_NODES) - field.surface_values)) / amplitude)
    points = (field.centres[:, None, :] + field.radii[:, None, None] * fibonacci_sphere(SURFACE_POINTS)[None]).reshape(-1, 3)
    owner = np.repeat(np.arange(len(field.radii)), SURFACE_POINTS)
    pointwise = np.abs(field(points) - field.surface_values[owner]) / amplitude
    row["surface_pointwise_rel_median"] = float(np.median(pointwise))
    row["surface_pointwise_rel_p90"] = float(np.percentile(pointwise, 90))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=paths.DIFFUSION_DIR)
    parser.add_argument("--output", type=Path,
                        default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "diffusion_prior.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    records = json.loads((args.data_dir / "patterns.json").read_text())
    files = [args.data_dir / f"clust_pattern_{r['pattern']}.npz" for r in records if r["split"] == "development"]
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(analyse, files), key=lambda r: r["pattern"])

    def quantiles(key):
        values = np.array([r[key] for r in rows], dtype=float)
        return {q: float(np.quantile(values, p)) for q, p in (("min", 0), ("q1", 0.25), ("median", 0.5),
                                                               ("q3", 0.75), ("max", 1))}

    summary = {key: quantiles(key) for key in rows[0] if key not in ("pattern",)}
    for eta in EFFICIENCIES:
        for name in ("ell", "xi"):
            key = f"crb_rel_{name}_eta{eta}"
            summary[f"share_{key}_below_0.25"] = float(np.mean([r[key] < 0.25 for r in rows]))
            summary[f"share_{key}_below_0.10"] = float(np.mean([r[key] < 0.10 for r in rows]))
    report = {"patterns": len(rows), "summary": summary, "per_pattern": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for key in ("precipitates", "signal_nats_eta0.37", "signal_nats_eta0.1", "crb_rel_ell_eta0.37", "crb_rel_xi_eta0.37",
                "crb_rel_c_eq_eta0.37", "crb_rel_ell_eta0.1", "crb_rel_xi_eta0.1", "pde_residual",
                "surface_mean_error_rel", "surface_pointwise_rel_median", "surface_pointwise_rel_p90", "mean_error"):
        print(f"{key:<30} " + "  ".join(f"{q} {v:.3g}" for q, v in summary[key].items()))
    for key, value in summary.items():
        if key.startswith("share_"):
            print(f"{key:<45} {value:.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
