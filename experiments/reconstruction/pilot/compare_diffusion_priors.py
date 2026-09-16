"""Which Stage 5 prior makes the physical constants recoverable? The comparison behind O8.

The first Stage 5 prior produced patterns whose capillary length no method could recover:
even a fit that knows the true precipitate geometry had a median Cramer-Rao bound of
about 45% on ell at efficiency 0.37. The information about ell comes from how the
Gibbs-Thomson surface values change with radius. Its size scales with c_eq, and it needs a
spread of radii, so a high supersaturation, a low matrix concentration or a narrow radius
spread hides it. The screening length is read from the shape of the depletion and
enrichment zones, which partly cancel when growing and dissolving precipitates mix.

Candidate priors are compared on the same 12 pattern indices each, generated in memory
with ``generate_diffusion_patterns.generate`` and measured with
``check_diffusion_prior.analyse_pattern``. Every candidate uses the generator's radius
floor, gap and box. A candidate sets the supersaturation either directly or through the
critical radius, as a fraction of cr. Candidate H is the prior the generator uses
(ROADMAP Stage 5, correction of 2026-09-16).

Usage
-----
    python -m experiments.reconstruction.pilot.compare_diffusion_priors
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.reconstruction import generate_diffusion_patterns as generator
from experiments.reconstruction.pilot import check_diffusion_prior as prior_check
from geom_mesh_net import paths

COMMON = {"cr": (3.0, 4.0), "volume_fraction": (0.08, 0.12), "rho_c": (0.4, 0.9)}
CANDIDATES = {
    "A first prior": {"rb": (0.1, 0.3), "matrix_mean": (0.02, 0.08), "ell": (1.0, 3.0), "supersaturation": (2.0, 4.0)},
    "B higher matrix concentration": {"rb": (0.1, 0.3), "matrix_mean": (0.05, 0.15), "ell": (1.0, 3.0),
                                      "supersaturation": (2.0, 4.0)},
    "C critical radius 0.8-1.2 cr, larger ell": {"rb": (0.1, 0.3), "matrix_mean": (0.02, 0.08), "ell": (2.0, 5.0),
                                                  "critical_radius_fraction": (0.8, 1.2)},
    "D = B + C": {"rb": (0.1, 0.3), "matrix_mean": (0.05, 0.15), "ell": (2.0, 5.0),
                  "critical_radius_fraction": (0.8, 1.2)},
    "E = D + wider radii": {"rb": (0.2, 0.4), "matrix_mean": (0.05, 0.15), "ell": (2.0, 5.0),
                            "critical_radius_fraction": (0.8, 1.2)},
    "F = B + larger ell, supersaturation 1.2-2": {"rb": (0.1, 0.3), "matrix_mean": (0.05, 0.15), "ell": (2.0, 5.0),
                                                  "supersaturation": (1.2, 2.0)},
    "G = F with supersaturation 1.5-3, ell 3-5": {"rb": (0.1, 0.3), "matrix_mean": (0.05, 0.15), "ell": (3.0, 5.0),
                                                  "supersaturation": (1.5, 3.0)},
    "H critical radius 0.6-0.8 cr (chosen)": {"rb": (0.2, 0.4), "matrix_mean": (0.05, 0.15), "ell": (2.0, 5.0),
                                              "critical_radius_fraction": (0.6, 0.8)},
    "I = F + wider radii": {"rb": (0.2, 0.4), "matrix_mean": (0.05, 0.15), "ell": (2.0, 5.0),
                            "supersaturation": (1.2, 2.0)},
}
PATTERNS = 12
ENTROPY = 20260919


def draw(name, index):
    ranges = {**COMMON, **CANDIDATES[name]}
    rng = np.random.default_rng(np.random.SeedSequence([ENTROPY, index]))
    p = {key: float(rng.uniform(*ranges[key])) for key in sorted(ranges)}
    p["pcp"] = p["matrix_mean"] + p["volume_fraction"] * (p["rho_c"] - p["matrix_mean"])
    if "critical_radius_fraction" in p:
        p["supersaturation"] = float(np.exp(p["ell"] / (p["critical_radius_fraction"] * p["cr"])))
    return p


def run(job):
    name, index = job
    try:
        arrays, record = generator.generate(index, draw(name, index))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return name, {"pattern": index, "error": f"{type(exc).__name__}: {exc}"}
    row = prior_check.analyse_pattern(index, arrays["coords"], arrays["labels"], arrays["physics"], arrays["parameters"])
    row.update(clipped_fraction=record["clipped_fraction"], field_max=record["field_range"][1],
               depleting_share=record["depleting_precipitates"] / record["precipitates"])
    return name, row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "diffusion_priors.json")
    args = parser.parse_args()
    jobs = [(name, i) for name in CANDIDATES for i in range(PATTERNS)]
    with Pool(args.workers, maxtasksperchild=4) as pool:
        results = pool.map(run, jobs)
    table = {}
    for name, row in results:
        table.setdefault(name, []).append(row)

    summary = {}
    for name, rows in table.items():
        ok = [r for r in rows if "error" not in r]
        median = lambda key: float(np.median([r[key] for r in ok]))  # noqa: E731
        summary[name] = {
            "prior": {**COMMON, **CANDIDATES[name]}, "patterns": len(ok), "failed": len(rows) - len(ok),
            "signal_nats_eta0.37_median": median("signal_nats_eta0.37"),
            "crb_rel_ell_eta0.37_median": median("crb_rel_ell_eta0.37"),
            "crb_rel_xi_eta0.37_median": median("crb_rel_xi_eta0.37"),
            "crb_rel_ell_eta0.1_median": median("crb_rel_ell_eta0.1"),
            "crb_rel_xi_eta0.1_median": median("crb_rel_xi_eta0.1"),
            "share_ell_eta0.37_below_0.25": float(np.mean([r["crb_rel_ell_eta0.37"] < 0.25 for r in ok])),
            "share_xi_eta0.37_below_0.25": float(np.mean([r["crb_rel_xi_eta0.37"] < 0.25 for r in ok])),
            "max_clipped_fraction": float(max(r["clipped_fraction"] for r in ok)),
            "max_field_value": float(max(r["field_max"] for r in ok)),
            "depleting_share_median": median("depleting_share"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"patterns_per_prior": PATTERNS, "summary": summary, "per_pattern": table},
                                      indent=1) + "\n")
    print(f"{'prior':<45} {'nats':>6} {'ell':>6} {'xi':>6} {'ell.1':>6} {'xi.1':>6} {'ell<.25':>7} {'xi<.25':>6} "
          f"{'clip':>6} {'max c':>6} {'deplete':>7}")
    for name, s in summary.items():
        print(f"{name:<45} {s['signal_nats_eta0.37_median']:>6.1f} {s['crb_rel_ell_eta0.37_median']:>6.3f} "
              f"{s['crb_rel_xi_eta0.37_median']:>6.3f} {s['crb_rel_ell_eta0.1_median']:>6.3f} "
              f"{s['crb_rel_xi_eta0.1_median']:>6.3f} {s['share_ell_eta0.37_below_0.25']:>7.2f} "
              f"{s['share_xi_eta0.37_below_0.25']:>6.2f} {s['max_clipped_fraction']:>6.4f} {s['max_field_value']:>6.3f} "
              f"{s['depleting_share_median']:>7.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
