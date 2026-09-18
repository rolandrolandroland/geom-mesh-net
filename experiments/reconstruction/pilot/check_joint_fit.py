"""Does fitting the precipitates with the law beat detecting them first?

Gate 5.2 failed on the capillary length: detected radii are noisy, and ell is read from how
surface concentration varies with radius, so the noise flattens it (walkthrough E17). The
radius information the detection throws away sits in the atoms inside each precipitate, which
the Stage 5.2 fit never sees: it admits matrix atoms only.

This pilot compares, on development patterns:

- **detect then fit** (Stage 5.2): precipitates from the smoothed field, then the four
  constants by maximum likelihood on admitted matrix atoms;
- **joint fit** (``fields/joint_fit.py``): the same detection only as a starting point, then
  the constants, every radius and the interior profile fitted together to *every* observed
  atom, with the centres held where detection put them;
- **joint fit, geometry free**: the same, with the centres fitted too and the interior profile
  piecewise linear rather than a power law. Holding the centres fixed turned out to be what
  biased the capillary length: on development pattern 3 the error falls from +46% to +10% when
  they are freed;
- **the true geometry**: the constants fitted with the simulator's own centres and radii, the
  upper bound of E17.

Each is scored by the relative error of ell and xi, and by that error divided by the
Cramer-Rao bound with the geometry unknown (``check_geometry_bound.py``) as well as the older
bound that assumes it known (``check_diffusion_prior.py``). The joint fit is also scored on
how far it moves the radii toward the truth.

The interface width ``WIDTH`` is an input, not a fitted quantity, and the fit is sensitive to
it: on development pattern 7 at efficiency 0.37, widths of 0.2 and 0.3 nm recovered ell within
9%, and 0.5 nm drove the screening length to its bound. ``--widths`` sweeps it.

Usage
-----
    python -m experiments.reconstruction.pilot.check_joint_fit
    python -m experiments.reconstruction.pilot.check_joint_fit --patterns 7 --widths 0.2,0.3,0.5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.reconstruction import benchmark as bm
from experiments.reconstruction import stage5_physics_fit as s52
from experiments.reconstruction.generate_diffusion_patterns import THINNING_ENTROPY, VOLUME
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.fields import cluster_extraction as ce
from geom_mesh_net.fields import joint_fit as jf
from geom_mesh_net.fields import physics

SCOPE = {0.37: (0, 1, 2, 3, 4, 5, 6, 7), 0.1: (0, 1, 2, 3)}
WIDTH = 0.3          # nm; the interface width measured from the simulator's own oracle
KNOTS = 6            # segments of the piecewise-linear interior profile, for the free variant
NAMES = ("c_eq", "ell", "xi", "c_inf")


def detected_geometry(x, guest, fitting, index):
    """Stage 5.2's detection, on the fitting atoms only."""
    bandwidth = fb.fit_baselines(x[fitting], guest[fitting], folds=5, seed=index).bandwidth
    guests, atoms = ce.smoothed_counts(x[fitting], guest[fitting], bandwidth)
    squared = ce.squared_kernel_counts(x[fitting], bandwidth)
    segmentation = ce.segment(ce.guest_fraction(guests, atoms), atoms, method="significance", squared=squared)
    admitted = fitting[ce.matrix_atoms(x[fitting], guest[fitting], guests, atoms, bandwidth, segmentation)]
    centres, radii = ce.resolve_overlaps(segmentation.centres, segmentation.radii, gap=s52.RESOLVE_GAP)
    return centres, radii, admitted, bandwidth


def radius_error(found_centres, found_radii, true_centres, true_radii):
    """Median |relative error| of matched radii, and the share of precipitates matched."""
    rows, cols = ce.match_precipitates(true_centres, true_radii, found_centres)
    if not len(rows):
        return None, 0.0
    return float(np.median(np.abs(found_radii[cols] / true_radii[rows] - 1))), len(rows) / len(true_radii)


def analyse(index, eta, widths, bounds_known, bounds_unknown, by_width=None):
    x, guest, p_star, inside, coords, truth, parameters = s52.load_cell(index, "diffusion")
    y = guest.astype(float)
    field = physics.DiffusionField.from_dict(truth)
    mask = bm.thinning_mask(index, eta, len(x), entropy=THINNING_ENTROPY)
    observed = np.flatnonzero(mask)
    scored = np.flatnonzero(~mask & ~inside)
    oracle_loss = s52.log_loss(y[scored], p_star[scored])
    u = np.random.default_rng(np.random.SeedSequence([s52.SPLIT_ENTROPY, index, int(round(eta * 1000))])).random(len(x))
    fitting = observed[u[observed] >= s52.SPLIT["acceptance"]]

    started = time.perf_counter()
    centres, radii, admitted, bandwidth = detected_geometry(x, guest, fitting, index)
    train = admitted[u[admitted] >= s52.SPLIT["acceptance"] + s52.SPLIT["early_stopping"]]
    detected_error, recall = radius_error(centres, radii, np.asarray(truth["centres"]), np.asarray(truth["radii"]))
    row = {"pattern": index, "efficiency": eta, "precipitates": {"true": int(len(field.radii)), "detected": int(len(radii)),
                                                                 "recall": recall, "radius_error": detected_error},
           "bounds": {"geometry_known": bounds_known, "geometry_unknown": bounds_unknown},
           "truth": {k: getattr(field, k) for k in NAMES}, "fits": {}}

    def score(name, constants, probabilities, extra=None, unknown=None):
        relative = {k: constants[k] / getattr(field, k) - 1 for k in NAMES}
        entry = {"constants": constants, "relative_error": relative,
                 "matrix_excess": s52.log_loss(y[scored], probabilities) - oracle_loss}
        for label, bound in (("geometry_known", bounds_known),
                             ("geometry_unknown", bounds_unknown if unknown is None else unknown)):
            if bound:
                entry[f"normalised_{label}"] = {k: abs(relative[k]) / bound[k] for k in ("ell", "xi") if bound.get(k)}
        row["fits"][name] = {**entry, **(extra or {})}

    mean = float(y[train].mean())
    best = s52.fit_physics(centres, radii, x[train], y[train])
    if best is not None and best[1]["constants"] is not None:
        score("detect_then_fit", best[1]["constants"], s52.physics_values(best[0], x[scored], mean))

    for width in widths:
        for name, options in (("joint_fit", {}), ("joint_free", dict(knots=KNOTS, free_centres=True))):
            model = jf.JointDiffusionModel(centres, radii, c_eq=mean / 2, ell=float(np.mean(radii)),
                                           xi=physics.screening_length(radii, VOLUME), c_inf=mean, width=width,
                                           **options)
            record = jf.fit_joint(model, x[fitting], y[fitting])
            found = model.centres().detach().numpy()
            fitted_error, fitted_recall = radius_error(found, record["radii"], np.asarray(truth["centres"]),
                                                       np.asarray(truth["radii"]))
            score(f"{name}_w{width}", record["constants"], jf.predict(model, x[scored], matrix_only=True),
                  {"shape": record["shape"], "radius_error": fitted_error, "recall": fitted_recall,
                   "train_loss": record["train_loss"], "seconds": record["seconds"]},
                  unknown=(by_width or {}).get(width, bounds_unknown))   # each fit against the bound at its own width

    true_train = observed[~inside[observed] & (u[observed] >= s52.SPLIT["acceptance"] + s52.SPLIT["early_stopping"])]
    oracle_best = s52.fit_physics(field.centres, field.radii, x[true_train], y[true_train])
    score("true_geometry", oracle_best[1]["constants"],
          s52.physics_values(oracle_best[0], x[scored], float(y[true_train].mean())))
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", default=None, help="comma-separated; the scope's patterns by default")
    parser.add_argument("--efficiencies", default=",".join(str(e) for e in SCOPE))
    parser.add_argument("--widths", default=str(WIDTH))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "joint_fit.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    widths = [float(w) for w in args.widths.split(",")]
    efficiencies = [float(e) for e in args.efficiencies.split(",")]

    known = {}
    prior = paths.RECONSTRUCTION_DIR / "pilot" / "results" / "diffusion_prior.json"
    if prior.exists():
        for r in json.loads(prior.read_text())["per_pattern"]:
            for eta in efficiencies:
                known[(r["pattern"], eta)] = {"ell": r.get(f"crb_rel_ell_eta{eta}"), "xi": r.get(f"crb_rel_xi_eta{eta}")}
    unknown, unknown_by_width = {}, {}
    geometry = paths.RECONSTRUCTION_DIR / "pilot" / "results" / "geometry_bound.json"
    if geometry.exists():
        report = json.loads(geometry.read_text())
        for r in report["per_pattern"]:
            for width in r["widths"]:
                for eta in efficiencies:
                    entry = r["widths"][width].get(str(eta), {}).get("radii_unknown")
                    if not entry:
                        continue
                    limits = {"ell": entry["ell"], "xi": entry["xi"]}
                    unknown_by_width[(r["pattern"], eta, float(width))] = limits
                    if abs(float(width) - WIDTH) < 1e-9:
                        unknown[(r["pattern"], eta)] = limits

    rows, started = [], time.perf_counter()
    for eta in efficiencies:
        patterns = [int(i) for i in args.patterns.split(",")] if args.patterns else list(SCOPE[eta])
        for index in patterns:
            row = analyse(index, eta, widths, known.get((index, eta)), unknown.get((index, eta)),
                          {w: unknown_by_width.get((index, eta, w)) for w in widths})
            rows.append(row)
            detected = row["fits"].get("detect_then_fit", {}).get("relative_error", {})
            joint = row["fits"].get(f"joint_fit_w{widths[0]}", {})
            free = row["fits"].get(f"joint_free_w{widths[0]}", {})
            print(f"pattern {index} eta {eta}: detect-then-fit ell {detected.get('ell', float('nan')):+.2f} "
                  f"xi {detected.get('xi', float('nan')):+.2f} | joint ell {joint['relative_error']['ell']:+.2f} "
                  f"xi {joint['relative_error']['xi']:+.2f} | geometry free ell "
                  f"{free['relative_error']['ell']:+.2f} xi {free['relative_error']['xi']:+.2f} | true geometry ell "
                  f"{row['fits']['true_geometry']['relative_error']['ell']:+.2f} | radii "
                  f"{row['precipitates']['radius_error']:.3f} -> {joint.get('radius_error', float('nan')):.3f} -> "
                  f"{free.get('radius_error', float('nan')):.3f} "
                  f"(recall {row['precipitates']['recall']:.2f}) [{row['seconds']} s]", flush=True)
            args.output.write_text(json.dumps({"design": {"scope": {str(k): list(v) for k, v in SCOPE.items()},
                                                          "widths": widths, "knots": KNOTS,
                                                          "split_protocol": s52.PROTOCOL},
                                               "per_cell": rows, "seconds": round(time.perf_counter() - started)},
                                              indent=1) + "\n")
    print(f"wrote {args.output} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
