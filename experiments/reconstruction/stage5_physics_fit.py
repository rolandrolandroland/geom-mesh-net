"""Stage 5.2: the diffusion law as a hard constraint, its controls, and Gate 5.2.

Stage 5.2 was corrected on development patterns before any test run
(``experiments/reconstruction/ROADMAP.md``, second Stage 5 correction). The soft-constraint
network of the first design could not recover the physical constants even when given the
true geometry (``pilot/check_soft_pinn.py``; walkthrough E16). The method under test is
now the analytic diffusion family of ``geom_mesh_net.fields.physics`` with trainable
constants, fitted by maximum likelihood on precipitates detected in the data.

For each cell (pattern, efficiency, matrix kind):

1. **Observe.** Atoms are kept by ``benchmark.thinning_mask`` with the Stage 5 entropy; the
   same mask serves every kind of the same pattern.
2. **Split first.** Observed atoms are split by one uniform number per atom, shared by all
   kinds: 20% are held back for the acceptance rule and never used again until it is applied,
   10% stop the network early, and 70% train both models. Nothing that touches a label of the
   held-back 20% may enter the models they judge, so the split comes before every
   label-dependent step below.
3. **Detect.** On the training and early-stopping atoms only: the B1 field at the
   cross-validated bandwidth is segmented by significance above the matrix level
   (``cluster_extraction.segment(method="significance")``), and overlapping spheres are merged
   or shrunk to a gap of ``RESOLVE_GAP``.
4. **Admit.** Those atoms enter the matrix by the conservative domain rule, each judged with
   its own label left out (``cluster_extraction.matrix_atoms``), so the domain does not select
   on the labels it will be fitted to. Held-back atoms are admitted by the same rule applied
   as a plain voxel lookup: their labels never entered the field, so there is nothing to leave
   out.
5. **Fit.** The physics model, from ``PHYSICS_STARTS``, by L-BFGS; the unconstrained network,
   the same SIREN as the first design with both penalties at zero (``NETWORK``).
6. **Score.** Matrix excess loss on the unobserved atoms of the true matrix, for the physics
   fit, the network, B1 and the constant matrix; the acceptance rule, with a cross-validated
   kernel smoother of the matrix labels reported beside it as a diagnostic; the constants against
   the truth, normalised by the pattern's Cramer-Rao bound (correctly specified cells
   only); detection and admission statistics. For correctly specified cells the physics is
   also fitted with the true geometry to the observed atoms of the true matrix, the upper
   bound the method approaches.

**Acceptance rule.** The physics is accepted for a cell, and its constants reported, only if
all three hold:

1. its log loss on the acceptance atoms is no greater than the unconstrained network's;
2. its log loss on the acceptance atoms is no greater than the constant matrix's, the mean of
   the training atoms: a law that predicts no better than no structure has no support;
3. no fitted constant lies at a bound of its search range (``ParametricDiffusionField.BOUNDS``,
   within 1%): an estimate at the edge of its range is not identified by the data.

The first condition is the rule as first designed. The second and third were added on
development patterns, where the first alone accepted most misspecified cells (second Stage 5
correction). The network's early stopping also considers the untrained network, so it cannot
end worse on its early-stopping atoms than a near-constant field.

**Gate 5.2**, on the test scope, with ``GATE`` thresholds:

- (i) the physics fit has a lower matrix excess loss than the network in at least two-thirds
  of correctly specified cells;
- (ii) for each of ell and xi, the median over correctly specified cells of
  |relative error| / (the pattern's Cramer-Rao bound) is below ``GATE["normalised_error"]``;
- (iii) the rule accepts in at least two-thirds of correctly specified cells, and rejects in
  at least two-thirds of misspecified cells, pooled over both controls. If (iii) fails, no
  physics-informed claim is made, whatever (i) and (ii) show.

Results are written after every cell, and a rerun skips the cells already in the output, but
only if its design matches the one recorded in the file. The gate is reported only for a
complete run: every cell of the split's scope, for every matrix kind, with no errors.

Usage
-----
    python -m experiments.reconstruction.stage5_physics_fit --split development
    python -m experiments.reconstruction.stage5_physics_fit --split test
    python -m experiments.reconstruction.stage5_physics_fit --split development --limit 1 --output /tmp/s52_smoke.json
"""

import argparse
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch

from experiments.reconstruction import benchmark as bm
from experiments.reconstruction import generate_misspecified_patterns as gm
from experiments.reconstruction import stage5_simulator
from experiments.reconstruction.generate_diffusion_patterns import THINNING_ENTROPY, VOLUME
from experiments.reconstruction.pilot.check_diffusion_prior import analyse_pattern
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.fields import cluster_extraction as ce
from geom_mesh_net.fields import physics
from geom_mesh_net.neural import implicit

SCOPE = {
    "development": {0.37: tuple(range(0, 8)), 0.1: tuple(range(0, 4))},
    "test": {0.37: tuple(range(50, 74)), 0.1: tuple(range(50, 62))},
}
KINDS = ("diffusion", "smoothed_noise", "shuffled_surface")
SPLIT = {"acceptance": 0.2, "early_stopping": 0.1}
SPLIT_ENTROPY = 20260927
PROTOCOL = "split-first"   # the atoms are split before any label-dependent step (harness correction, 2026-09-17)
RESOLVE_GAP = 0.5
PHYSICS_STARTS = ((None, None), (1.0, 3.0), (6.0, 12.0))  # (ell, xi); None: mean radius and sink-strength xi
BOUND_TOLERANCE = 1.01   # a constant within 1% of a search bound counts as at the bound
SMOOTHER_BANDWIDTHS = (1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0, 18.0)
SMOOTHER_FOLDS = 5
NETWORK = dict(w0=3.0, width=128, depth=3, lr=1e-4, steps=3000, eval_every=10, patience=300)
GATE = {"fraction": 2 / 3, "normalised_error": 2.0}
NAMES = ("c_eq", "ell", "xi", "c_inf")
CLIP = 1e-6
_BOUNDS = {}


def log_loss(y, q):
    return fb.log_loss(y, np.clip(q, CLIP, 1 - CLIP))


def load_cell(index, kind):
    """Coordinates, guest labels, oracle probabilities, true-matrix flags, stored physics and parameters."""
    with np.load(paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        coords, labels = d["coords"].item(), d["labels"]
        truth, parameters = d["physics"].item(), d["parameters"].item()
    x = np.column_stack([coords[a] for a in "xyz"])
    p_star, inside = stage5_simulator.load_oracle(index)
    matrix = np.isin(labels, (0, 3))
    if not np.array_equal(~inside, matrix):
        raise RuntimeError(f"pattern {index}: oracle membership disagrees with the labels")
    if kind != "diffusion":
        labels, values, _ = gm.load(kind, index)
        p_star = p_star.copy()
        p_star[matrix] = np.clip(values, 0.0, 1.0)
    return x, np.isin(labels, (2, 3)), p_star, inside, coords, truth, parameters


def bound(index, eta, coords, labels, truth, parameters):
    key = (index, eta)
    if key not in _BOUNDS:
        row = analyse_pattern(index, coords, labels, truth, parameters)
        for e in (0.37, 0.1):
            _BOUNDS[(index, e)] = {"ell": row[f"crb_rel_ell_eta{e}"], "xi": row[f"crb_rel_xi_eta{e}"]}
    return _BOUNDS[key]


def fit_physics(centres, radii, x, y):
    """The analytic family by maximum likelihood from each start; the best by training loss."""
    mean = float(np.mean(y))
    if len(radii) == 0:
        return None, {"constants": None, "train_loss": log_loss(y, np.full(len(y), mean)), "starts": 0}
    best = None
    for ell0, xi0 in PHYSICS_STARTS:
        ell0 = float(np.mean(radii)) if ell0 is None else ell0
        xi0 = physics.screening_length(radii, VOLUME) if xi0 is None else xi0
        model = implicit.ParametricDiffusionField(centres, radii, c_eq=mean / 2, ell=ell0, xi=xi0, c_inf=mean)
        record = implicit.fit_parametric(model, x, y)
        if not np.isfinite(record["train_loss"]):
            continue
        if best is None or record["train_loss"] < best[1]["train_loss"]:
            best = (model, record)
    return best


def smoother_loss(x, y, train, target, seed):
    """Loss on ``target`` of a kernel smoother of the training labels, its bandwidth or a constant chosen by CV.

    A diagnostic beside the acceptance rule: a flexible model that nests the constant matrix.
    Returns (loss, bandwidth), with bandwidth None when cross-validation chose the constant.
    """
    fold = np.random.default_rng(seed).integers(0, SMOOTHER_FOLDS, len(train))
    totals = np.zeros(len(SMOOTHER_BANDWIDTHS) + 1)
    for k in range(SMOOTHER_FOLDS):
        fit, held = train[fold != k], train[fold == k]
        smoother = fb.SmoothedCounts(x[fit], y[fit].astype(bool), bandwidths=SMOOTHER_BANDWIDTHS)
        for j in range(len(SMOOTHER_BANDWIDTHS)):
            totals[j] += log_loss(y[held], smoother.fixed(x[held], j)) * len(held)
        totals[-1] += log_loss(y[held], np.full(len(held), y[fit].mean())) * len(held)
    choice = int(np.argmin(totals))
    if choice == len(SMOOTHER_BANDWIDTHS):
        return log_loss(y[target], np.full(len(target), y[train].mean())), None
    smoother = fb.SmoothedCounts(x[train], y[train].astype(bool), bandwidths=SMOOTHER_BANDWIDTHS)
    return log_loss(y[target], smoother.fixed(x[target], choice)), SMOOTHER_BANDWIDTHS[choice]


def at_bounds(constants):
    """Names of fitted constants within ``BOUND_TOLERANCE`` of a bound of their search range; all of them if there was no fit."""
    if constants is None:
        return list(NAMES)
    bounds = implicit.ParametricDiffusionField.BOUNDS
    return [k for k, v in constants.items() if v <= bounds[k][0] * BOUND_TOLERANCE or v >= bounds[k][1] / BOUND_TOLERANCE]


def physics_values(model, points, mean):
    if model is None:
        return np.full(len(points), mean)
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(points[s:s + 20000], dtype=torch.float64)).numpy()
                               for s in range(0, len(points), 20000)])


def run_cell(index, eta, kind, device):
    started = time.perf_counter()
    x, guest, p_star, inside, coords, truth, parameters = load_cell(index, kind)
    y = guest.astype(float)
    mask = bm.thinning_mask(index, eta, len(x), entropy=THINNING_ENTROPY)
    observed = np.flatnonzero(mask)
    scored = np.flatnonzero(~mask & ~inside)
    oracle_loss = log_loss(y[scored], p_star[scored])

    # The split comes first: nothing derived from the held-back atoms' labels may reach the
    # models the acceptance rule then compares on them.
    u = np.random.default_rng(np.random.SeedSequence([SPLIT_ENTROPY, index, int(round(eta * 1000))])).random(len(x))
    held_back = observed[u[observed] < SPLIT["acceptance"]]
    fitting = observed[u[observed] >= SPLIT["acceptance"]]

    bandwidth = fb.fit_baselines(x[fitting], guest[fitting], folds=5, seed=index).bandwidth
    guests, atoms = ce.smoothed_counts(x[fitting], guest[fitting], bandwidth)
    squared = ce.squared_kernel_counts(x[fitting], bandwidth)
    segmentation = ce.segment(ce.guest_fraction(guests, atoms), atoms, method="significance", squared=squared)
    admitted_fitting = fitting[ce.matrix_atoms(x[fitting], guest[fitting], guests, atoms, bandwidth, segmentation)]
    acceptance = held_back[segmentation.in_matrix(x[held_back])]
    admitted = np.sort(np.concatenate([admitted_fitting, acceptance]))
    centres, radii = ce.resolve_overlaps(segmentation.centres, segmentation.radii, gap=RESOLVE_GAP)

    early = admitted_fitting[u[admitted_fitting] < SPLIT["acceptance"] + SPLIT["early_stopping"]]
    train = admitted_fitting[u[admitted_fitting] >= SPLIT["acceptance"] + SPLIT["early_stopping"]]
    mean = float(y[train].mean())

    t = time.perf_counter()
    best = fit_physics(centres, radii, x[train], y[train])
    model, record = best if best is not None else (None, {"constants": None})
    physics_seconds = round(time.perf_counter() - t, 1)

    torch.manual_seed(0)
    network = implicit.DiffusionPINN(c0=mean, c_eq=mean / 2, ell=1.0, xi=5.0, c_inf=mean, w0=NETWORK["w0"],
                                     width=NETWORK["width"], depth=NETWORK["depth"])
    network_record = implicit.fit_pinn(network, x[train], y[train], x[early], y[early], steps=NETWORK["steps"],
                                       lr=NETWORK["lr"], eval_every=NETWORK["eval_every"], patience=NETWORK["patience"],
                                       device=device, evaluate_initial=True)

    q_physics = physics_values(model, x[scored], mean)
    q_network = implicit.predict(network, x[scored], device=device)
    q_b1 = fb.SmoothedCounts(x[fitting], guest[fitting]).fixed(x[scored], fb.BANDWIDTHS.index(bandwidth))
    accept_physics = log_loss(y[acceptance], physics_values(model, x[acceptance], mean))
    accept_network = log_loss(y[acceptance], implicit.predict(network, x[acceptance], device=device))
    accept_constant = log_loss(y[acceptance], np.full(len(acceptance), mean))
    accept_smoother, smoother_bandwidth = smoother_loss(x, y, train, acceptance, seed=index)
    at_bound = at_bounds(record["constants"])

    rows, cols = ce.match_precipitates(np.asarray(truth["centres"]), np.asarray(truth["radii"]), centres)
    true_radii = np.asarray(truth["radii"])
    matrix_admitted = admitted[~inside[admitted]]
    z = (y[matrix_admitted].sum() - p_star[matrix_admitted].sum()) / np.sqrt(
        np.sum(p_star[matrix_admitted] * (1 - p_star[matrix_admitted])))
    cell = {
        "pattern": index, "efficiency": eta, "kind": kind, "bandwidth": bandwidth,
        "detection": {"found": int(len(radii)), "true": int(len(true_radii)), "recall": len(rows) / len(true_radii),
                      "precision": len(rows) / max(len(radii), 1),
                      "radius_ratio_median": float(np.median(radii[cols] / true_radii[rows])) if len(rows) else None},
        "admission": {"atoms": int(len(admitted)), "share_of_observed_matrix": float(len(matrix_admitted) / np.sum(~inside[observed])),
                      "precipitate_atoms": int(np.sum(inside[admitted])), "bias_z": float(z),
                      "held_back_atoms": int(len(acceptance)), "detection_atoms": int(len(fitting))},
        "atoms": {"train": int(len(train)), "early_stopping": int(len(early)), "acceptance": int(len(acceptance)),
                  "scored": int(len(scored))},
        "matrix_excess": {"physics": log_loss(y[scored], q_physics) - oracle_loss,
                          "network": log_loss(y[scored], q_network) - oracle_loss,
                          "b1": log_loss(y[scored], q_b1) - oracle_loss,
                          "constant": log_loss(y[scored], np.full(len(scored), mean)) - oracle_loss},
        "acceptance": {"physics_loss": accept_physics, "network_loss": accept_network, "constant_loss": accept_constant,
                       "smoother_loss": accept_smoother, "smoother_bandwidth": smoother_bandwidth,
                       "difference_nats": (accept_network - accept_physics) * len(acceptance),
                       "beats_network": bool(accept_physics <= accept_network),
                       "beats_constant": bool(accept_physics <= accept_constant),
                       "at_bound": at_bound,
                       "accepted": bool(accept_physics <= accept_network and accept_physics <= accept_constant and not at_bound)},
        "physics": {"constants": record["constants"], "seconds": physics_seconds},
        "network": {"best_step": network_record["best_step"], "seconds": network_record["seconds"]},
    }
    if kind == "diffusion":
        crb = bound(index, eta, coords, np.where(inside, 2, 0), truth, parameters)
        if record["constants"] is not None:
            relative = {k: record["constants"][k] / truth[k] - 1 for k in NAMES}
            cell["physics"]["relative_error"] = relative
            cell["physics"]["normalised_error"] = {k: abs(relative[k]) / crb[k] for k in ("ell", "xi")}
        cell["bound"] = crb
        true_train = observed[~inside[observed] & (u[observed] >= SPLIT["acceptance"] + SPLIT["early_stopping"])]
        field = physics.DiffusionField.from_dict(truth)
        oracle_best = fit_physics(field.centres, field.radii, x[true_train], y[true_train])
        q_true = physics_values(oracle_best[0], x[scored], float(y[true_train].mean()))
        cell["true_geometry"] = {"constants": oracle_best[1]["constants"],
                                 "relative_error": {k: oracle_best[1]["constants"][k] / truth[k] - 1 for k in NAMES},
                                 "matrix_excess": log_loss(y[scored], q_true) - oracle_loss}
    cell["seconds"] = round(time.perf_counter() - started, 1)
    return cell


def expected_cells(split):
    """Every cell the split's scope requires: (kind, efficiency, pattern)."""
    return {(kind, eta, index) for kind in KINDS for eta, indices in SCOPE[split].items() for index in indices}


def gate(cells, split):
    """The Gate 5.2 verdict, or a report of what is missing.

    A verdict is given only for a complete run: every cell of the scope, for every matrix
    kind, fitted without error. A partial run cannot pass.
    """
    missing = sorted(expected_cells(split) - {(c["kind"], c["efficiency"], c["pattern"]) for c in cells if "error" not in c})
    errors = [(c["kind"], c["efficiency"], c["pattern"]) for c in cells if "error" in c]
    complete = not missing and not errors
    cells = [c for c in cells if "error" not in c]
    correct = [c for c in cells if c["kind"] == "diffusion"]
    wrong = [c for c in cells if c["kind"] != "diffusion"]
    if not correct:
        return {"split": split, "complete": False, "missing_cells": len(missing), "errored_cells": len(errors),
                "verdict": "incomplete", "passed": False, "physics_claim_permitted": False}
    wins = np.mean([c["matrix_excess"]["physics"] < c["matrix_excess"]["network"] for c in correct])
    normalised = {k: float(np.median([c["physics"]["normalised_error"][k] if "normalised_error" in c["physics"] else np.inf
                                      for c in correct])) for k in ("ell", "xi")}
    accepted = float(np.mean([c["acceptance"]["accepted"] for c in correct]))
    rejected = float(np.mean([not c["acceptance"]["accepted"] for c in wrong])) if wrong else None
    conditions = {
        "i_physics_beats_network": {"measured": float(wins), "required": GATE["fraction"], "passed": bool(wins >= GATE["fraction"])},
        "ii_median_normalised_error": {"measured": normalised, "required_below": GATE["normalised_error"],
                                       "passed": bool(all(v < GATE["normalised_error"] for v in normalised.values()))},
        "iii_acceptance_rule": {"accepted_correct": accepted, "rejected_misspecified": rejected, "required": GATE["fraction"],
                                "passed": bool(accepted >= GATE["fraction"] and rejected is not None and rejected >= GATE["fraction"])},
    }
    by_kind = {kind: float(np.mean([not c["acceptance"]["accepted"] for c in wrong if c["kind"] == kind]))
               for kind in KINDS[1:] if any(c["kind"] == kind for c in wrong)}
    return {"split": split, "complete": complete,
            "missing_cells": len(missing), "missing": [list(m) for m in missing[:20]], "errored_cells": len(errors),
            "cells": {"correct": len(correct), "misspecified": len(wrong)}, "conditions": conditions,
            "rejected_by_kind": by_kind,
            "verdict": ("passed" if all(c["passed"] for c in conditions.values()) else "failed") if complete else "incomplete",
            "passed": bool(complete and all(c["passed"] for c in conditions.values())),
            "physics_claim_permitted": bool(complete and conditions["iii_acceptance_rule"]["passed"])}


def worker(task):
    index, eta, kind, device, threads = task
    torch.set_num_threads(threads)
    try:
        return run_cell(index, eta, kind, device)
    except Exception as exc:  # noqa: BLE001 - recorded in the results, not swallowed
        return {"pattern": index, "efficiency": eta, "kind": kind, "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=tuple(SCOPE), required=True)
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="cells per kind and efficiency, for smoke runs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or paths.RECONSTRUCTION_DIR / "results" / f"stage5_physics_fit_{args.split}.json"
    kinds = args.kinds.split(",")

    report = json.loads(output.read_text()) if output.exists() else {"cells": []}
    done = {(c["pattern"], c["efficiency"], c["kind"]) for c in report["cells"] if "error" not in c}
    report["cells"] = [c for c in report["cells"] if "error" not in c]
    tasks = [(index, eta, kind, args.device, max(1, 6 // args.workers))
             for kind in kinds for eta, indices in SCOPE[args.split].items() for index in indices[:args.limit]
             if (index, eta, kind) not in done]
    design = {"protocol": PROTOCOL, "scope": {k: {str(e): list(v) for e, v in s.items()} for k, s in SCOPE.items()}, "kinds": KINDS,
                        "split": SPLIT, "resolve_gap": RESOLVE_GAP, "physics_starts": PHYSICS_STARTS, "network": NETWORK,
                        "acceptance_rule": "no worse than the network and the constant on acceptance atoms, no constant at a bound",
                        "bounds": implicit.ParametricDiffusionField.BOUNDS, "bound_tolerance": BOUND_TOLERANCE,
                        "gate": GATE, "detection": {"method": "significance", "region_z": ce.REGION_Z, "seed_z": ce.SEED_Z,
                                                    "exclusion_z": ce.EXCLUSION_Z, "footprint": ce.FOOTPRINT,
                                                    "min_volume": ce.MIN_VOLUME, "exclusion_dilation": ce.EXCLUSION_DILATION},
                        "entropy": {"thinning": THINNING_ENTROPY, "split": SPLIT_ENTROPY}}
    digest = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:16]
    if report["cells"] and report.get("design_hash") != digest:
        raise SystemExit(f"{output} holds cells from a different design ({report.get('design_hash')} != {digest}); "
                         "delete it or choose another --output rather than mixing runs")
    report["design"], report["design_hash"] = design, digest
    report["environment"] = {"torch": torch.__version__, "device": args.device}
    print(f"{len(tasks)} cells to run, {len(done)} already done", flush=True)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers) as pool:
        for n, cell in enumerate(pool.imap_unordered(worker, tasks), 1):
            report["cells"].append(cell)
            report["cells"].sort(key=lambda c: (KINDS.index(c["kind"]), -c["efficiency"], c["pattern"]))
            report["gate"] = gate([c for c in report["cells"] if "error" not in c], args.split)
            output.write_text(json.dumps(report, indent=1) + "\n")
            if "error" in cell:
                print(f"  {n}/{len(tasks)} pattern {cell['pattern']} eta {cell['efficiency']} {cell['kind']}: {cell['error']}", flush=True)
                continue
            e = cell["matrix_excess"]
            constants = cell["physics"].get("normalised_error")
            print(f"  {n}/{len(tasks)} pattern {cell['pattern']} eta {cell['efficiency']} {cell['kind']}: excess physics "
                  f"{e['physics']:.5f} network {e['network']:.5f} constant {e['constant']:.5f}; accepted "
                  f"{cell['acceptance']['accepted']} ({cell['acceptance']['difference_nats']:+.1f} nats); "
                  f"found {cell['detection']['found']}/{cell['detection']['true']}"
                  + (f"; normalised error ell {constants['ell']:.2f} xi {constants['xi']:.2f}" if constants else "")
                  + f" ({cell['seconds']} s)", flush=True)
    report["seconds"] = round(time.perf_counter() - started)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["gate"], indent=1))


if __name__ == "__main__":
    main()
