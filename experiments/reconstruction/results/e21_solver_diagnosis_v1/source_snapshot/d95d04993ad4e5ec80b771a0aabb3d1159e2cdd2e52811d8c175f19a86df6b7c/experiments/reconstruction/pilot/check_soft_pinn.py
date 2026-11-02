"""Can the soft-constraint PINN of the first Stage 5.2 design recover the diffusion constants?

Stage 5.2 was designed as a standard physics-informed inverse problem
(``experiments/reconstruction/ROADMAP.md``): a SIREN field fitted to the observed matrix
labels, with the screened diffusion equation and the Gibbs-Thomson condition as penalties
and c_eq, ell, xi and c_inf trainable (``geom_mesh_net.neural.implicit.DiffusionPINN``).
On development patterns it failed. This pilot reproduces the failure with nothing else to
blame: every fit is given the true precipitate geometry and the observed atoms of the true
matrix, so neither segmentation nor the choice of matrix domain enters.

For each development pattern in ``PATTERNS``, at efficiency 0.37:

- **Settings.** The network is fitted under each of ``SETTINGS``: without penalties (the
  unconstrained control) at two frequencies, and with penalties across their weight, both
  residual forms, a lower frequency, a slower learning rate for the constants, and a
  warm-up with the constants frozen. Each fit records the trajectory of the constants and
  losses, and scores its early-stopped field by the matrix excess loss on the unobserved
  matrix atoms and by the loss on held-out observed atoms.
- **The loss at fixed constants.** The network is fitted with the constants held at their
  true values, and again at flat values: ell near zero and c_eq = c_inf = the observed
  matrix mean, under which a uniform field satisfies both penalties exactly. If the loss
  ends lower at the flat values, the objective itself prefers the wrong constants, and no
  optimiser setting can rescue it.
- **References.** The analytic family (``ParametricDiffusionField``) fitted by maximum
  likelihood to the same training atoms with the true geometry; the constant matrix; and
  the data margin, the per-atom training loss by which the true field beats that constant.

Training on MPS in float32 is not repeatable from run to run, so conclusions rest on
what holds across patterns and settings, not on any single number.

Usage
-----
    python -m experiments.reconstruction.pilot.check_soft_pinn
    python -m experiments.reconstruction.pilot.check_soft_pinn --patterns 0 --steps 40 --compare-steps 40 \
        --output /tmp/soft_pinn_smoke.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.reconstruction import benchmark as bm
from experiments.reconstruction import stage5_simulator
from experiments.reconstruction.generate_diffusion_patterns import THINNING_ENTROPY, VOLUME
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.fields import physics
from geom_mesh_net.neural import implicit

PATTERNS = (0, 3, 7)
EFFICIENCY = 0.37
HELD_OUT = 0.2
HELD_OUT_ENTROPY = 20260924
COLLOCATION = 200_000
SURFACE_DIRECTIONS = 32
EVAL_EVERY = 50
NAMES = ("c_eq", "ell", "xi", "c_inf")

DEFAULTS = dict(lam=1.0, w0=10.0, lr=1e-4, lr_constants=1e-2, freeze=0, residual="reference", width=128, depth=3)
SETTINGS = {
    "unconstrained, w0 10": dict(lam=0.0),
    "unconstrained, w0 3": dict(lam=0.0, w0=3.0),
    "scaled residual, lambda 10": dict(lam=10.0, residual="scaled"),
    "lambda 1": dict(lam=1.0),
    "lambda 0.1": dict(lam=0.1),
    "lambda 1, w0 3": dict(lam=1.0, w0=3.0),
    "lambda 1, slow constants": dict(lam=1.0, lr_constants=1e-3),
    "lambda 1, constants frozen for 500 steps": dict(lam=1.0, freeze=500),
}
COMPARE_LAMBDAS = (1.0, 0.1)


def rounded(value, digits=6):
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def outside_spheres(points, centres, radii, chunk=20000):
    keep = np.ones(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        x = points[start:start + chunk]
        gap = np.sqrt(((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)) - radii[None, :]
        keep[start:start + chunk] = gap.min(axis=1) > 0
    return keep


def analytic_reference(centres, radii, x, y, held_x, scored_x, init):
    """Maximum likelihood of the analytic family from three starting points; the best by training loss."""
    best = None
    for ell0, xi0 in ((init["ell"], init["xi"]), (1.0, 3.0), (6.0, 12.0)):
        model = implicit.ParametricDiffusionField(centres, radii, c_eq=init["c_eq"], ell=ell0, xi=xi0, c_inf=init["c_inf"])
        record = implicit.fit_parametric(model, x, y)
        if best is None or record["train_loss"] < best[1]["train_loss"]:
            best = (model, record)
    model, record = best

    def values(points):
        with torch.no_grad():
            return np.clip(np.concatenate([model(torch.as_tensor(points[s:s + 20000], dtype=torch.float64)).numpy()
                                           for s in range(0, len(points), 20000)]), 1e-9, 1 - 1e-9)

    return record["constants"], values(held_x), values(scored_x)


def fit(setting, data, init, steps, device, frozen_constants=None):
    cfg = {**DEFAULTS, **setting}
    constants = init if frozen_constants is None else {**init, **frozen_constants}
    torch.manual_seed(0)
    model = implicit.DiffusionPINN(c0=init["c0"], c_eq=constants["c_eq"], ell=constants["ell"], xi=constants["xi"],
                                   c_inf=constants["c_inf"], xi_ref=init["xi"], residual=cfg["residual"], w0=cfg["w0"],
                                   width=cfg["width"], depth=cfg["depth"])
    record = implicit.fit_pinn(model, data["train_x"], data["train_y"], data["held_x"], data["held_y"],
                               collocation=data["collocation"], boundary=data["boundary"], lambda_pde=cfg["lam"],
                               lambda_bc=cfg["lam"], scale=init["c0"], steps=steps, lr=cfg["lr"],
                               lr_constants=cfg["lr_constants"], eval_every=EVAL_EVERY, device=device,
                               freeze_constants=steps if frozen_constants is not None else cfg["freeze"])
    q = implicit.predict(model, data["scored_x"], device=device)
    tail = record["history"][-5:]
    return {
        "config": cfg,
        "best_step": record["best_step"],
        "held_out_loss": record["valid_loss"],
        "matrix_excess": fb.log_loss(data["scored_y"], q) - data["scored_oracle_loss"],
        "constants_best": record["constants"],
        "constants_last": {k: record["history"][-1][k] for k in NAMES},
        "last": {k: float(np.mean([h[k] for h in tail])) for k in ("train_bce", "pde", "bc")},
        "objective_last": float(np.mean([h["train_bce"] + cfg["lam"] * (h["pde"] + h["bc"]) for h in tail])),
        "seconds": record["seconds"],
        "history": record["history"],
    }


def prepare(index):
    with np.load(paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        coords, labels = d["coords"].item(), d["labels"]
        truth = d["physics"].item()
    x = np.column_stack([coords[a] for a in "xyz"])
    guest = np.isin(labels, (2, 3)).astype(float)
    field = physics.DiffusionField.from_dict(truth)
    p_star, inside = stage5_simulator.load_oracle(index)
    mask = bm.thinning_mask(index, EFFICIENCY, len(x), entropy=THINNING_ENTROPY)
    observed = np.flatnonzero(mask & ~inside)
    generator = np.random.default_rng(np.random.SeedSequence([HELD_OUT_ENTROPY, index]))
    held = generator.random(len(observed)) < HELD_OUT
    train, held_out = observed[~held], observed[held]
    candidates = generator.uniform(0, 60, size=(3 * COLLOCATION, 3))
    collocation = candidates[outside_spheres(candidates, field.centres, field.radii)][:COLLOCATION]
    scored = np.flatnonzero(~mask & ~inside)
    mean = float(guest[observed].mean())
    init = {"c0": mean, "c_eq": mean / 2, "ell": float(np.mean(field.radii)),
            "xi": physics.screening_length(field.radii, VOLUME), "c_inf": mean}
    data = {
        "train_x": x[train], "train_y": guest[train], "held_x": x[held_out], "held_y": guest[held_out],
        "collocation": collocation, "scored_x": x[scored], "scored_y": guest[scored],
        "scored_oracle_loss": fb.log_loss(guest[scored], p_star[scored]),
        "boundary": (field.centres, field.radii, implicit.fibonacci_directions(SURFACE_DIRECTIONS)),
    }
    train_mean = float(guest[train].mean())
    summary = {
        "pattern": index, "precipitates": int(len(field.radii)), "truth": {k: getattr(field, k) for k in NAMES},
        "init": init, "atoms": {"train": int(len(train)), "held_out": int(len(held_out)), "scored": int(len(scored))},
        "data_margin_per_atom": fb.log_loss(guest[train], np.full(len(train), train_mean)) - fb.log_loss(guest[train], field(x[train])),
        "held_out_loss": {"oracle": fb.log_loss(guest[held_out], field(x[held_out])),
                          "constant": fb.log_loss(guest[held_out], np.full(len(held_out), train_mean))},
        "constant_matrix_excess": fb.log_loss(guest[scored], np.full(len(scored), train_mean)) - data["scored_oracle_loss"],
    }
    return field, data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", default=",".join(map(str, PATTERNS)))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--compare-steps", type=int, default=3000)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "soft_pinn.json")
    args = parser.parse_args()
    patterns = [int(i) for i in args.patterns.split(",")]
    if any(i >= 50 for i in patterns):
        raise SystemExit("this pilot uses development patterns only (0-49)")

    report = {
        "design": {"patterns": patterns, "efficiency": EFFICIENCY, "held_out": HELD_OUT, "steps": args.steps,
                   "compare_steps": args.compare_steps, "collocation_points": COLLOCATION,
                   "surface_directions": SURFACE_DIRECTIONS, "eval_every": EVAL_EVERY, "defaults": DEFAULTS,
                   "settings": SETTINGS, "compare_lambdas": COMPARE_LAMBDAS,
                   "entropy": {"thinning": THINNING_ENTROPY, "held_out": HELD_OUT_ENTROPY},
                   "geometry": "true centres and radii", "matrix": "observed atoms outside every true sphere"},
        "environment": {"torch": torch.__version__, "device": args.device},
        "patterns": [],
    }
    started = time.perf_counter()
    for index in patterns:
        field, data, summary = prepare(index)
        print(f"pattern {index}: {summary['precipitates']} precipitates, data margin {summary['data_margin_per_atom']:.5f} nats "
              f"per atom, constant matrix excess {summary['constant_matrix_excess']:.5f}", flush=True)
        report["patterns"].append(summary)
        constants, held_q, scored_q = analytic_reference(field.centres, field.radii, data["train_x"], data["train_y"],
                                                         data["held_x"], data["scored_x"], summary["init"])
        summary["analytic_family"] = {"constants": constants, "held_out_loss": fb.log_loss(data["held_y"], held_q),
                                      "matrix_excess": fb.log_loss(data["scored_y"], scored_q) - data["scored_oracle_loss"]}
        print(f"  analytic family: excess {summary['analytic_family']['matrix_excess']:.5f}",
              {k: round(v, 3) for k, v in constants.items()}, flush=True)
        summary["settings"] = {}
        for name, setting in SETTINGS.items():
            result = fit(setting, data, summary["init"], args.steps, args.device)
            summary["settings"][name] = result
            print(f"  {name}: excess {result['matrix_excess']:.5f}, best step {result['best_step']}, "
                  f"last constants", {k: round(v, 3) for k, v in result["constants_last"].items()},
                  f"({result['seconds']} s)", flush=True)
            args.output.write_text(json.dumps(rounded(report), indent=1) + "\n")
        flat = {"c_eq": summary["init"]["c0"], "ell": 1e-3, "xi": field.xi, "c_inf": summary["init"]["c0"]}
        summary["fixed_constants"] = {}
        for lam in COMPARE_LAMBDAS:
            summary["fixed_constants"][str(lam)] = {}
            for label, constants in (("true", {k: getattr(field, k) for k in NAMES}), ("flat", flat)):
                result = fit({"lam": lam}, data, summary["init"], args.compare_steps, args.device, frozen_constants=constants)
                summary["fixed_constants"][str(lam)][label] = result
                print(f"  fixed {label} constants, lambda {lam}: objective {result['objective_last']:.5f} = training loss "
                      f"{result['last']['train_bce']:.5f} + lambda x ({result['last']['pde']:.5f} + {result['last']['bc']:.5f})",
                      flush=True)
                args.output.write_text(json.dumps(rounded(report), indent=1) + "\n")
    report["seconds"] = round(time.perf_counter() - started)
    args.output.write_text(json.dumps(rounded(report), indent=1) + "\n")
    print(f"wrote {args.output} in {report['seconds']} s")


if __name__ == "__main__":
    main()
