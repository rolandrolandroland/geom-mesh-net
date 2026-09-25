"""Stage 2 development pilot: fix the Fourier-feature field's design before any test cell is fitted.

Everything here runs on **development** patterns (``stage2_design.json``, scope
``development_pilot``). Within a cell, σ and the stopping checkpoint are chosen on validation
atoms exactly as the gate harness will choose them. Across cells, the pilot also scores every
candidate on the removed atoms of those development patterns. That is what development
patterns are for: those scores decide the *global* design (the σ grid, learning rate, width,
refit rule, seed policy), which is then frozen. No removed-atom score ever chooses anything
inside a cell, here or in the harness.

Presets, each answering one question of the Stage 2 plan (steps 7 to 10):

- ``explore``: four contrasting cells, a wide σ grid and three learning rates, to set the
  pilot's own starting grid;
- ``grid``: every pilot cell, the Fourier MLP over the σ grid, plus both controls, each at
  learning rates of its own;
- ``width``: the Fourier MLP at hidden width 128, the one smaller alternative;
- ``seeds``: seeds 0, 1 and 2 of the Fourier MLP, their equal-weight ensemble, and (with
  ``--refit updates,passes``) every refit rule, for the refit and seed policies; seed 0 again
  on MPS measures the device's run-to-run variation against ``grid``;
- ``device``: seed 0 on another device (``--device cpu``) for a few cells;
- ``ablation``: the gradient penalty.

Usage
-----
    python -m experiments.reconstruction.pilot.check_fourier_field --preset explore
    python -m experiments.reconstruction.pilot.check_fourier_field --preset grid
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from experiments.reconstruction import stage2_field as s2
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.neural import implicit

RESULTS = paths.RECONSTRUCTION_DIR / "pilot" / "results"
EXPLORE_CELLS = [(4, 0.37), (4, 0.1), (35, 0.37), (1, 0.8)]


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


def trim(record):
    """A fit record without the model, with its history rounded, for the report."""
    out = {k: v for k, v in record.items() if k != "history"}
    out["history"] = [[row["update"], round(row["fit_bce"], 6), round(row.get("validation_bce", float("nan")), 6)]
                      for row in record["history"]]
    return out


def specs_for(preset, design):
    """The model configurations a preset runs; each is fitted and selected independently."""
    hidden = design["architecture"]["hidden"]
    lr = design["training"]["learning_rate"]
    grid = design["sigma_grid"]
    if preset == "explore":
        return [dict(name=f"fourier_lr{rate:g}", kind="fourier", hidden=hidden, lr=rate, sigmas=[0.75, 1.5, 3, 6, 12, 24],
                     seed=0) for rate in (1e-3, 3e-4, 1e-4)]
    if preset == "grid":
        # each control gets learning rates of its own, so an optimisation failure is not
        # mistaken for a limitation of the model
        return ([dict(name="fourier", kind="fourier", hidden=hidden, lr=lr, sigmas=grid, seed=0)]
                + [dict(name=f"linear_lr{rate:g}", kind="linear", hidden=[], lr=rate, sigmas=grid, seed=0)
                   for rate in (1e-3, 1e-2, 3e-2)]
                + [dict(name=f"raw_lr{rate:g}", kind="raw", hidden=hidden, lr=rate, sigmas=None, seed=0)
                   for rate in (1e-4, 1e-3)])
    if preset == "width":
        return [dict(name="fourier_w128", kind="fourier", hidden=[128] * len(hidden), lr=lr, sigmas=grid, seed=0)]
    if preset == "seeds":
        # seed 0 is refitted here too: against the grid preset's seed 0 it measures MPS repeatability
        return [dict(name=f"fourier_seed{s}", kind="fourier", hidden=hidden, lr=lr, sigmas=grid, seed=s) for s in (0, 1, 2)]
    if preset == "device":
        return [dict(name="fourier_repeat", kind="fourier", hidden=hidden, lr=lr, sigmas=grid, seed=0)]
    if preset == "ablation":
        # σ is held at the cell's unpenalised selection from the grid preset, so the penalty is the
        # only thing that changes; λ = 0 is refitted here too, on the same device (the CPU)
        return [dict(name=f"fourier_penalty{lam:g}", kind="fourier", hidden=hidden, lr=lr, sigmas="grid_selection",
                     seed=0, penalty=lam) for lam in (0.0, 1e-4, 1e-3, 1e-2)]
    if preset == "edge":
        # one step below the grid, for the cells whose selection sat on its lower edge
        return [dict(name="fourier_sigma0.1875", kind="fourier", hidden=hidden, lr=lr, sigmas=[0.1875], seed=0)]
    raise ValueError(f"unknown preset {preset!r}")


ENSEMBLES = {"seeds": {"ensemble": ["fourier_seed0", "fourier_seed1", "fourier_seed2"]}}


def mean_squared_gradient(model, points, device, chunk=16384):
    """mean |∇ₓ f|² of the logit over ``points``, in physical units: the scale the penalty acts on.

    Always computed on the CPU: input gradients through this network are unreliable on MPS
    (see ``implicit.fit_field``). ``device`` is where the model lives; it is returned there.
    """
    model = model.to("cpu")
    total = 0.0
    for start in range(0, len(points), chunk):
        x = torch.as_tensor(np.asarray(points[start:start + chunk]), dtype=torch.float32)
        x.requires_grad_(True)
        gradient = torch.autograd.grad(model.logits(x).sum(), x)[0]
        total += float((gradient ** 2).sum())
    model.to(device)
    return total / len(points)


def grid_selection(index, eta):
    """The σ the grid preset's unpenalised Fourier MLP selected in this cell."""
    for cell in json.loads((RESULTS / "fourier_field_grid.json").read_text())["cells"]:
        if cell["pattern"] == index and cell["efficiency"] == eta:
            entry = cell["specs"]["fourier"]
            return entry["candidates"][entry["selected"]]["sigma"]
    raise LookupError(f"the grid preset has not run pattern {index} at {eta}")


def fit_spec(spec, data, design, device):
    """Fit one configuration and select within the cell by validation BCE alone."""
    if spec.get("sigmas") == "grid_selection":
        spec = dict(spec, sigmas=[grid_selection(data.index, data.eta)])
    options = {}
    if spec.get("penalty"):
        options["gradient_penalty"] = spec["penalty"]
    if spec["kind"] == "raw":
        model, record = s2.fit_raw(data, design, spec["seed"], device, hidden=spec["hidden"], learning_rate=spec["lr"],
                                   **options)
        return [(model, record)], 0
    candidates = s2.fit_fourier_candidates(data, design, spec["seed"], device, hidden=spec["hidden"],
                                           sigmas=spec["sigmas"], learning_rate=spec["lr"],
                                           model_code=1 if spec["kind"] == "linear" else 0, **options)
    return candidates, s2.select(candidates)


def run_cell(index, eta, specs, design, device, refit_rules=(), scored_candidates=True, ensembles=None):
    started = time.perf_counter()
    data, scoring, checks = s2.prepare_cell(index, eta, design)
    t0 = time.perf_counter()
    fit = s2.baselines(data, index, eta)
    predictions = fb.predict_baselines(fit, data.observed_x, data.observed_y.astype(bool), scoring.all_x)
    baseline_seconds = time.perf_counter() - t0
    row = {"pattern": index, "efficiency": eta, "checks": checks, "baseline_seconds": round(baseline_seconds, 1),
           "b2": {"k": fit.adaptive_k, "c": fit.adaptive_c}, "b1_bandwidth": fit.bandwidth, "specs": {}}
    for spec in specs:
        t1 = time.perf_counter()
        candidates, chosen = fit_spec(spec, data, design, device)
        entry = {"spec": spec, "selected": chosen, "candidates": [trim(r) for _, r in candidates],
                 "mean_squared_gradient": mean_squared_gradient(candidates[chosen][0], data.validation_x, device)}
        for position, (model, record) in enumerate(candidates):
            if scored_candidates or position == chosen:
                predictions[f"{spec['name']}#{position}"] = implicit.predict(model, scoring.all_x, device=device)
        for rule in refit_rules:
            model, record = s2.refit(data, design, candidates[chosen], rule, device,
                                     kind="raw" if spec["kind"] == "raw" else "fourier",
                                     model_code=1 if spec["kind"] == "linear" else 0, learning_rate=spec["lr"])
            predictions[f"{spec['name']}#refit_{rule}"] = implicit.predict(model, scoring.all_x, device=device)
            entry[f"refit_{rule}"] = trim(record)
        entry["seconds"] = round(time.perf_counter() - t1, 1)
        row["specs"][spec["name"]] = entry
    for name, members in (ensembles or {}).items():
        # an ensemble is its own prediction procedure: the equal-weight average of the members'
        # selected fits, or of their refits under the same rule
        for variant in ["keep"] + [f"refit_{rule}" for rule in refit_rules]:
            keys = [f"{m}#{row['specs'][m]['selected']}" if variant == "keep" else f"{m}#{variant}" for m in members]
            predictions[f"{name}#{variant}"] = np.mean([predictions[k] for k in keys], axis=0)
    row["score"] = s2.score(predictions, scoring)
    row["seconds"] = round(time.perf_counter() - started, 1)
    row["peak_rss_gb"] = round(peak_rss_gb(), 2)
    if device == "mps":
        row["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--report", action="store_true", help="print the pilot's evidence from its result files")
    parser.add_argument("--patterns", default=None, help="comma-separated; the pilot scope by default")
    parser.add_argument("--efficiencies", default=None)
    parser.add_argument("--cells", default=None, help="explicit pattern:efficiency pairs, e.g. 4:0.37,7:0.1")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--refit", default="", help="comma-separated refit rules to apply to each selection")
    parser.add_argument("--no-candidate-scores", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.report:
        pilot_report()
        return
    if not args.preset:
        parser.error("--preset is required unless --report is given")

    design = s2.load_design()
    specs = specs_for(args.preset, design)
    if args.cells:
        cells = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cells.split(",")]
    elif args.patterns:
        patterns = [int(p) for p in args.patterns.split(",")]
        efficiencies = [float(e) for e in (args.efficiencies or ",".join(map(str, design["scope"]["efficiencies"]))).split(",")]
        cells = [(p, e) for p in patterns for e in efficiencies]
    elif args.preset == "explore":
        cells = EXPLORE_CELLS
    else:
        cells = [(p, e) for p in design["scope"]["development_pilot"]["patterns"] for e in design["scope"]["efficiencies"]]
    output = args.output or RESULTS / f"fourier_field_{args.preset}{'_' + args.device if args.preset == 'device' else ''}.json"
    report = json.loads(output.read_text()) if output.exists() else {"cells": []}
    done = {(c["pattern"], c["efficiency"]) for c in report["cells"]}
    report.update(preset=args.preset, device=args.device, design_hash=s2.design_hash(design), specs=specs,
                  refit_rules=[r for r in args.refit.split(",") if r])
    started = time.perf_counter()
    todo = [c for c in cells if c not in done]
    print(f"{len(todo)} cells to run, {len(done)} already done ({args.preset}, {args.device})", flush=True)
    for position, (index, eta) in enumerate(todo, 1):
        row = run_cell(index, eta, specs, design, args.device, report["refit_rules"], not args.no_candidate_scores,
                       ENSEMBLES.get(args.preset))
        report["cells"].append(row)
        excess = row["score"]["excess"]
        parts = [f"B1 {excess['B1']:.4f}", f"B2 {excess['B2']:.4f}"]
        for name, entry in row["specs"].items():
            chosen = entry["candidates"][entry["selected"]]
            parts.append(f"{name} {excess[name + '#' + str(entry['selected'])]:.4f} "
                         f"(sigma {chosen['sigma']}, update {chosen['best_update']})")
        print(f"{position}/{len(todo)} pattern {index} eta {eta}: " + " | ".join(parts)
              + f" [{row['seconds']} s, {row['peak_rss_gb']} GB]", flush=True)
        report["seconds"] = report.get("seconds", 0) + row["seconds"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=None) + "\n")
    print(f"wrote {output} in {time.perf_counter() - started:.0f} s")


# ---------------------------------------------------------------------------------------------
# The pilot report: every table of Steps 7 to 10, regenerated from the tracked result files
# ---------------------------------------------------------------------------------------------

def _load(name):
    path = RESULTS / f"fourier_field_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _excess(cell, key):
    return cell["score"]["excess"][key]


def _chosen(cell, spec):
    return f"{spec}#{cell['specs'][spec]['selected']}"


def pilot_report():
    """Print the pilot's evidence for every design decision, from whichever presets have run."""
    from collections import Counter

    grid_report = _load("grid")
    grid = {(c["pattern"], c["efficiency"]): c for c in (grid_report or {"cells": []})["cells"]}
    if grid_report:
        cells = grid_report["cells"]
        specs = list(cells[0]["specs"])
        print(f"== grid: {len(cells)} cells, design {grid_report['design_hash']} ==")
        for eta in sorted({c["efficiency"] for c in cells}):
            sel = [c for c in cells if c["efficiency"] == eta]
            parts = [f"B1 {np.median([_excess(c, 'B1') for c in sel]):.4f}", f"B2 {np.median([_excess(c, 'B2') for c in sel]):.4f}"]
            for s in specs:
                wins = sum(_excess(c, _chosen(c, s)) < min(_excess(c, "B1"), _excess(c, "B2")) for c in sel)
                parts.append(f"{s} {np.median([_excess(c, _chosen(c, s)) for c in sel]):.4f} (beats both in {wins})")
            print(f"  eta {eta}, {len(sel)} cells, median excess: " + " | ".join(parts))
        head = [c for c in cells if c["score"]["headroom"]]
        other = [c for c in cells if not c["score"]["headroom"]]
        print(f"  headroom cells {len(head)}; the gate's statistics, per configuration:")
        for s in specs:
            wins = sum(_excess(c, _chosen(c, s)) < min(_excess(c, "B1"), _excess(c, "B2")) for c in head)
            share = np.median([c["score"]["share_of_b1_gap_closed"][_chosen(c, s)] for c in head])
            ece = np.median([c["score"]["ece"][_chosen(c, s)] - c["score"]["ece"]["B1"] for c in head])
            harm = np.median([_excess(c, _chosen(c, s)) - _excess(c, "B1") for c in other]) if other else float("nan")
            print(f"    {s:16s} beats the better baseline {wins}/{len(head)}; B1 gap closed {share:+.3f}; "
                  f"ECE - ECE_B1 {ece:+.4f}; elsewhere excess - B1 {harm:+.4f}")
        print("  selection and stopping:")
        for s in specs:
            chosen = [c["specs"][s]["candidates"][c["specs"][s]["selected"]] for c in cells]
            sigmas = cells[0]["specs"][s]["spec"]["sigmas"]
            edges = sum(1 for ch in chosen if isinstance(sigmas, list) and ch["sigma"] in (min(sigmas), max(sigmas)))
            caps = sum(1 for c in cells for ch in c["specs"][s]["candidates"] if ch["stopped"] == "cap")
            total = sum(len(c["specs"][s]["candidates"]) for c in cells)
            regret = [_excess(c, _chosen(c, s)) - min(_excess(c, f"{s}#{i}") for i in range(len(c["specs"][s]["candidates"])))
                      for c in cells]
            print(f"    {s:16s} sigma {dict(Counter(ch['sigma'] for ch in chosen))}; on a grid edge {edges}; "
                  f"update 0 kept {sum(ch['best_update'] == 0 for ch in chosen)}; cap {caps}/{total} candidates; "
                  f"median best update {np.median([ch['best_update'] for ch in chosen]):.0f}; selection regret median "
                  f"{np.median(regret):.5f}, max {max(regret):.4f}; {np.median([c['specs'][s]['seconds'] for c in cells]):.0f} s/cell")
        print(f"  cost: median {np.median([c['seconds'] for c in cells]):.0f} s per cell for all configurations; "
              f"peak RSS {max(c['peak_rss_gb'] for c in cells)} GB; MPS driver {max(c.get('mps_driver_gb', 0) for c in cells)} GB")

    device = _load("device_cpu")
    if device and grid:
        print("== device: seed 0 on the CPU against the grid's seed 0 on MPS ==")
        for c in device["cells"]:
            g = grid.get((c["pattern"], c["efficiency"]))
            if not g:
                continue
            ce, ge = c["specs"]["fourier_repeat"], g["specs"]["fourier"]
            cs, gs = ce["candidates"][ce["selected"]], ge["candidates"][ge["selected"]]
            print(f"  {(c['pattern'], c['efficiency'])}: sigma {cs['sigma']} / {gs['sigma']}, update {cs['best_update']} / "
                  f"{gs['best_update']}, excess {_excess(c, _chosen(c, 'fourier_repeat')):.4f} / {_excess(g, _chosen(g, 'fourier')):.4f}")

    edge = _load("edge")
    if edge and grid:
        print("== sigma 0.1875, one step below the grid ==")
        for c in edge["cells"]:
            g = grid.get((c["pattern"], c["efficiency"]))
            if not g:
                continue
            chosen = g["specs"]["fourier"]["candidates"][g["specs"]["fourier"]["selected"]]
            low = c["specs"]["fourier_sigma0.1875"]["candidates"][0]
            gain = chosen["best_validation_bce"] - low["best_validation_bce"]
            print(f"  {(c['pattern'], c['efficiency'])}: grid chose {chosen['sigma']}; 0.1875 would "
                  f"{'be selected, validation gain ' + format(gain, '.5f') if gain > 0 else 'not be selected'}; excess "
                  f"{_excess(g, _chosen(g, 'fourier')):.4f} -> {_excess(c, 'fourier_sigma0.1875#0'):.4f}")

    seeds = _load("seeds")
    if seeds:
        cells = seeds["cells"]
        names = [f"fourier_seed{s}" for s in (0, 1, 2)]
        print(f"== seeds and refit: {len(cells)} cells ==")
        rows = []
        for c in cells:
            single = np.array([_excess(c, _chosen(c, s)) for s in names])
            rows.append({"cell": (c["pattern"], c["efficiency"]), "single": single,
                         "sigmas": [c["specs"][s]["candidates"][c["specs"][s]["selected"]]["sigma"] for s in names],
                         "updates": np.array([_excess(c, f"{s}#refit_updates") for s in names]),
                         "passes": np.array([_excess(c, f"{s}#refit_passes") for s in names]),
                         "ensemble": {v: _excess(c, f"ensemble#{v}") for v in ("keep", "refit_updates", "refit_passes")}})
            g = grid.get((c["pattern"], c["efficiency"]))
            if g:
                gs = g["specs"]["fourier"]["candidates"][g["specs"]["fourier"]["selected"]]
                ss = c["specs"]["fourier_seed0"]["candidates"][c["specs"]["fourier_seed0"]["selected"]]
                rows[-1]["repeat"] = (gs["sigma"] == ss["sigma"], gs["best_update"] == ss["best_update"],
                                      abs(_excess(g, _chosen(g, "fourier")) - single[0]))
        repeats = [r["repeat"] for r in rows if "repeat" in r]
        if repeats:
            print(f"  MPS repeat of seed 0: same sigma {sum(r[0] for r in repeats)}/{len(repeats)}, same update "
                  f"{sum(r[1] for r in repeats)}/{len(repeats)}, largest excess difference {max(r[2] for r in repeats):.5f}")
        for label, subset in (("all", rows), ("eta 0.1", [r for r in rows if r["cell"][1] == 0.1]),
                              ("eta 0.37", [r for r in rows if r["cell"][1] == 0.37])):
            if not subset:
                continue
            keep = np.array([r["single"] for r in subset])
            upd = np.array([r["updates"] for r in subset])
            pas = np.array([r["passes"] for r in subset])
            ens = {v: np.array([r["ensemble"][v] for r in subset]) for v in ("keep", "refit_updates", "refit_passes")}
            print(f"  {label}, {len(subset)} cells, all seeds: refit by updates {np.median(upd - keep):+.5f} (better in "
                  f"{int((upd < keep).sum())}/{keep.size}); by passes {np.median(pas - keep):+.5f} (better in "
                  f"{int((pas < keep).sum())}/{keep.size})")
            print(f"    seed spread (max - min excess) median {np.median(keep.max(1) - keep.min(1)):.5f}; seeds disagree "
                  f"on sigma in {sum(len(set(r['sigmas'])) > 1 for r in subset)}/{len(subset)} cells; ensemble vs seed 0 "
                  f"{np.median(ens['keep'] - keep[:, 0]):+.5f} (better in {int((ens['keep'] < keep[:, 0]).sum())}/{len(subset)}); "
                  f"ensemble refit by updates {np.median(ens['refit_updates'] - keep[:, 0]):+.5f}, by passes "
                  f"{np.median(ens['refit_passes'] - keep[:, 0]):+.5f}")

    width = _load("width")
    if width and grid:
        cells = [c for c in width["cells"] if (c["pattern"], c["efficiency"]) in grid]
        print(f"== width 128 against 256: {len(cells)} cells ==")
        for eta in sorted({c["efficiency"] for c in cells}):
            sel = [c for c in cells if c["efficiency"] == eta]
            d = np.array([_excess(c, _chosen(c, "fourier_w128")) - _excess(grid[(c["pattern"], eta)],
                                                                           _chosen(grid[(c["pattern"], eta)], "fourier"))
                          for c in sel])
            print(f"  eta {eta}: 128 minus 256 median {np.median(d):+.5f}, 128 better in {int((d < 0).sum())}/{len(d)}")
        print(f"  seconds per cell, median: 128 {np.median([c['specs']['fourier_w128']['seconds'] for c in cells]):.0f}, "
              f"256 {np.median([grid[(c['pattern'], c['efficiency'])]['specs']['fourier']['seconds'] for c in cells]):.0f}")

    ablation = _load("ablation")
    if ablation:
        cells = ablation["cells"]
        print(f"== gradient penalty (CPU, sigma fixed at the grid selection): {len(cells)} cells ==")
        for region in ("overall", "rim", "core", "matrix", "interior"):
            parts = []
            for lam in ("0.0001", "0.001", "0.01"):
                key, zero = f"fourier_penalty{lam}#0", "fourier_penalty0#0"
                if region == "overall":
                    d = [_excess(c, key) - _excess(c, zero) for c in cells]
                else:
                    d = [c["score"]["excess_by_region"][key][region] - c["score"]["excess_by_region"][zero][region]
                         for c in cells if region in c["score"]["excess_by_region"][key]]
                parts.append(f"lambda {lam}: {np.median(d):+.5f} (worse in {sum(x > 0 for x in d)}/{len(d)})")
            print(f"  {region:8s} change from lambda 0, median: " + "; ".join(parts))
        grads = [c["specs"]["fourier_penalty0"]["mean_squared_gradient"] for c in cells]
        print(f"  mean squared logit gradient at lambda 0 (CPU): median {np.median(grads):.3g}, "
              f"range {min(grads):.3g} to {max(grads):.3g} per unit length squared")


if __name__ == "__main__":
    main()
