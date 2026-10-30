"""Stage 2: a Fourier-feature field fitted to one pattern's observed atoms; decides Gate 2.

The question (``experiments/reconstruction/ROADMAP.md``, Stage 2): fitted only to one pattern's
observed atoms, does a coordinate network with Fourier features estimate p* better than the
tuned B1 and B2 smoothers? Two controls separate what the encoding does from what the network
does: an MLP of the same width and depth on raw coordinates, and a linear logit on the same
Fourier features. Everything the experiment fixes is in ``stage2_design.json``; this module
reads it and refuses test cells until its status is ``frozen``.

For each (pattern, efficiency) *cell*:

1. The frozen dataset and thinning mask are checked against their recorded checksums.
2. Observed atoms are split 80 / 20 into fitting and validation atoms by a dedicated random
   stream, before anything reads a label. The split is hashed.
3. Each neural model is fitted to the fitting atoms. σ and the stopping checkpoint are chosen
   by validation cross-entropy alone; update 0, the constant field, is always a candidate.
4. The frozen refit and seed policies turn the selected fits into one prediction procedure.
5. B0, B1 and B2 are fitted to all observed atoms by Stage 1's cross-validation, B2 on the
   grid widened by O12.
6. Only then are removed atoms scored, by Stage 1's metrics, against the Stage 0 oracle.

``prepare_cell`` returns the atoms a model may see and, separately, the atoms it may not. No
fitting routine in this module accepts the second.

Usage
-----
    python -m experiments.reconstruction.stage2_field --split development-confirmation
    python -m experiments.reconstruction.stage2_field --split test
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.reconstruction import benchmark as bm
from experiments.reconstruction import stage1_baselines as s1
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.neural import implicit

DESIGN_PATH = paths.RECONSTRUCTION_DIR / "stage2_design.json"
FROZEN_DIR = bm.BENCHMARK_DIR
RESULTS = paths.RECONSTRUCTION_DIR / "results"


def load_design(path=DESIGN_PATH):
    return json.loads(Path(path).read_text())


def design_hash(design):
    """Sixteen hex digits identifying a design; every stored cell carries the hash it ran under."""
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:16]


def per_mille(eta):
    value = int(round(eta * 1000))
    if not np.isclose(value / 1000, eta):
        raise ValueError(f"efficiency {eta} is not a whole number of per mille")
    return value


def stream_seed(design, index, eta, stream, *extra):
    """A seed for one named random stream of one cell, independent of every other stream."""
    words = [design["split"]["entropy"], int(index), per_mille(eta), design["split"]["streams"][stream]]
    return int(np.random.SeedSequence(words + [int(e) for e in extra]).generate_state(1)[0])


@dataclass
class FittingData:
    """Everything a model may see: observed atoms only, already split."""

    index: int
    eta: float
    fit_x: np.ndarray
    fit_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    observed_x: np.ndarray
    observed_y: np.ndarray
    fit_rate: float
    observed_rate: float
    split_hash: str


@dataclass
class ScoringData:
    """What no model may see: removed labels, the oracle, and the regions derived from the truth."""

    index: int
    eta: float
    all_x: np.ndarray
    all_y: np.ndarray
    removed: np.ndarray
    p_star: np.ndarray
    region: np.ndarray
    domain: dict
    pattern: dict


_FROZEN = {}


def _frozen(name):
    if name not in _FROZEN:
        _FROZEN[name] = json.loads((FROZEN_DIR / name).read_text())
    return _FROZEN[name]


def file_checksum(index):
    return hashlib.sha256((bm.DATA_DIR / f"clust_pattern_{index}.npz").read_bytes()).hexdigest()


def prepare_cell(index, eta, design, verify=True):
    """Load one cell, verify it against the frozen records, and split its observed atoms.

    Returns ``(FittingData, ScoringData, checks)``. The checks record the checksums compared,
    the split's hash, and the leakage conditions, each of which raises if it fails: fitting and
    validation atoms are disjoint and together are exactly the observed atoms, no observed atom
    is removed, and the constant the models start from uses fitting labels only.
    """
    pattern = bm.load_pattern(index)
    x, y = pattern["coords"].astype(np.float64), pattern["guest"].astype(np.float64)
    mask = bm.thinning_mask(index, eta, len(x))
    checks = {}
    if verify:
        recorded = _frozen("dataset_checksums.json")[str(index)]
        if file_checksum(index) != recorded:
            raise RuntimeError(f"pattern {index}: dataset differs from the frozen checksum")
        if bm.mask_checksum(mask) != _frozen("mask_checksums.json")[f"{index}:{eta}"]:
            raise RuntimeError(f"pattern {index} at {eta}: thinning mask differs from the frozen checksum")
        checks["dataset_checksum"] = recorded
        checks["mask_checksum"] = _frozen("mask_checksums.json")[f"{index}:{eta}"]

    observed, removed = np.flatnonzero(mask), np.flatnonzero(~mask)
    draw = np.random.default_rng(stream_seed(design, index, eta, "split")).random(len(observed))
    in_validation = draw < design["split"]["validation_fraction"]
    fit, validation = observed[~in_validation], observed[in_validation]
    split_hash = hashlib.sha256(np.packbits(in_validation).tobytes()).hexdigest()

    if np.intersect1d(fit, validation).size:
        raise RuntimeError("fitting and validation atoms overlap")
    if not np.array_equal(np.sort(np.concatenate([fit, validation])), observed):
        raise RuntimeError("fitting and validation atoms are not exactly the observed atoms")
    if np.intersect1d(observed, removed).size or len(observed) + len(removed) != len(x):
        raise RuntimeError("observed and removed atoms are not a partition")
    fit_rate = float(y[fit].mean())
    checks.update(split_hash=split_hash, n_fit=int(len(fit)), n_validation=int(len(validation)),
                  n_removed=int(len(removed)), leakage="passed")

    fitting = FittingData(index, eta, x[fit], y[fit], x[validation], y[validation], x[observed], y[observed],
                          fit_rate, float(y[observed].mean()), split_hash)
    p_star, n_spheres = bm.load_oracle(index)
    region = s1.assign_regions(x, pattern["centres"], pattern["radii"], n_spheres > 0)
    scoring = ScoringData(index, eta, x, y, removed, p_star, region, pattern["domain"], pattern)
    return fitting, scoring, checks


# ---------------------------------------------------------------------------------------------
# Fitting: every routine here takes FittingData and nothing derived from removed atoms
# ---------------------------------------------------------------------------------------------

def training_options(design, learning_rate=None, **overrides):
    t = design["training"]
    options = dict(learning_rate=t["learning_rate"] if learning_rate is None else learning_rate,
                   batch_size=t["batch_size"], max_updates=t["max_updates"], eval_every=t["eval_every"],
                   patience=t["patience"])
    options.update(overrides)
    return options


def fit_fourier_candidates(data, design, seed, device, hidden=None, sigmas=None, learning_rate=None,
                           model_code=0, **overrides):
    """One fit per σ, all from the same B0, initial weights and batch order; returns them all.

    ``model_code`` separates the initialisation streams of models that share a seed (0 for the
    Fourier MLP, 1 for the linear control), so neither's weights depend on the other's.
    """
    hidden = tuple(design["architecture"]["hidden"] if hidden is None else hidden)
    sigmas = design["sigma_grid"] if sigmas is None else sigmas
    base = implicit.base_frequencies(design["architecture"]["fourier_features"],
                                     stream_seed(design, data.index, data.eta, "frequencies", seed))
    init_seed = stream_seed(design, data.index, data.eta, "initialisation", seed, model_code)
    batch_seed = stream_seed(design, data.index, data.eta, "minibatch", seed)
    candidates = []
    for sigma in sigmas:
        model = implicit.build_field("fourier", init_seed, base=base, sigma=sigma, hidden=hidden)
        model.initialise_constant(data.fit_rate)
        record = implicit.fit_field(model, data.fit_x, data.fit_y, data.validation_x, data.validation_y,
                                    batch_seed=batch_seed, device=device,
                                    **training_options(design, learning_rate, **overrides))
        record.update(sigma=float(sigma), seed=int(seed), hidden=list(hidden))
        candidates.append((model, record))
    return candidates


def fit_raw(data, design, seed, device, hidden=None, learning_rate=None, **overrides):
    hidden = tuple(design["architecture"]["hidden"] if hidden is None else hidden)
    model = implicit.build_field("raw", stream_seed(design, data.index, data.eta, "initialisation", seed, 2),
                                 hidden=hidden)
    model.initialise_constant(data.fit_rate)
    record = implicit.fit_field(model, data.fit_x, data.fit_y, data.validation_x, data.validation_y,
                                batch_seed=stream_seed(design, data.index, data.eta, "minibatch", seed),
                                device=device, **training_options(design, learning_rate, **overrides))
    record.update(sigma=None, seed=int(seed), hidden=list(hidden))
    return model, record


def select(candidates):
    """The candidate with the lowest validation cross-entropy; ties go to the smaller σ."""
    order = sorted(range(len(candidates)),
                   key=lambda i: (candidates[i][1]["best_validation_bce"], candidates[i][1]["sigma"] or 0.0))
    return order[0]


def refit_updates(record, n_all, batch_size, rule):
    """How long a restart on all observed atoms trains, for the refit rules of the design.

    ``"updates"`` repeats the selected number of optimiser steps. ``"passes"`` repeats the
    selected number of passes through the data: identical to ``"updates"`` when both fits are
    full-batch, and 1.25 times as many steps when both use fixed-size minibatches.
    """
    if rule == "updates":
        return int(record["best_update"])
    if rule == "passes":
        # the product is often a whole number that floating point overshoots by 1e-13: round first
        return int(np.ceil(round(record["best_passes"] * n_all / min(batch_size, n_all), 6)))
    raise ValueError(f"unknown refit rule {rule!r}")


def refit(data, design, selected, rule, device, kind="fourier", model_code=0, learning_rate=None):
    """Restart the selected configuration on all observed atoms for a length set by ``rule``.

    The restart uses the same B0, σ, initial weights and batch stream as the selected fit, and
    starts from the constant of all observed atoms. It has no validation data, so it trains for
    exactly the converted number of updates.
    """
    model, record = selected
    seed = record["seed"]
    batch = design["training"]["batch_size"]
    updates = refit_updates(record, len(data.observed_x), batch, rule)
    init_seed = stream_seed(design, data.index, data.eta, "initialisation", seed, model_code if kind == "fourier" else 2)
    if kind == "fourier":
        fresh = implicit.build_field("fourier", init_seed, base=model.base_frequencies.cpu().numpy(),
                                     sigma=float(model.sigma), hidden=tuple(record["hidden"]))
    else:
        fresh = implicit.build_field("raw", init_seed, hidden=tuple(record["hidden"]))
    fresh.initialise_constant(data.observed_rate)
    result = implicit.fit_field(fresh, data.observed_x, data.observed_y, None, None,
                                batch_seed=stream_seed(design, data.index, data.eta, "minibatch", seed),
                                device=device, **training_options(design, learning_rate, max_updates=updates))
    result.update(sigma=record["sigma"], seed=int(seed), hidden=record["hidden"], rule=rule)
    return fresh, result


def baselines(data, index, eta, grid_k=None, grid_c=None):
    """B0, B1 and B2 by Stage 1's procedure: 5-fold cross-validation on all observed atoms."""
    fit = fb.fit_baselines(data.observed_x, data.observed_y.astype(bool), folds=5, seed=s1.cell_seed(index, eta, 0),
                           adaptive_k=fb.ADAPTIVE_K_WIDE if grid_k is None else grid_k,
                           adaptive_c=fb.ADAPTIVE_C_WIDE if grid_c is None else grid_c)
    return fit


# ---------------------------------------------------------------------------------------------
# Scoring: runs after every selection is complete
# ---------------------------------------------------------------------------------------------

def score(predictions, scoring):
    """Stage 1's metrics for every method, on removed atoms, against the oracle.

    ``predictions`` maps a method name to its probability at *every* atom of the pattern (the
    predictive check relabels all of them); removed atoms are selected here.
    """
    removed = scoring.removed
    y = scoring.all_y[removed]
    p_star = scoring.p_star[removed]
    region = scoring.region[removed]
    oracle = fb.log_loss(y, p_star)
    out = {"loss": {}, "excess": {}, "excess_by_region": {}, "brier_excess": {}, "ece": {}}
    oracle_by_region = {}
    for r_id, name in enumerate(s1.REGIONS):
        sel = region == r_id
        if sel.any():
            oracle_by_region[name] = fb.log_loss(y[sel], p_star[sel])
    for method, q_all in predictions.items():
        q = np.clip(np.asarray(q_all, dtype=np.float64)[removed], fb.EPS, 1 - fb.EPS)
        loss = fb.log_loss(y, q)
        out["loss"][method] = loss
        out["excess"][method] = loss - oracle
        out["excess_by_region"][method] = {name: fb.log_loss(y[region == s1.REGIONS.index(name)],
                                                              q[region == s1.REGIONS.index(name)]) - value
                                           for name, value in oracle_by_region.items()}
        out["brier_excess"][method] = float(np.mean((q - p_star) ** 2))
        out["ece"][method] = s1.expected_calibration_error(y, q)
    out["loss"]["oracle"] = oracle
    out["region_share"] = {name: float((region == r_id).mean()) for r_id, name in enumerate(s1.REGIONS)}
    gap = out["loss"]["B0"] - oracle
    out["gap"] = gap
    out["gap_closed"] = {m: (out["loss"]["B0"] - out["loss"][m]) / gap for m in predictions if m != "B0"}
    b1_gap = out["loss"]["B1"] - oracle
    out["share_of_b1_gap_closed"] = {m: (out["loss"]["B1"] - out["loss"][m]) / b1_gap
                                     for m in predictions if m not in ("B0", "B1")}
    out["headroom"] = bool(bm.has_headroom(out["loss"]["B0"], out["loss"]["B1"], oracle))
    return out


# ---------------------------------------------------------------------------------------------
# The gate harness
# ---------------------------------------------------------------------------------------------

MODELS = ("fourier", "linear", "raw")          # the gated method first; the two controls after it
PREDICTIONS_DIR = RESULTS / "stage2_predictions"   # gitignored: removed-atom predictions per cell
SPLIT_SCOPES = {"test": "test", "development-pilot": "development_pilot",
                "development-confirmation": "development_confirmation"}


def model_settings(design, name):
    """How one of the three models is fitted: architecture, learning rate, and its stream code."""
    hidden = design["architecture"]["hidden"]
    if name == "fourier":
        return dict(kind="fourier", hidden=hidden, learning_rate=design["training"]["learning_rate"], model_code=0)
    if name == "linear":
        return dict(kind="fourier", hidden=[], learning_rate=design["controls"]["linear_fourier"]["learning_rate"],
                    model_code=1)
    if name == "raw":
        return dict(kind="raw", hidden=hidden, learning_rate=design["controls"]["raw_mlp"]["learning_rate"], model_code=2)
    raise ValueError(name)


def summary_of(record):
    keep = ("sigma", "seed", "best_update", "best_passes", "best_validation_bce", "initial_validation_bce",
            "updates_run", "passes_run", "stopped", "nonfinite", "full_batch", "mean_residual", "seconds", "parameters")
    return {k: record.get(k) for k in keep}


def fit_model(data, design, name, device):
    """Fit one model under the frozen seed and refit policies; returns its prediction procedure.

    Each seed selects its own σ and checkpoint on validation atoms. The refit rule then either
    keeps that checkpoint or restarts on all observed atoms. With the ensemble policy the
    returned predictor averages the seeds' probabilities; with the single-seed policy it is the
    first seed's.
    """
    settings = model_settings(design, name)
    rule = design["refit"]["rule"]
    models, records = [], []
    for seed in design["seeds"]["seeds"]:
        if settings["kind"] == "raw":
            model, record = fit_raw(data, design, seed, device, hidden=settings["hidden"],
                                    learning_rate=settings["learning_rate"])
            candidates, chosen = [(model, record)], 0
        else:
            candidates = fit_fourier_candidates(data, design, seed, device, hidden=settings["hidden"],
                                                learning_rate=settings["learning_rate"],
                                                model_code=settings["model_code"])
            chosen = select(candidates)
        entry = {"selected": summary_of(candidates[chosen][1]),
                 "candidates": [summary_of(r) for _, r in candidates]}
        final = candidates[chosen][0]
        if rule != "keep":
            final, refit_record = refit(data, design, candidates[chosen], rule, device, kind=settings["kind"],
                                        model_code=settings["model_code"], learning_rate=settings["learning_rate"])
            entry["refit"] = summary_of(refit_record)
        models.append(final)
        records.append(entry)
    return models, records


def predict_procedure(models, points, device):
    """The equal-weight average of the members' probabilities (one member: its own)."""
    return np.mean([implicit.predict(m, points, device=device) for m in models], axis=0)


def run_cell(index, eta, design, device, save_predictions=True):
    started = time.perf_counter()
    theta = bm.load_theta()[index]
    data, scoring, checks = prepare_cell(index, eta, design)
    fit = baselines(data, index, eta)
    predictions = fb.predict_baselines(fit, data.observed_x, data.observed_y.astype(bool), scoring.all_x)
    cell = {"pattern": int(index), "efficiency": float(eta),
            "theta": dict(zip(("rho_c", "rho_b", "cr", "rb"), (float(v) for v in theta))),
            "cr_band": bm.band(theta[2], bm.CR_BANDS), "rho_c_band": bm.band(theta[0], bm.RHO_C_BANDS),
            "checks": checks, "design_hash": design_hash(design),
            "baselines": {"b1_bandwidth": fit.bandwidth, "b2_k": fit.adaptive_k, "b2_c": fit.adaptive_c},
            "models": {}, "flags": []}
    for name in MODELS:
        t0 = time.perf_counter()
        models, records = fit_model(data, design, name, device)
        predictions[name] = predict_procedure(models, scoring.all_x, device)
        cell["models"][name] = {"seeds": records, "seconds": round(time.perf_counter() - t0, 1),
                                "parameters": models[0].parameter_count()}
        for position, record in enumerate(records):
            for part in ("selected", "refit"):
                if record.get(part, {}).get("nonfinite"):
                    cell["flags"].append(f"{name} seed {position} {part}: training went non-finite")
    predictions["oracle"] = scoring.p_star

    # Everything above chose from observed atoms only. Removed atoms are read from here on.
    cell["score"] = score({k: v for k, v in predictions.items() if k != "oracle"}, scoring)
    cell["predictive_check"] = predictive_check(predictions, scoring, design)
    if save_predictions:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PREDICTIONS_DIR / f"cell_{index}_{per_mille(eta)}.npz", removed=scoring.removed,
                            **{k: np.asarray(v, dtype=np.float32)[scoring.removed] for k, v in predictions.items()})
    cell["seconds"] = round(time.perf_counter() - started, 1)
    return cell


def predictive_check(predictions, scoring, design):
    """Stage 1's O11: relabel every atom from each method's field and recompute the 14 features."""
    from geom_mesh_net.statistics.presets import build_config
    config = build_config(preset="stage1")
    pattern = scoring.pattern
    out = {"truth": s1.global_features(scoring.all_x, scoring.all_y.astype(bool), scoring.domain, config).tolist()}
    for position, (name, q) in enumerate(sorted(predictions.items())):
        rng = np.random.default_rng(stream_seed(design, scoring.index, scoring.eta, "relabel", position))
        relabelled = rng.random(len(scoring.all_x)) < np.asarray(q)
        out[name] = s1.global_features(scoring.all_x, relabelled, scoring.domain, config).tolist()
    return out


def expected_cells(design, split):
    scope = design["scope"][SPLIT_SCOPES[split]]
    return {(int(p), float(e)) for p in scope["patterns"] for e in design["scope"]["efficiencies"]}


def _median(values):
    return float(np.median(values)) if len(values) else float("nan")


def gate(cells, design, split):
    """Gate 2 on the recorded cells, refusing a verdict unless every expected cell is present and valid."""
    present = {(c["pattern"], c["efficiency"]) for c in cells if "error" not in c}
    missing = sorted(expected_cells(design, split) - present)
    errored = sorted((c["pattern"], c["efficiency"]) for c in cells if "error" in c)
    valid = [c for c in cells if "error" not in c]
    head = [c for c in valid if c["score"]["headroom"]]
    other = [c for c in valid if not c["score"]["headroom"]]
    loss = lambda c, m: c["score"]["loss"][m]                                                      # noqa: E731
    beats = [loss(c, "fourier") < min(loss(c, "B1"), loss(c, "B2")) for c in head]
    conditions = {
        "beats_better_baseline": {"measured": float(np.mean(beats)) if beats else float("nan"),
                                  "cells": f"{int(np.sum(beats))} of {len(beats)}", "required": 2 / 3},
        "closes_b1_gap": {"measured": _median([c["score"]["share_of_b1_gap_closed"]["fourier"] for c in head]),
                          "required": 0.20},
        "calibration": {"measured": _median([c["score"]["ece"]["fourier"] - c["score"]["ece"]["B1"] for c in head]),
                        "required_at_most": 0.005},
        "no_harm": {"measured": _median([c["score"]["excess"]["fourier"] - c["score"]["excess"]["B1"] for c in other]),
                    "required_at_most": 0.002},
    }
    conditions["beats_better_baseline"]["passed"] = bool(beats) and conditions["beats_better_baseline"]["measured"] >= 2 / 3
    conditions["closes_b1_gap"]["passed"] = bool(head) and conditions["closes_b1_gap"]["measured"] >= 0.20
    conditions["calibration"]["passed"] = bool(head) and conditions["calibration"]["measured"] <= 0.005
    conditions["no_harm"]["passed"] = bool(other) and conditions["no_harm"]["measured"] <= 0.002
    complete = not missing and not errored
    passed = complete and all(c["passed"] for c in conditions.values())
    return {"split": split, "complete": complete, "missing": missing, "errored": errored,
            "headroom_cells": len(head), "other_cells": len(other),
            "flagged_cells": sum(1 for c in valid if c.get("flags")),
            "conditions": conditions,
            "verdict": ("passed" if passed else "failed") if complete else "incomplete", "passed": passed}


def bootstrap_over_patterns(cells, statistic, reps=2000, seed=0):
    """Median of ``statistic(cells)`` with a 95% interval, resampling patterns with their efficiencies."""
    by_pattern = {}
    for c in cells:
        by_pattern.setdefault(c["pattern"], []).append(c)
    groups = list(by_pattern.values())
    if not groups:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    point = statistic(cells)
    draws = [statistic([c for g in rng.choice(len(groups), len(groups)) for c in groups[g]]) for _ in range(reps)]
    return [float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def summarise(cells):
    """Paired comparisons and strata for the report; exploratory, separate from the gate."""
    valid = [c for c in cells if "error" not in c]
    methods = ("fourier", "linear", "raw", "B2")
    median_of = lambda key, m, ref: (lambda cs: np.median([c["score"][key][m] - c["score"][key][ref] for c in cs]))  # noqa: E731
    out = {"paired_excess_vs_B1": {m: bootstrap_over_patterns(valid, median_of("excess", m, "B1")) for m in methods},
           "paired_excess_vs_B2": {m: bootstrap_over_patterns(valid, median_of("excess", m, "B2"))
                                   for m in ("fourier", "linear", "raw")},
           "encoding": bootstrap_over_patterns(valid, median_of("excess", "fourier", "raw")),
           "nonlinearity": bootstrap_over_patterns(valid, median_of("excess", "fourier", "linear")),
           "strata": {}}
    for key, bands in (("cr_band", range(3)), ("rho_c_band", range(2)), ("efficiency", (0.1, 0.37, 0.8))):
        for value in bands:
            sel = [c for c in valid if c[key] == value]
            if not sel:
                continue
            out["strata"][f"{key}={value}"] = {
                "cells": len(sel),
                "headroom": sum(c["score"]["headroom"] for c in sel),
                "median_excess": {m: _median([c["score"]["excess"][m] for c in sel])
                                  for m in ("B1", "B2", "fourier", "linear", "raw")},
                "fourier_beats_better_baseline": float(np.mean([c["score"]["loss"]["fourier"]
                                                                < min(c["score"]["loss"]["B1"], c["score"]["loss"]["B2"])
                                                                for c in sel])),
                "median_excess_by_region": {m: {r: _median([c["score"]["excess_by_region"][m][r] for c in sel
                                                            if r in c["score"]["excess_by_region"][m]])
                                                for r in s1.REGIONS} for m in ("B1", "B2", "fourier", "linear", "raw")},
            }
    return out


def worker(task):
    index, eta, device, design = task
    try:
        return run_cell(index, eta, design, device)
    except Exception as error:                     # a failed cell stays visible and blocks the verdict
        return {"pattern": int(index), "efficiency": float(eta), "error": f"{type(error).__name__}: {error}",
                "design_hash": design_hash(design)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=list(SPLIT_SCOPES))
    parser.add_argument("--limit", type=int, default=None, help="run only this many cells (a smoke test)")
    parser.add_argument("--device", default=None, help="overrides the design only for an unfrozen smoke test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--design", type=Path, default=DESIGN_PATH,
                        help="another design file, for smoke tests; an unfrozen design cannot run test cells")
    args = parser.parse_args()

    design = load_design(args.design)
    digest = design_hash(design)
    if design["status"] != "frozen" and (args.split == "test" or args.limit is None):
        raise SystemExit("stage2_design.json is not frozen: only a --limit smoke test on development cells may run")
    device = args.device if (args.device and design["status"] != "frozen") else design["device"]
    if device == "undecided":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    output = args.output or RESULTS / f"stage2_field_{args.split}.json"
    report = json.loads(output.read_text()) if output.exists() else {"cells": []}
    if report["cells"] and report.get("design_hash") != digest:
        raise SystemExit(f"{output} holds cells from design {report.get('design_hash')}, not {digest}; "
                         "version the design or move the file before running")
    report.update(split=args.split, design=design, design_hash=digest)
    done = {(c["pattern"], c["efficiency"]) for c in report["cells"] if "error" not in c}
    report["cells"] = [c for c in report["cells"] if "error" not in c]
    todo = sorted(expected_cells(design, args.split) - done)[: args.limit]
    print(f"{len(todo)} cells to run, {len(done)} already done ({args.split}, design {digest}, {device})", flush=True)
    started = time.perf_counter()
    for position, (index, eta) in enumerate(todo, 1):
        cell = worker((index, eta, device, design))
        report["cells"].append(cell)
        if "error" in cell:
            print(f"  {position}/{len(todo)} pattern {index} eta {eta}: ERROR {cell['error']}", flush=True)
        else:
            s = cell["score"]
            print(f"  {position}/{len(todo)} pattern {index} eta {eta}: excess B1 {s['excess']['B1']:.4f} "
                  f"B2 {s['excess']['B2']:.4f} fourier {s['excess']['fourier']:.4f} linear {s['excess']['linear']:.4f} "
                  f"raw {s['excess']['raw']:.4f}{' headroom' if s['headroom'] else ''}"
                  f"{' FLAGS ' + '; '.join(cell['flags']) if cell['flags'] else ''} ({cell['seconds']} s)", flush=True)
        report["seconds"] = round(report.get("seconds", 0) + (time.perf_counter() - started), 1)
        started = time.perf_counter()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report) + "\n")
    report["gate"] = gate(report["cells"], design, args.split)
    if report["gate"]["complete"]:
        report["summary"] = summarise(report["cells"])
    output.write_text(json.dumps(report) + "\n")
    print(json.dumps(report["gate"], indent=1))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
