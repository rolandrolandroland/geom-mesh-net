"""Instrumented, unweighted Fourier training with auditable checkpoint histories.

The architecture, batches, objective, initialization and checkpoint selection are
the existing Stage 2/E19 procedure. This separate implementation adds artifact
collection without changing the historical experiment modules. Equivalence tests
compare its selected weights and loss history against ``implicit.fit_field``.
Only observed fitting/validation data enter training. Truth and held-back labels
belong to scoring performed after this function returns, never to selection.
"""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import implicit


def _snapshot(model):
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _observation_hashes(data):
    return {name: _array_hash(getattr(data, name))
            for name in ("fit_x", "fit_y", "validation_x", "validation_y", "observed_x", "observed_y")}


def _safe_json(value):
    """Preserve failed-run flags while using standards-compliant JSON."""
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_checkpoint(path, model, *, update, role, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    payload = {"state_dict": _snapshot(model), "hidden": list(model.hidden),
               "kind": "fourier" if hasattr(model, "base_frequencies") else "raw",
               "box": float(model.box.detach().cpu()), "update": int(update),
               "role": role, "metadata": _safe_json(metadata)}
    torch.save(payload, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "update": int(update), "role": role}


def load_checkpoint(path, *, device="cpu", expected_sha256=None):
    """Restore only tensors/primitive metadata; optionally verify the recorded bytes."""
    path = Path(path)
    if expected_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    state = bundle["state_dict"]
    kwargs = ({"base": state["base_frequencies"].numpy(), "sigma": float(state["sigma"])}
              if bundle["kind"] == "fourier" else {})
    model = implicit.build_field(bundle["kind"], 0, hidden=bundle["hidden"], box=bundle["box"], **kwargs)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, {key: value for key, value in bundle.items() if key != "state_dict"}


def _training_arrays(x, y, name):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} coordinates must be a nonempty finite (N, 3) array")
    if y.shape != (len(x),) or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError(f"{name} targets must be finite values in [0, 1], one per coordinate")
    return x, y


def fit_field_with_artifacts(model, fit_x, fit_y, val_x=None, val_y=None, *, artifact_dir,
                             metadata=None, periodic_every=None, learning_rate=1e-3,
                             batch_size=32768, max_updates=2000, eval_every=25,
                             patience=200, batch_seed=0, device="cpu", chunk=65536):
    """The existing unweighted Adam fit, with selected and terminal checkpoints.

    Optional periodic snapshots cannot select or stop training. Validation chooses
    checkpoints exactly as in the historical implementation; without validation,
    all ``max_updates`` are run and the final model is returned. No periodic truth
    diagnostic is evaluated here. A failed run retains its flag and artifacts.
    """
    fit_x, fit_y = _training_arrays(fit_x, fit_y, "fitting")
    if (val_x is None) != (val_y is None):
        raise ValueError("validation coordinates and targets must be supplied together")
    validating = val_x is not None
    if validating:
        val_x, val_y = _training_arrays(val_x, val_y, "validation")
    for value, name, minimum in ((batch_size, "batch_size", 1), (max_updates, "max_updates", 0),
                                 (eval_every, "eval_every", 1), (patience, "patience", 1), (chunk, "chunk", 1)):
        if not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if periodic_every is not None and (not np.isscalar(periodic_every) or not np.isfinite(periodic_every)
                                      or int(periodic_every) != periodic_every or periodic_every < 1):
        raise ValueError("periodic_every must be a positive integer or None")
    folder = Path(artifact_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError(f"training artifact directory is not empty: {folder}")
    metadata = dict(metadata or {})
    json.dumps(metadata, allow_nan=False)
    started = time.perf_counter()
    model.to(device)
    fx, fy = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (fit_x, fit_y))
    if validating:
        vx, vy = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (val_x, val_y))
    stream = implicit._BatchStream(len(fx), batch_size, batch_seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def evaluate(update):
        row = {"update": update, "passes": update * stream.size / stream.n,
               "fit_bce": implicit._bce(model, fx, fy, chunk)}
        if validating:
            row["validation_bce"] = implicit._bce(model, vx, vy, chunk)
        return row

    model.eval()
    history = [evaluate(0)]
    best = {"update": 0, "validation_bce": history[0].get("validation_bce"), "state": _snapshot(model)}
    periodic = []
    if periodic_every is not None:
        periodic.append(_save_checkpoint(folder / "update_000000.pt", model, update=0,
                                         role="periodic", metadata=metadata))
    stopped, nonfinite, update = ("cap" if validating else "fixed"), False, 0
    model.train()
    while update < max_updates:
        index = stream.next()
        xb, yb = (fx, fy) if index is None else (fx[index.to(device)], fy[index.to(device)])
        loss = F.binary_cross_entropy_with_logits(model.logits(xb), yb)
        if not torch.isfinite(loss):
            nonfinite, stopped = True, "nonfinite"
            break
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        update += 1
        if periodic_every is not None and update % periodic_every == 0:
            periodic.append(_save_checkpoint(folder / f"update_{update:06d}.pt", model, update=update,
                                             role="periodic", metadata=metadata))
        if validating and (update % eval_every == 0 or update == max_updates):
            model.eval()
            row = evaluate(update)
            model.train()
            history.append(row)
            if not math.isfinite(row["validation_bce"]):
                nonfinite, stopped = True, "nonfinite"
                break
            if row["validation_bce"] < best["validation_bce"]:
                best = {"update": update, "validation_bce": row["validation_bce"], "state": _snapshot(model)}
            elif update - best["update"] >= patience:
                stopped = "patience"
                break
        elif not validating and update % eval_every == 0:
            # Extra refit diagnostics do not alter optimizer state or update budget.
            model.eval()
            history.append(evaluate(update))
            model.train()
    model.eval()
    terminal = _save_checkpoint(folder / "terminal.pt", model, update=update, role="terminal", metadata=metadata)
    if validating:
        model.load_state_dict(best["state"])
    else:
        if history[-1]["update"] != update:
            history.append(evaluate(update))
        best = {"update": update, "validation_bce": None}
        if not math.isfinite(history[-1]["fit_bce"]):
            nonfinite, stopped = True, "nonfinite"
    selected = _save_checkpoint(folder / "selected.pt", model, update=best["update"],
                                role="selected" if validating else "final_refit", metadata=metadata)
    rate = float(fy.mean())
    record = {"history": history, "best_update": int(best["update"]),
              "best_passes": best["update"] * stream.size / stream.n,
              "best_validation_bce": best["validation_bce"],
              "initial_validation_bce": history[0].get("validation_bce"),
              "updates_run": update, "passes_run": update * stream.size / stream.n,
              "stopped": stopped, "nonfinite": nonfinite, "full_batch": stream.full,
              "batch_size": stream.size, "n_fit": len(fx), "n_validation": len(vx) if validating else 0,
              "fit_rate": rate, "mean_residual": implicit._mean_probability(model, fx, chunk) - rate,
              "seconds": round(time.perf_counter() - started, 2), "device": str(device),
              "parameters": model.parameter_count(), "metadata": metadata,
              "artifacts": {"selected": selected, "terminal": terminal, "periodic": periodic}}
    (folder / "history.json").write_text(json.dumps(_safe_json(record), indent=2, allow_nan=False) + "\n")
    return record


def fit_fourier_with_artifacts(data, design, *, artifact_dir, device="cpu",
                               periodic_every=None, compare_passes=False):
    """Return final models/records under the frozen sigma, seed and refit rules.

    ``data`` has Stage 2 FittingData's observed-only interface. Every frequency's
    selected checkpoint and full history are retained; the chosen pre-refit state
    has its own artifact. ``compare_passes`` adds a diagnostic refit under the other
    budget rule, but never replaces the primary model or chooses between rules.
    Enable periodic checkpoints and the extra refit only for development cases.
    """
    from experiments.reconstruction import stage2_field as s2

    folder = Path(artifact_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError(f"Fourier artifact directory is not empty: {folder}")
    rule = design["refit"]["rule"]
    if rule not in {"keep", "updates", "passes"}:
        raise ValueError("unknown refit rule")
    if compare_passes and rule == "keep":
        raise ValueError("matched-pass diagnostic requires an updates or passes primary refit")
    hidden = tuple(design["architecture"]["hidden"])
    box = float(design.get("data", {}).get("box_length", 60.0))
    observation_hashes = _observation_hashes(data)
    models, records = [], []
    for seed in design["seeds"]["seeds"]:
        seed_folder = folder / f"seed_{seed}"
        seed_folder.mkdir()
        base = implicit.base_frequencies(design["architecture"]["fourier_features"],
                                        s2.stream_seed(design, data.index, data.eta, "frequencies", seed))
        init_seed = s2.stream_seed(design, data.index, data.eta, "initialisation", seed, 0)
        batch_seed = s2.stream_seed(design, data.index, data.eta, "minibatch", seed)
        common = {"index": int(data.index), "eta": float(data.eta), "seed": int(seed),
                  "hidden": list(hidden), "split_hash": data.split_hash,
                  "design_hash": s2.design_hash(design), "observation_hashes": observation_hashes,
                  "objective": "unweighted_binary_cross_entropy"}
        candidates = []
        for i, sigma in enumerate(design["sigma_grid"]):
            model = implicit.build_field("fourier", init_seed, base=base, sigma=sigma, hidden=hidden, box=box)
            model.initialise_constant(data.fit_rate)
            record = fit_field_with_artifacts(model, data.fit_x, data.fit_y, data.validation_x, data.validation_y,
                                              artifact_dir=seed_folder / f"candidate_{i:02d}",
                                              metadata={**common, "sigma": float(sigma)},
                                              periodic_every=periodic_every, batch_seed=batch_seed,
                                              device=device, **s2.training_options(design))
            record.update(sigma=float(sigma), seed=int(seed), hidden=list(hidden))
            candidates.append((model, record))
        chosen = s2.select(candidates)
        selected, selected_record = candidates[chosen]
        prerefit = _save_checkpoint(seed_folder / "selected_prerefit.pt", selected,
                                   update=selected_record["best_update"], role="selected_prerefit",
                                   metadata={**common, "sigma": selected_record["sigma"], "candidate_index": chosen})
        entry = {"selected": selected_record, "candidates": [r for _, r in candidates],
                 "candidate_index": chosen, "prerefit_artifact": prerefit}
        final = selected
        if rule != "keep":
            refit_rules = [rule] + (["passes" if rule == "updates" else "updates"] if compare_passes else [])
            for refit_rule in refit_rules:
                updates = s2.refit_updates(selected_record, len(data.observed_x),
                                          design["training"]["batch_size"], refit_rule)
                fresh = implicit.build_field("fourier", init_seed, base=base, sigma=selected_record["sigma"],
                                             hidden=hidden, box=box).initialise_constant(data.observed_rate)
                record = fit_field_with_artifacts(fresh, data.observed_x, data.observed_y,
                                                  artifact_dir=seed_folder / f"refit_{refit_rule}",
                                                  metadata={**common, "sigma": selected_record["sigma"],
                                                            "rule": refit_rule, "diagnostic_only": refit_rule != rule},
                                                  periodic_every=periodic_every, batch_seed=batch_seed,
                                                  device=device, **s2.training_options(design, max_updates=updates))
                record.update(sigma=selected_record["sigma"], seed=int(seed), hidden=list(hidden), rule=refit_rule)
                if refit_rule == rule:
                    final, entry["refit"] = fresh, record
                else:
                    entry["diagnostic_refit"] = record
        entry["final_artifact"] = _save_checkpoint(seed_folder / "final.pt", final,
                                                   update=entry.get("refit", selected_record)["best_update"],
                                                   role="final_prediction_procedure", metadata={**common, "rule": rule})
        models.append(final)
        records.append(entry)
    manifest = {"version": "fourier_training_diagnostics_v1", "design": design,
                "index": int(data.index), "eta": float(data.eta), "split_hash": data.split_hash,
                "observation_hashes": observation_hashes,
                "periodic_every": periodic_every, "compare_passes": compare_passes,
                "records": records,
                "artifact_hashes": {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(folder.rglob("*")) if path.is_file()}}
    (folder / "manifest.json").write_text(json.dumps(_safe_json(manifest), indent=2, allow_nan=False) + "\n")
    return models, records


def load_fourier_artifacts(data, design, *, artifact_dir, device="cpu"):
    """Verify every saved artifact and exact observed input before reusing final fields.

    A partial directory, changed candidate checkpoint, altered observation, or
    changed design fails explicitly. The caller retains such an error as a failed
    attempt; this loader never retrains or silently chooses another checkpoint.
    """
    folder = Path(artifact_dir)
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("version") != "fourier_training_diagnostics_v1" or manifest.get("design") != design
            or manifest.get("index") != int(data.index) or manifest.get("eta") != float(data.eta)
            or manifest.get("split_hash") != data.split_hash
            or manifest.get("observation_hashes") != _observation_hashes(data)):
        raise ValueError("cached Fourier design or observed inputs changed")
    expected = manifest.get("artifact_hashes", {})
    actual_paths = {str(path.relative_to(folder)) for path in folder.rglob("*")
                    if path.is_file() and path != folder / "manifest.json"}
    if not expected or actual_paths != set(expected):
        raise ValueError("cached Fourier artifact inventory changed or is incomplete")
    for relative, checksum in expected.items():
        path = folder / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("invalid artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"cached Fourier artifact changed: {relative}")
    records = manifest["records"]
    if len(records) != len(design["seeds"]["seeds"]):
        raise ValueError("cached Fourier seed count changed")
    models = []
    for seed, record in zip(design["seeds"]["seeds"], records):
        path = folder / f"seed_{seed}" / "final.pt"
        models.append(load_checkpoint(path, device=device,
                                      expected_sha256=record["final_artifact"]["sha256"])[0])
    return models, records


def training_failure_flags(records):
    """Identify failed primary/candidate fits separately from optional diagnostics.

    All candidates stay visible, including unselected failures. A failed optional
    matched-pass fit does not silently replace the primary model; the caller can
    report its failure separately. These flags use training records only.
    """
    flags = {"primary": [], "diagnostic": []}

    def inspect_record(record, name, group):
        reasons = []
        if record.get("nonfinite") is not False or record.get("stopped") == "nonfinite":
            reasons.append("nonfinite_or_missing_status")
        history = record.get("history", [])
        if not history:
            reasons.append("missing_history")
        elif any(not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key])
                 for row in history for key in ("fit_bce", "validation_bce")
                 if key == "fit_bce" or key in row):
            reasons.append("invalid_history_loss")
        if reasons:
            flags[group].append({"record": name, "reasons": reasons})

    for i, entry in enumerate(records):
        for j, candidate in enumerate(entry.get("candidates", [])):
            inspect_record(candidate, f"seed_{i}/candidate_{j}", "primary")
        for part in ("selected", "refit", "diagnostic_refit"):
            if part in entry:
                inspect_record(entry[part], f"seed_{i}/{part}", "diagnostic" if part == "diagnostic_refit" else "primary")
    return flags


def score_particle_signal(fields, coords, labels, centres, radii, *, p_star=None,
                          observed_coords=None, observed_labels=None):
    """Scoring-only direct function diagnostics by true particle and matrix.

    ``fields`` maps names (e.g. Gaussian, selected_prerefit, final_refit) to
    callables returning one probability per coordinate. This evaluates original
    scoring coordinates directly, without an extraction grid. Truth geometry and
    any oracle probabilities enter only this terminal reporting helper. Its
    output must never select checkpoints, boundary proposals, or physical starts.

    The observed arrays should be the exact effective fitting observations of
    the procedure under review. Their counts describe available local evidence;
    they do not become weights or additional training data. True spheres must
    not overlap, so per-particle and matrix summaries form a partition.
    """
    from geom_mesh_net.fields.particle_detection import ParticleGeometry

    coords, labels = _training_arrays(coords, labels, "scoring")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("scoring labels must be binary")
    geometry = ParticleGeometry(centres, radii)
    distances = np.linalg.norm(geometry.centres[:, None, :] - geometry.centres[None, :, :], axis=-1)
    overlap = distances < geometry.radii[:, None] + geometry.radii[None, :]
    np.fill_diagonal(overlap, False)
    if overlap.any():
        raise ValueError("true spheres must not overlap for partitioned diagnostics")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("fields must be a nonempty name-to-callable mapping")
    if (observed_coords is None) != (observed_labels is None):
        raise ValueError("observed coordinates and labels must be supplied together")
    if observed_coords is not None:
        observed_coords, observed_labels = _training_arrays(observed_coords, observed_labels, "observed")
        if not np.all(np.isin(observed_labels, [0, 1])):
            raise ValueError("observed labels must be binary")
    if p_star is not None:
        p_star = np.asarray(p_star, dtype=float)
        if p_star.shape != labels.shape or not np.all(np.isfinite(p_star)) or np.any((p_star < 0) | (p_star > 1)):
            raise ValueError("p_star must be one finite oracle probability in [0, 1] per scoring coordinate")

    def assignment(points):
        result = np.full(len(points), -1, dtype=int)
        for k, (centre, radius) in enumerate(zip(geometry.centres, geometry.radii)):
            result[np.sum((points - centre) ** 2, axis=-1) <= radius ** 2] = k
        return result

    groups = assignment(coords)
    observed_groups = assignment(observed_coords) if observed_coords is not None else None
    predictions = {}
    for name, callable_field in fields.items():
        if not isinstance(name, str) or not name or not callable(callable_field):
            raise ValueError("field names must be nonempty strings and field values callable")
        value = np.asarray(callable_field(coords), dtype=float)
        if value.shape != labels.shape:
            raise ValueError(f"field {name} must return one prediction per scoring coordinate")
        predictions[name] = value

    def summary(mask):
        count = int(mask.sum())
        result = {"atom_count": count, "guest_fraction": float(labels[mask].mean()) if count else None,
                  "oracle_probability_mean": float(p_star[mask].mean()) if count and p_star is not None else None,
                  "fields": {}}
        for name, all_predictions in predictions.items():
            values = all_predictions[mask]
            valid = np.isfinite(values) & (values >= 0) & (values <= 1)
            entry = {"invalid_probability_count": int((~valid).sum()),
                     "probability_mean": None, "probability_max": None, "probability_p90": None,
                     "bce": None, "brier": None, "oracle_mse": None, "mean_prediction_bias": None}
            # Invalid predictions invalidate the whole region's score; do not
            # quietly score only the finite subset or clip material violations.
            if count and valid.all():
                clipped = np.clip(values, implicit.RATE_CLIP, 1 - implicit.RATE_CLIP)
                targets = labels[mask]
                entry.update(probability_mean=float(values.mean()), probability_max=float(values.max()),
                             probability_p90=float(np.quantile(values, .9)),
                             bce=float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))),
                             brier=float(np.mean((values - targets) ** 2)))
                if p_star is not None:
                    entry.update(oracle_mse=float(np.mean((values - p_star[mask]) ** 2)),
                                 mean_prediction_bias=float(np.mean(values - p_star[mask])))
            result["fields"][name] = entry
        return result

    particles = []
    matrix = summary(groups == -1)
    for k, radius in enumerate(geometry.radii):
        result = summary(groups == k)
        observed = observed_groups == k if observed_groups is not None else None
        result.update(truth_index=k, radius=float(radius), diameter=float(2 * radius),
                      observed_atom_count=int(observed.sum()) if observed is not None else None,
                      observed_guest_count=int(observed_labels[observed].sum()) if observed is not None else None,
                      observed_guest_fraction=float(observed_labels[observed].mean())
                      if observed is not None and observed.any() else None,
                      oracle_particle_matrix_contrast=(result["oracle_probability_mean"] - matrix["oracle_probability_mean"])
                      if result["oracle_probability_mean"] is not None and matrix["oracle_probability_mean"] is not None else None)
        particles.append(result)
    return {"purpose": "scoring_only_never_selection", "evaluated_at": "supplied_atom_coordinates_directly",
            "whole": summary(np.ones(len(coords), dtype=bool)), "matrix": matrix, "particles": particles,
            "field_names": list(fields), "observed_atom_count": len(observed_coords) if observed_coords is not None else None}
