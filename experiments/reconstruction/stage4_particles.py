"""Particle detection from the Fourier field, followed by the same joint physics fit.

Run ``--phase calibration``, then ``--phase confirmation``. ``--phase test`` is
refused unless the completed confirmation passes its fixed promotion conditions.
``--phase run`` executes the pilot and conditionally the reserved test. Every phase
resumes only compatible records. Large field checkpoints/grids are cached separately
from scoring so a reporting change does not require training the neural field again.

This is a diffusion-data pilot at one efficiency, not the full Stage 4 claim over
the original reconstruction benchmark or a test of rejection of incorrect physics.
"""

import argparse
import hashlib
import inspect
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from experiments.reconstruction import benchmark as bm
from experiments.reconstruction import generate_diffusion_patterns as gd
from experiments.reconstruction import stage2_field as s2
from experiments.reconstruction import stage5_physics_fit as s5
from geom_mesh_net import paths
from geom_mesh_net.fields import baselines as fb
from geom_mesh_net.fields import cluster_extraction as ce
from geom_mesh_net.fields import joint_fit as jf
from geom_mesh_net.fields import particle_detection as pd
from geom_mesh_net.fields import particle_physics as pp
from geom_mesh_net.neural import implicit

ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path(__file__).with_name("stage4_particles_design.json")
RESULTS = paths.RECONSTRUCTION_DIR / "results"
ARTIFACTS = RESULTS / "stage4_artifacts"
METHODS = ("legacy_b1", "b1", "fourier", "dbscan", "maximum_separation")
TUNED = METHODS[1:]
REFERENCE_METHODS = ("legacy_b1", "b1", "dbscan", "maximum_separation")
SOURCE_PATHS = (
    "experiments/reconstruction/stage4_particles.py",
    "experiments/reconstruction/stage2_field.py",
    "experiments/reconstruction/stage5_physics_fit.py",
    "experiments/reconstruction/benchmark.py",
    "geom_mesh_net/fields/particle_detection.py",
    "geom_mesh_net/fields/particle_physics.py",
    "geom_mesh_net/fields/cluster_extraction.py",
    "geom_mesh_net/fields/baselines.py",
    "geom_mesh_net/fields/joint_fit.py",
    "geom_mesh_net/fields/physics.py",
    "geom_mesh_net/neural/implicit.py",
)


def json_safe(value):
    return pp._json_safe(value)


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_design(path=DESIGN_PATH):
    design = json.loads(Path(path).read_text())
    groups = [design["scope"][name] for name in ("calibration", "confirmation", "test")]
    flat = [p for group in groups for p in group]
    if len(flat) != len(set(flat)):
        raise ValueError("calibration, confirmation and test patterns must be unique and disjoint")
    if any(p < 0 or p >= 50 for p in groups[0] + groups[1]):
        raise ValueError("calibration and confirmation must use development patterns")
    if any(p < 74 or p >= 150 for p in design["scope"]["test"]):
        raise ValueError("test patterns must be beyond the published Stage 5.2 scope")
    if not 0 < design["efficiency"] < 1 or not 0 < design["held_back_fraction"] < 1:
        raise ValueError("efficiency and held-back fraction must be in (0, 1)")
    if any(design["promotion"].get(key) is not True for key in (
            "require_common_radius_improvement", "require_capillary_improvement", "require_all_fourier_fits_reliable")):
        raise ValueError("this experiment requires geometry, capillarity and validity promotion checks")
    return design


def field_design(design):
    return s2.load_design(DESIGN_PATH.parent / design["field_design"])


def source_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def provenance(design):
    inputs = {}
    for index in sorted({p for group in design["scope"].values() for p in group}):
        # Hashing reserved files records identity without inspecting their outcomes.
        files = {"dataset": paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz",
                 "oracle": s5.stage5_simulator.oracle_path(index)}
        inputs[str(index)] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in files.items()}
    return {"design": design, "field_design": field_design(design), "sources": source_hashes(),
            "inputs": inputs,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__,
                        "device": design["device"], "threads": design["threads"]}}


def split_indices(n, index, design, neural_design):
    """Positions of fitting/validation/held-back/removed atoms, with no labels as input."""
    eta = design["efficiency"]
    observed_mask = bm.thinning_mask(index, eta, n, entropy=gd.THINNING_ENTROPY)
    draw = np.random.default_rng(np.random.SeedSequence(
        [design["outer_split_entropy"], index, int(round(eta * 1000))])).random(n)
    fitting = np.flatnonzero(observed_mask & (draw >= design["held_back_fraction"]))
    held_back = np.flatnonzero(observed_mask & (draw < design["held_back_fraction"]))
    inner = np.random.default_rng(s2.stream_seed(neural_design, index, eta, "split")).random(len(fitting))
    validation = inner < neural_design["split"]["validation_fraction"]
    return {"fitting": fitting, "inner_fit": fitting[~validation],
            "inner_validation": fitting[validation], "held_back": held_back,
            "removed": np.flatnonzero(~observed_mask)}


def load_observations(index, design):
    """Only observed arrays enter FittingData; geometry/oracle are loaded separately for scoring."""
    path = paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz"
    with np.load(path, allow_pickle=True) as data:
        coords, labels = data["coords"].item(), data["labels"]
        checksum = gd.content_hash(coords, labels, data["radii"], data["centers"].item())
    frozen = json.loads((gd.FROZEN_DIR / "dataset_checksums.json").read_text())
    if checksum != frozen[str(index)]:
        raise RuntimeError(f"diffusion pattern {index} differs from its frozen content hash")
    x = np.column_stack([coords[a] for a in "xyz"])
    y = np.isin(labels, (2, 3)).astype(float)
    neural_design = field_design(design)
    split = split_indices(len(x), index, design, neural_design)
    fit, val, observed = (split[k] for k in ("inner_fit", "inner_validation", "fitting"))
    split_hash = array_hash(*[split[k] for k in sorted(split)])
    fitting = s2.FittingData(index, design["efficiency"], x[fit], y[fit], x[val], y[val],
                             x[observed], y[observed], float(y[fit].mean()), float(y[observed].mean()), split_hash)
    checks = {"dataset_hash": checksum, "split_hash": split_hash,
              "counts": {key: len(value) for key, value in split.items()},
              "observed_input_hash": array_hash(fitting.observed_x, fitting.observed_y)}
    return fitting, (x, y, split), checks


def load_truth(index, x):
    with np.load(paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz", allow_pickle=True) as data:
        physical = data["physics"].item()
    p_star, inside = s5.stage5_simulator.load_oracle(index)
    centres, radii = np.asarray(physical["centres"]), np.asarray(physical["radii"])
    composition = []
    for centre, radius in zip(centres, radii):
        mask = np.linalg.norm(x - centre, axis=1) <= radius
        composition.append(float(p_star[mask].mean()))
    return {"centres": centres, "radii": radii, "constants": physical,
            "probability": p_star, "inside": inside, "composition": composition}


def grid_coordinates(design):
    n = int(round(design["box"] / design["grid"]))
    if not np.isclose(n * design["grid"], design["box"]):
        raise ValueError("grid must divide box length")
    axis = (np.arange(n) + 0.5) * design["grid"]
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3), (n, n, n)


def training_fingerprint(data, design, checks):
    """Cache fitting only: detector/scoring edits cannot relabel an old training procedure."""
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS
               if p.endswith(("implicit.py", "stage2_field.py", "baselines.py", "cluster_extraction.py"))}
    preparation = {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                   for f in (split_indices, load_observations, prepare_fields, grid_coordinates, save_field_checkpoint)}
    return digest({"field_design": field_design(design), "sources": sources, "checks": checks,
                   "preparation_sources": preparation,
                   "grid": design["grid"], "box": design["box"], "device": design["device"],
                   "threads": design["threads"], "torch": torch.__version__, "numpy": np.__version__})


def save_field_checkpoint(path, model, design):
    """Save tensors and architecture; weights_only loading never unpickles a model object."""
    torch.save({"architecture": design["architecture"], "state_dict": model.cpu().state_dict()}, path)


def load_field_checkpoint(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    architecture = bundle["architecture"]
    model = implicit.build_field("fourier", 0, base=np.zeros((architecture["fourier_features"], 3)),
                                  sigma=1, hidden=tuple(architecture["hidden"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def prepare_fields(index, design, artifacts=ARTIFACTS):
    data, observations, checks = load_observations(index, design)
    fingerprint = training_fingerprint(data, design, checks)
    folder = Path(artifacts) / f"pattern_{index}_{fingerprint[:16]}"
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["fingerprint"] != fingerprint:
            raise RuntimeError("cached field fingerprint mismatch")
        for name, expected in metadata["artifact_hashes"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"cached artifact changed: {folder / name}")
        with np.load(folder / "grids.npz") as grids:
            arrays = {key: grids[key] for key in grids.files}
        return data, observations, checks, metadata, arrays, folder
    started = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    print(f"pattern {index}: fitting Gaussian baseline", flush=True)
    baseline = fb.fit_baselines(data.observed_x, data.observed_y.astype(bool), seed=index,
                                adaptive_k=(4,), adaptive_c=(0.5,))
    guests, atoms = ce.smoothed_counts(data.observed_x, data.observed_y.astype(bool), baseline.bandwidth,
                                      upper=design["box"], grid=design["grid"])
    squared = ce.squared_kernel_counts(data.observed_x, baseline.bandwidth, upper=design["box"], grid=design["grid"])
    b1 = ce.guest_fraction(guests, atoms)
    print(f"pattern {index}: fitting Fourier candidates and refit", flush=True)
    models, records = s2.fit_model(data, field_design(design), "fourier", design["device"])
    if len(models) != 1:
        raise ValueError("this checkpoint format expects the frozen single-seed Stage 2 design")
    if any(r.get(part, {}).get("nonfinite") for r in records for part in ("selected", "refit")):
        raise RuntimeError("neural field training became non-finite")
    points, shape = grid_coordinates(design)
    fourier = s2.predict_procedure(models, points, design["device"]).reshape(shape)
    arrays = {"b1": b1.astype(np.float32), "fourier": fourier.astype(np.float32),
              "atoms": atoms.astype(np.float32), "squared": squared.astype(np.float32)}
    np.savez_compressed(folder / "grids.npz", **arrays)
    save_field_checkpoint(folder / "field.pt", models[0], field_design(design))
    metadata = {"fingerprint": fingerprint, "checks": checks, "bandwidth": baseline.bandwidth,
                "bandwidth_cv": baseline.cv_loss_fixed, "training": records,
                "seconds": round(time.perf_counter() - started, 2),
                "artifact_hashes": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                    for name in ("field.pt", "grids.npz")}}
    save_json(metadata_path, metadata)
    print(f"pattern {index}: fields ready in {metadata['seconds']:.1f}s", flush=True)
    return data, observations, checks, metadata, arrays, folder


def candidates(design, method):
    if method == "legacy_b1":
        return [{}]
    if method in ("b1", "fourier"):
        return [dict(threshold=t, peak_distance=p, min_volume=design["field_min_volume"])
                for t, p in itertools.product(design["field_thresholds"], design["field_peak_distances"])]
    if method == "dbscan":
        return [dict(eps=e, min_samples=m, min_cluster_size=n) for e, m, n in itertools.product(
            design["point_eps"], design["dbscan_min_samples"], design["point_min_cluster_size"])]
    if method == "maximum_separation":
        return [dict(eps=e, min_cluster_size=n) for e, n in itertools.product(
            design["point_eps"], design["point_min_cluster_size"])]
    raise ValueError(method)


def detect(method, setting, data, grids, design):
    if method == "legacy_b1":
        segmented = ce.segment(grids["b1"], grids["atoms"], grid=design["grid"],
                               method="significance", squared=grids["squared"])
        return pd.ParticleGeometry(segmented.centres, segmented.radii,
                                   {"method": "existing Gaussian-count significance detector"})
    if method in ("b1", "fourier"):
        valid = grids["atoms"] > 0.2 * np.median(grids["atoms"])
        return pd.extract_field(grids[method], grid=design["grid"], valid=valid, **setting)
    return pd.cluster_points(data.observed_x, data.observed_y, method=method, **setting)


def particle_scores(geometry, truth, observations):
    x, y, split = observations
    ids = split["held_back"]
    measured = pd.measure_particles(geometry, x[ids], y[ids])
    metrics = pd.evaluate_particles(geometry, truth["centres"], truth["radii"],
                                    true_composition=truth["composition"], measured=measured)
    metrics["composition_target"] = "oracle mean in each true sphere; measurements use held-back atoms in predicted spheres"
    return {"geometry": geometry.to_dict(), "metrics": metrics, "measurement": measured}


def mean(values):
    return float(np.mean(values)) if len(values) and all(v is not None and np.isfinite(v) for v in values) else None


def median(values):
    return float(np.median(values)) if len(values) and all(v is not None and np.isfinite(v) for v in values) else None


def selection_key(metrics, position=0):
    f1, precision = mean([m["f1"] for m in metrics]), mean([m["precision"] for m in metrics])
    radius = mean([m["relative_radius_mae"] for m in metrics])
    return (-f1 if f1 is not None else float("inf"),
            -precision if precision is not None else float("inf"),
            radius if radius is not None else float("inf"), position)


def select_settings(cells, design):
    if sorted(c["pattern"] for c in cells) != sorted(design["scope"]["calibration"]):
        raise ValueError("calibration must contain every required pattern exactly once")
    if any("error" in c for c in cells):
        raise ValueError("calibration contains failed patterns")
    for cell in cells:
        if set(cell.get("candidates", {})) != set(METHODS):
            raise ValueError("calibration candidate methods do not match the design")
        for method in METHODS:
            expected = candidates(design, method)
            actual = cell["candidates"][method]
            if len(actual) != len(expected) or [entry.get("setting") for entry in actual] != expected:
                raise ValueError(f"calibration candidate grid changed for {method}")
            for entry in actual:
                for metric in ("f1", "precision", "recall"):
                    value = entry.get("metrics", {}).get(metric)
                    if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"invalid calibration metric {metric}")
    settings, aggregates = {"legacy_b1": {}}, {}
    for method in METHODS:
        options = candidates(design, method)
        scores = [[c["candidates"][method][i]["metrics"] for c in cells] for i in range(len(options))]
        order = sorted(range(len(options)), key=lambda i: selection_key(scores[i], i))
        chosen = order[0]
        settings[method] = options[chosen]
        aggregates[method] = {"candidate": chosen, "mean_f1": mean([m["f1"] for m in scores[chosen]]),
                              "mean_precision": mean([m["precision"] for m in scores[chosen]]),
                              "mean_relative_radius_mae": mean([m["relative_radius_mae"] for m in scores[chosen]])}
    reference = min(REFERENCE_METHODS, key=lambda m: (
        -aggregates[m]["mean_f1"], -aggregates[m]["mean_precision"],
        aggregates[m]["mean_relative_radius_mae"] if aggregates[m]["mean_relative_radius_mae"] is not None else float("inf"),
        REFERENCE_METHODS.index(m)))
    return {"settings": settings, "calibration_scores": aggregates, "reference": reference,
            "physics_methods": list(dict.fromkeys(("legacy_b1", "fourier", reference))),
            "selection": design["selection"], "frozen_after": "calibration"}


def calibration_cell(index, design, artifacts):
    started = time.perf_counter()
    data, observations, checks, metadata, grids, folder = prepare_fields(index, design, artifacts)
    truth = load_truth(index, observations[0])
    out = {"pattern": index, "checks": checks, "field": metadata, "artifact_folder": str(folder), "candidates": {}}
    for method in METHODS:
        entries = []
        for setting in candidates(design, method):
            geometry = detect(method, setting, data, grids, design)
            metrics = pd.evaluate_particles(geometry, truth["centres"], truth["radii"])
            entries.append({"setting": setting, "metrics": {k: v for k, v in metrics.items()
                                                              if not isinstance(v, (list, dict))}})
        out["candidates"][method] = entries
    out["seconds"] = round(time.perf_counter() - started, 2)
    return out


def common_radius_errors(a, b):
    left = dict(zip(a["matched_true_indices"], np.abs(a["relative_radius_error"])))
    right = dict(zip(b["matched_true_indices"], np.abs(b["relative_radius_error"])))
    common = sorted(left.keys() & right.keys())
    return {"count": len(common), "fourier": mean([left[i] for i in common]),
            "reference": mean([right[i] for i in common]),
            "difference": mean([left[i] - right[i] for i in common])}


def field_scores(data, observations, truth, metadata, folder, design):
    """Diagnostic field transfer scores; never used to select detectors or promote a run."""
    x, y, split = observations
    counts = fb.SmoothedCounts(data.observed_x, data.observed_y, bandwidths=(metadata["bandwidth"],),
                               grid=design["grid"], upper=design["box"])
    model = load_field_checkpoint(Path(folder) / "field.pt").to(design["device"])
    result = {}
    for subset in ("held_back", "removed"):
        ids = split[subset]
        predictions = {"constant": np.full(len(ids), data.observed_rate),
                       "b1": counts.fixed(x[ids], 0),
                       "fourier": s2.predict_procedure([model], x[ids], design["device"])}
        oracle_loss = fb.log_loss(y[ids], truth["probability"][ids])
        result[subset] = {"atoms": len(ids), "oracle_loss": oracle_loss, "methods": {
            name: {"loss": fb.log_loss(y[ids], q),
                   "excess_loss": fb.log_loss(y[ids], q) - oracle_loss,
                   "mean_squared_probability_error": float(np.mean((q - truth["probability"][ids]) ** 2))}
            for name, q in predictions.items()}}
    return result


def evaluate_cell(index, design, selection, artifacts):
    started = time.perf_counter()
    data, observations, checks, metadata, grids, folder = prepare_fields(index, design, artifacts)
    x, y, split = observations
    truth = load_truth(index, x)
    out = {"pattern": index, "checks": checks, "field": metadata, "artifact_folder": str(folder),
           "methods": {}, "physics": {}}
    out["field_scores"] = field_scores(data, observations, truth, metadata, folder, design)
    geometries = {}
    for method in METHODS:
        geometries[method] = detect(method, selection["settings"][method], data, grids, design)
        out["methods"][method] = particle_scores(geometries[method], truth, observations)
    reference = selection["reference"]
    out["common_matched_radius"] = common_radius_errors(out["methods"]["fourier"]["metrics"],
                                                         out["methods"][reference]["metrics"])
    matrix_removed = split["removed"][~truth["inside"][split["removed"]]]
    oracle_matrix_loss = fb.log_loss(y[matrix_removed], truth["probability"][matrix_removed])
    for method in selection["physics_methods"]:
        geometry = geometries[method]
        centres, radii = ce.resolve_overlaps(geometry.centres, geometry.radii, gap=design["initial_surface_gap"])
        adjusted = pd.ParticleGeometry(centres, radii, {"overlap_resolution_gap": design["initial_surface_gap"]})
        print(f"pattern {index}: joint physics for {method}, {len(radii)} particles", flush=True)
        diagnostics = np.concatenate([split["held_back"], split["removed"]])
        model, fit = pp.fit_particle_physics(centres, radii, data.observed_x, data.observed_y,
                                             validation_points=x[diagnostics], **design["physics"])
        fit["probability_check_domain"] = "held-back and removed positions outside fitted spheres; no labels"
        fit["adjusted_geometry"] = particle_scores(adjusted, truth, observations)
        if model is not None:
            fit["relative_error"] = {k: fit["constants"][k] / truth["constants"][k] - 1
                                      for k in ("ell", "xi", "c_eq", "c_inf")}
            fit["held_back_loss"] = fb.log_loss(y[split["held_back"]], jf.predict(model, x[split["held_back"]]))
            fit["removed_loss"] = fb.log_loss(y[split["removed"]], jf.predict(model, x[split["removed"]]))
            fit["matrix_excess"] = fb.log_loss(y[matrix_removed], jf.predict(model, x[matrix_removed], matrix_only=True)) - oracle_matrix_loss
            fitted_geometry = pd.ParticleGeometry(fit["centres"], fit["radii"])
            fit["fitted_geometry"] = particle_scores(fitted_geometry, truth, observations)
        out["physics"][method] = fit
        print(f"pattern {index}: {method} status={fit['status']} ell_error={fit.get('relative_error', {}).get('ell')}", flush=True)
    out["seconds"] = round(time.perf_counter() - started, 2)
    return out


def completeness(cells, expected, selection=None):
    ids = [c["pattern"] for c in cells]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    missing, unexpected = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
    errors = [c["pattern"] for c in cells if "error" in c]
    missing_methods, unexpected_methods = [], []
    if selection is not None:
        for cell in cells:
            if "error" in cell:
                continue
            for group, required in (("methods", METHODS), ("physics", selection["physics_methods"])):
                for method in required:
                    if method not in cell.get(group, {}):
                        missing_methods.append([cell["pattern"], group, method])
                for method in set(cell.get(group, {})) - set(required):
                    unexpected_methods.append([cell["pattern"], group, method])
    return {"complete": not (missing or unexpected or duplicates or errors or missing_methods or unexpected_methods),
            "missing": missing, "unexpected": unexpected, "duplicates": duplicates,
            "errored": errors, "missing_methods": missing_methods, "unexpected_methods": unexpected_methods}


def promotion(cells, design, selection, phase="confirmation"):
    checked = completeness(cells, design["scope"][phase], selection)
    result = {**checked, "passed": False, "verdict": "incomplete", "conditions": {}}
    if not checked["complete"]:
        return result
    reference = selection["reference"]
    metrics = lambda c, m: c["methods"][m]["metrics"]
    f1_wins = [metrics(c, "fourier")["f1"] > metrics(c, reference)["f1"] for c in cells]
    precision = median([metrics(c, "fourier")["precision"] - metrics(c, reference)["precision"] for c in cells])
    radius = median([c["common_matched_radius"]["difference"] for c in cells])
    capillary, matrix = [], []
    for c in cells:
        a, b = c["physics"]["fourier"], c["physics"][reference]
        capillary.append(abs(a["relative_error"]["ell"]) - abs(b["relative_error"]["ell"])
                         if "relative_error" in a and "relative_error" in b else None)
        matrix.append(a["matrix_excess"] - b["matrix_excess"] if "matrix_excess" in a and "matrix_excess" in b else None)
    capillary_delta, matrix_delta = median(capillary), median(matrix)
    rules = design["promotion"]
    conditions = {
        "f1_win_fraction": {"value": mean(f1_wins), "required_at_least": rules["f1_win_fraction"],
                            "passed": bool(mean(f1_wins) >= rules["f1_win_fraction"])},
        "median_precision_difference": {"value": precision, "required_at_least": rules["minimum_median_precision_difference"],
                                        "passed": precision is not None and precision >= rules["minimum_median_precision_difference"]},
        "common_matched_radius_difference": {"value": radius, "required_below": 0,
                                             "passed": radius is not None and radius < 0},
        "capillary_error_difference": {"value": capillary_delta, "required_below": 0,
                                       "passed": capillary_delta is not None and capillary_delta < 0},
        "matrix_excess_difference": {"value": matrix_delta, "required_at_most": rules["maximum_median_matrix_loss_difference"],
                                     "passed": matrix_delta is not None and matrix_delta <= rules["maximum_median_matrix_loss_difference"]},
        "all_fourier_fits_reliable": {"value": sum(c["physics"]["fourier"]["reliable"] for c in cells),
                                      "required": len(cells),
                                      "passed": all(c["physics"]["fourier"]["reliable"] for c in cells)},
    }
    result.update(conditions=conditions, passed=all(v["passed"] for v in conditions.values()))
    result["verdict"] = "passed" if result["passed"] else "failed"
    return result


def paired_interval(values, reps=2000):
    """Exploratory pattern-bootstrap interval; never used for fitting or promotion."""
    if median(values) is None:
        return {"median": None, "interval95": None, "patterns": len(values)}
    array = np.asarray(values)
    samples = np.random.default_rng(20261015).choice(array, size=(reps, len(array)))
    return {"median": float(np.median(array)),
            "interval95": np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist(),
            "patterns": len(values), "note": "exploratory paired pattern bootstrap; small pilot"}


def summarise(cells, reference=None):
    valid = [c for c in cells if "error" not in c]
    out = {"patterns": len(cells), "errors": len(cells) - len(valid), "methods": {}}
    for method in METHODS:
        entries = [c["methods"][method]["metrics"] for c in valid if method in c.get("methods", {})]
        if not entries:
            continue
        values = {k: median([r[k] for r in entries]) for k in
                  ("f1", "precision", "recall", "relative_radius_mae", "centre_mae", "composition_mae")}
        values["false_positives"] = sum(r["false_positive_count"] for r in entries)
        values["missed"] = sum(r["missed_count"] for r in entries)
        fits = [c["physics"][method] for c in valid if method in c.get("physics", {})]
        if fits:
            values["physics"] = {"fits": len(fits), "reliable": sum(f["reliable"] for f in fits),
                                  "median_abs_relative_ell_error": median([abs(f["relative_error"]["ell"]) if "relative_error" in f else None for f in fits]),
                                  "median_abs_relative_xi_error": median([abs(f["relative_error"]["xi"]) if "relative_error" in f else None for f in fits]),
                                  "median_matrix_excess": median([f.get("matrix_excess") for f in fits]),
                                  "flags": [dict(pattern=c["pattern"], flags=c["physics"][method]["flags"])
                                            for c in valid if method in c.get("physics", {}) and c["physics"][method]["flags"]]}
        out["methods"][method] = values
    if reference is not None:
        paired = {}
        for metric in ("f1", "precision", "recall", "relative_radius_mae"):
            values = []
            for cell in valid:
                a = cell["methods"]["fourier"]["metrics"][metric]
                b = cell["methods"][reference]["metrics"][metric]
                values.append(a - b if a is not None and b is not None else None)
            paired[metric] = paired_interval(values)
        paired["common_matched_radius"] = paired_interval([c["common_matched_radius"]["difference"] for c in valid])
        for metric in ("ell", "xi"):
            values = []
            for cell in valid:
                a, b = cell["physics"]["fourier"], cell["physics"][reference]
                values.append(abs(a["relative_error"][metric]) - abs(b["relative_error"][metric])
                              if "relative_error" in a and "relative_error" in b else None)
            paired[f"absolute_relative_{metric}_error"] = paired_interval(values)
        out["paired_fourier_minus_reference"] = {"reference": reference, "metrics": paired}
    return out


def report_path(output, phase):
    return Path(output) / f"stage4_particles_{phase}.json"


def read_compatible(path, fingerprint):
    report = json.loads(Path(path).read_text())
    if report.get("fingerprint") != fingerprint:
        raise RuntimeError(f"{path}: provenance changed; use a new output directory, do not mix runs")
    return report


def require_test_permission(confirmation, design, selection):
    """A saved 'passed' boolean is insufficient: recompute all conditions from the cases."""
    decision = promotion(confirmation["cells"], design, selection)
    if not decision["passed"]:
        raise RuntimeError("reserved test refused: confirmation is incomplete or failed")


def run_phase(phase, design, output, artifacts, patterns=None):
    torch.set_num_threads(design["threads"])
    prov = provenance(design)
    fingerprint = digest(prov)
    target = report_path(output, phase)
    requested = design["scope"][phase] if patterns is None else patterns
    if len(requested) != len(set(requested)) or not set(requested) <= set(design["scope"][phase]):
        raise ValueError("requested patterns must be a unique subset of the phase's prescribed scope")
    selection = None
    if phase != "calibration":
        calibration = read_compatible(report_path(output, "calibration"), fingerprint)
        selection = select_settings(calibration["cells"], design)
        if phase == "test":
            confirmation = read_compatible(report_path(output, "confirmation"), fingerprint)
            require_test_permission(confirmation, design, selection)
    report = read_compatible(target, fingerprint) if target.exists() else {
        "phase": phase, "fingerprint": fingerprint, "provenance": prov, "cells": [], "selection": selection}
    if report.get("selection") != selection:
        raise RuntimeError("frozen detector selection differs from the saved report")
    if len({c["pattern"] for c in report["cells"]}) != len(report["cells"]):
        raise RuntimeError("duplicate saved pattern identities")
    done = {c["pattern"] for c in report["cells"]}
    for cell in report["cells"]:
        if "error" in cell:
            continue
        _, _, checks = load_observations(cell["pattern"], design)
        if checks != cell.get("checks"):
            raise RuntimeError("saved observations or data split changed; use a new output directory")
    for index in requested:
        if index in done:
            continue
        try:
            cell = calibration_cell(index, design, artifacts) if phase == "calibration" else evaluate_cell(index, design, selection, artifacts)
        except Exception as error:
            cell = {"pattern": index, "error": f"{type(error).__name__}: {error}"}
        report["cells"].append(cell)
        report["cells"].sort(key=lambda c: c["pattern"])
        report["completeness"] = completeness(report["cells"], design["scope"][phase], selection)
        if phase != "calibration":
            report["decision"] = promotion(report["cells"], design, selection, phase)
            report["summary"] = summarise(report["cells"], selection["reference"])
        save_json(target, report)
        print(f"{phase} pattern {index}: {'ERROR ' + cell['error'] if 'error' in cell else 'recorded'}", flush=True)
    report["completeness"] = completeness(report["cells"], design["scope"][phase], selection)
    if phase == "calibration" and report["completeness"]["complete"]:
        report["frozen_selection"] = select_settings(report["cells"], design)
    elif phase != "calibration":
        report["decision"] = promotion(report["cells"], design, selection, phase)
        report["summary"] = summarise(report["cells"], selection["reference"])
    save_json(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "confirmation", "test", "run"), default="run")
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--patterns", help="comma-separated subset for a partial run; never changes required scope")
    args = parser.parse_args()
    design = load_design(args.design)
    patterns = [int(p) for p in args.patterns.split(",")] if args.patterns else None
    if args.phase == "run":
        if patterns is not None:
            parser.error("--patterns requires an explicit phase")
        calibration = run_phase("calibration", design, args.output_dir, args.artifacts_dir)
        if not calibration["completeness"]["complete"]:
            raise RuntimeError("calibration incomplete; inspect recorded failures")
        confirmation = run_phase("confirmation", design, args.output_dir, args.artifacts_dir)
        if confirmation["decision"]["passed"]:
            run_phase("test", design, args.output_dir, args.artifacts_dir)
        else:
            print("Confirmation did not pass. Reserved test patterns were not evaluated.", flush=True)
    else:
        run_phase(args.phase, design, args.output_dir, args.artifacts_dir, patterns)


if __name__ == "__main__":
    main()
