"""Fresh, independently seeded E21 simulations; no historical dataset is read.

The matrix is the existing screened field. Interiors retain the simulator family's
fixed-count, distance-weighted sampling, so marks within a particle are dependent.
Geometry is a new, explicit variant: every sphere is fully inside the box. Its
distribution is not claimed to be identical to the historical buffered-centre
generator, nor is the sharp interface matched to the fitter's smooth blend.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from experiments.reconstruction import e21_protocol as protocol
from experiments.reconstruction import stage2_field as s2
from geom_mesh_net.fields import physics


PRIOR_NAMES = ("cr", "rb", "volume_fraction", "rho_c", "matrix_mean", "ell",
               "critical_radius_fraction")


def generate_case(spec, design):
    """Generate exactly one case; failures are not replaced by new random draws."""
    config = deepcopy(design["simulation"])
    if config["version"] != "e21-full-sphere-fixed-count-v1":
        raise ValueError("unsupported simulation version")
    seeds = spec["seeds"]
    parameter_rng = np.random.default_rng(seeds["parameters"])
    p = {name: float(parameter_rng.uniform(*config[name])) for name in PRIOR_NAMES}
    box, count = float(config["box"]), int(config["atom_count"])
    if box <= 0 or count < 1:
        raise ValueError("positive box and atom count required")
    # The analytic normal third moment sets an approximate target volume fraction;
    # truncation and realised radii mean the realised fraction is reported separately.
    expected_volume = 4 * np.pi / 3 * p["cr"] ** 3 * (1 + 3 * p["rb"] ** 2)
    number = max(1, int(round(p["volume_fraction"] * box ** 3 / expected_volume)))
    geometry_rng = np.random.default_rng(seeds["geometry"])
    radii = geometry_rng.normal(p["cr"], p["cr"] * p["rb"], number)
    for _ in range(1000):
        low = radii < config["min_radius"]
        if not low.any():
            break
        radii[low] = geometry_rng.normal(p["cr"], p["cr"] * p["rb"], int(low.sum()))
    else:
        raise RuntimeError("radius draw exhausted its declared truncation budget")
    if np.any(2 * radii >= box):
        raise RuntimeError("a drawn sphere cannot fit inside the box")
    centres = np.empty((number, 3))
    placed = []
    proposals = 0
    # Largest first aids packing without rejecting and redrawing difficult radii.
    for k in np.argsort(-radii, kind="stable"):
        for _ in range(int(config["placement_attempts"])):
            candidate = geometry_rng.uniform(radii[k], box - radii[k], 3)
            proposals += 1
            if not placed or np.all(np.linalg.norm(centres[placed] - candidate, axis=1)
                                    >= radii[placed] + radii[k] + config["min_gap"]):
                centres[k] = candidate
                placed.append(int(k))
                break
        else:
            raise RuntimeError(f"placement budget exhausted at sphere {k}")
    x = np.random.default_rng(seeds["positions"]).uniform(0, box, (count, 3))
    owner = np.full(count, -1, dtype=np.int32)
    for start in range(0, count, 8000):
        distances = cdist(x[start:start + 8000], centres)
        inside = distances <= radii
        if np.any(inside.sum(axis=1) > 1):
            raise RuntimeError("generated overlapping spheres")
        owner[start:start + 8000] = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    matrix = owner < 0
    supersaturation = np.exp(p["ell"] / (p["critical_radius_fraction"] * p["cr"]))
    field, q = physics.fit_to_matrix_mean(centres, radii, x[matrix], ell=p["ell"],
                                        supersaturation=supersaturation,
                                        matrix_mean=p["matrix_mean"], volume=box ** 3)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise RuntimeError("generated matrix probabilities are invalid; no clipping or replacement")
    label_rng = np.random.default_rng(seeds["labels"])
    y = np.zeros(count, dtype=np.uint8)
    interior_counts = []
    for k in range(number):
        indices = np.flatnonzero(owner == k)
        guests = int(round(len(indices) * p["rho_c"]))
        if guests >= len(indices):
            chosen = indices
        elif guests == 0:
            chosen = np.empty(0, dtype=int)
        else:
            distances = np.linalg.norm(x[indices] - centres[k], axis=1)
            weights = 1 - distances / distances.max() if distances.max() > 0 else np.ones(len(indices))
            if weights.sum() == 0:
                weights = np.ones(len(indices))
            chosen = label_rng.choice(indices, size=guests, replace=False, p=weights / weights.sum())
        y[chosen] = 1
        interior_counts.append({"atoms": len(indices), "guests": guests})
    y[matrix] = label_rng.random(int(matrix.sum())) < q
    observed = np.random.default_rng(seeds["thinning"]).random(count) < design["efficiency"]
    heldback = observed & (np.random.default_rng(seeds["split"]).random(count)
                           < design["held_back_fraction"])
    fitting = observed & ~heldback
    removed = ~observed
    metadata = {"case": deepcopy(spec), "configuration": config, "parameters": p,
                "observation": {"efficiency": design["efficiency"],
                                "held_back_fraction": design["held_back_fraction"]},
                "field": field.to_dict(), "placement_proposals": proposals,
                "realised_volume_fraction": float((4 * np.pi / 3 * radii ** 3).sum() / box ** 3),
                "interior_counts": interior_counts,
                "matrix_probability_range": [float(q.min()), float(q.max())],
                "probability_clipping": False,
                "counts": {"all": count, "fitting": int(fitting.sum()),
                           "heldback": int(heldback.sum()), "removed": int(removed.sum())},
                "model_mismatch": "sharp interface; dependent fixed-count interior marks"}
    arrays = dict(x=x, y=y, centres=centres, radii=radii, owner=owner,
                  fitting=fitting, heldback=heldback, removed=removed)
    metadata["array_identity"] = protocol.array_hash(*[arrays[key] for key in sorted(arrays)])
    metadata["data_identity"] = protocol.digest(metadata)
    return arrays, metadata


def save_case(folder, arrays, metadata):
    """Write once; the enclosing runner reserves a fresh case directory."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "data.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    (folder / "metadata.json").write_text(json.dumps(protocol.json_safe(metadata), indent=2,
                                                   allow_nan=False) + "\n")


def load_case(folder, spec, design):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    observation = {"efficiency": design["efficiency"], "held_back_fraction": design["held_back_fraction"]}
    if (metadata["case"] != spec or metadata["configuration"] != design["simulation"]
            or metadata["observation"] != observation):
        raise RuntimeError("cached simulation specification changed")
    with np.load(folder / "data.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    expected = protocol.array_hash(*[arrays[key] for key in sorted(arrays)])
    claimed = metadata.pop("data_identity")
    if expected != metadata["array_identity"] or protocol.digest(metadata) != claimed:
        raise RuntimeError("cached data or physical truth changed")
    metadata["data_identity"] = claimed
    return arrays, metadata


def neural_fitting_data(arrays, spec, design, field_design):
    """Inner selection sees only the outer fitting set, never held-back labels."""
    x, y = arrays["x"][arrays["fitting"]], arrays["y"][arrays["fitting"]].astype(float)
    field_design = deepcopy(field_design)
    field_design["split"]["entropy"] = spec["seeds"]["network"]
    field_design["data"]["box_length"] = design["simulation"]["box"]
    field_design["device"] = design["runtime"]["neural_device"]
    draw = np.random.default_rng(s2.stream_seed(field_design, spec["case_index"],
                                                design["efficiency"], "split")).random(len(x))
    validation = draw < field_design["split"]["validation_fraction"]
    if not validation.any() or validation.all():
        raise RuntimeError("empty inner training or validation set")
    fitting = s2.FittingData(spec["case_index"], design["efficiency"], x[~validation], y[~validation],
                            x[validation], y[validation], x, y, float(y[~validation].mean()),
                            float(y.mean()), protocol.array_hash(validation))
    return fitting, field_design
