"""Simulate the Stage 5 patterns: separated precipitates in a matrix that obeys a diffusion law.

Each pattern is a clustersim pattern with two changes from the benchmark
(``experiments/reconstruction/ROADMAP.md``, Stage 5 and its correction of 2026-09-16):

1. No two precipitates come closer than ``MIN_GAP``, surface to surface, and no radius
   is below ``R_MIN`` (clustersim's ``min_gap`` and ``r_min``). Every precipitate
   therefore has a whole spherical interface, which the boundary condition needs.
2. Matrix atoms are relabelled. Each is a guest with the probability given by the
   screened diffusion field of ``geom_mesh_net.fields.physics``, instead of the uniform
   rho_b. The field's mean over the pattern's matrix atoms equals its matrix
   concentration. clustersim draws matrix labels last, independently per atom, so
   replacing them leaves the rest of the pattern exactly as clustersim made it.

Precipitates are small and numerous, because the pilot of 2026-09-16 found the physical
constants unrecoverable in patterns with few precipitates. ``PRIOR`` fixes open definition
O8. It was chosen on development patterns, before any test pattern existed, by the
Cramer-Rao bounds of ``pilot/check_diffusion_prior.py`` (ROADMAP Stage 5 correction):

- The supersaturation is set through the critical radius, a fraction of the mean radius,
  so most precipitates grow in depletion zones while the smallest dissolve in enriched
  ones, as in coarsening.
- A higher matrix concentration, a larger capillary length and a wider radius spread make
  the capillary length and screening length recoverable. The radius floor keeps every
  Gibbs-Thomson value finite.

Every pattern can be regenerated on its own. Its parameters, geometry and matrix labels
come from generators seeded by (entropy, pattern index). clustersim draws from a
module-level generator, so that generator is replaced before each pattern, and NumPy's
global generator, used by clustersim's volume estimate, is seeded with the index.

Patterns 0-49 are the development split and 50-149 the test split. Checksums hash array
contents rather than file bytes, since zip archives record write times.

Usage
-----
    python -m experiments.reconstruction.generate_diffusion_patterns --split development
    python -m experiments.reconstruction.generate_diffusion_patterns --split test
    python -m experiments.reconstruction.generate_diffusion_patterns --split development --limit 2 \
        --output-dir /tmp/diffusion_smoke
"""

import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

from geom_mesh_net import paths
from geom_mesh_net.fields import physics
from geom_mesh_net.simulation import clustersim as csim

SIDE, OPP_SIDE, INTENSITY = 60, 30, 1
VOLUME = float(SIDE ** 3)
SPLITS = {"development": (0, 50), "test": (50, 150)}

# Uniform ranges, drawn in this order. Two values follow from them:
#   pcp = matrix_mean + volume_fraction * (rho_c - matrix_mean)
#   supersaturation = exp(ell / (critical_radius_fraction * cr)), so that precipitates of
#   radius critical_radius_fraction * cr neither grow nor dissolve.
PRIOR = {
    "cr": (3.0, 4.0),
    "rb": (0.2, 0.4),
    "volume_fraction": (0.08, 0.12),
    "rho_c": (0.4, 0.9),
    "matrix_mean": (0.05, 0.15),
    "ell": (2.0, 5.0),
    "critical_radius_fraction": (0.6, 0.8),
}
R_MIN = 2.0
MIN_GAP = 1.0
OPP_OVERSAMPLE = 64
ENTROPY = {"parameters": 20260916, "geometry": 20260917, "labels": 20260918}
# Stage 5.2 observes these patterns through benchmark.thinning_mask with this entropy, so its
# masks are independent of the benchmark's. Generation does not use it.
THINNING_ENTROPY = 20260920
FROZEN_DIR = paths.RECONSTRUCTION_DIR / "benchmark" / "diffusion"


def draw_parameters(index):
    generator = np.random.default_rng(np.random.SeedSequence([ENTROPY["parameters"], index]))
    values = {name: float(generator.uniform(low, high)) for name, (low, high) in PRIOR.items()}
    values["pcp"] = values["matrix_mean"] + values["volume_fraction"] * (values["rho_c"] - values["matrix_mean"])
    values["supersaturation"] = float(np.exp(values["ell"] / (values["critical_radius_fraction"] * values["cr"])))
    return values


def content_hash(coords, labels, radii, centres):
    digest = hashlib.sha256()
    for axis in "xyz":
        digest.update(np.ascontiguousarray(coords[axis], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    digest.update(np.ascontiguousarray(radii, dtype=np.float64).tobytes())
    for axis in "xyz":
        digest.update(np.ascontiguousarray(centres[axis], dtype=np.float64).tobytes())
    return digest.hexdigest()


def generate(index, parameters=None):
    """One pattern: returns the arrays to save and a record of what was drawn and realised.

    ``parameters`` overrides the prior draw; ``pilot/compare_diffusion_priors.py`` uses it.
    """
    p = draw_parameters(index) if parameters is None else dict(parameters)
    csim.rng = np.random.default_rng(np.random.SeedSequence([ENTROPY["geometry"], index]))
    np.random.seed(index)  # clustersim's cluster-volume estimate draws from the global generator
    domain = {axis: np.array([0.0, float(SIDE)]) for axis in "xyz"}
    domain_opp = {axis: np.array([0.0, float(OPP_SIDE)]) for axis in "xyz"}
    with contextlib.redirect_stdout(io.StringIO()):
        points, labels = csim.gen_rand_points(intensity=INTENSITY, dim1=SIDE, dim2=SIDE, dim3=SIDE)
        upp = csim.PointPattern3(points, domain=domain, labels=labels)
        opp_points, opp_labels = csim.gen_rand_points(intensity=INTENSITY, dim1=OPP_SIDE, dim2=OPP_SIDE,
                                                      dim3=OPP_SIDE)
        opp = csim.PointPattern3(opp_points, domain=domain_opp, labels=opp_labels)
        pattern, radii, centres = csim.clustersim(
            opp=opp, upp=upp, pcp=p["pcp"], rho_c=p["rho_c"], rho_b=p["matrix_mean"], cr=p["cr"], rb=p["rb"],
            cut="buffered", buffer_factor=1, selection="sampled", prob_function="Gaussian_decay",
            opp_oversample=OPP_OVERSAMPLE, min_gap=MIN_GAP, r_min=R_MIN,
        )
    radii = np.asarray(radii, dtype=float)
    centres = {axis: np.asarray(centres[axis], dtype=float) for axis in "xyz"}
    x = np.column_stack([pattern.coords[axis] for axis in "xyz"])
    c = np.column_stack([centres[axis] for axis in "xyz"])
    matrix = np.isin(pattern.labels, (0, 3))

    field, values = physics.fit_to_matrix_mean(c, radii, x[matrix], ell=p["ell"],
                                               supersaturation=p["supersaturation"],
                                               matrix_mean=p["matrix_mean"], volume=VOLUME)
    clipped = (values < 0) | (values > 1)
    generator = np.random.default_rng(np.random.SeedSequence([ENTROPY["labels"], index]))
    guest = generator.random(int(matrix.sum())) < np.clip(values, 0.0, 1.0)
    new_labels = pattern.labels.copy()
    new_labels[matrix] = np.where(guest, 3, 0).astype(new_labels.dtype)

    distance = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distance, np.inf)
    surface_values = field.surface_values
    record = {
        "pattern": index,
        "parameters": p,
        "precipitates": int(len(radii)),
        "radius_min": float(radii.min()),
        "radius_mean": float(radii.mean()),
        "radius_max": float(radii.max()),
        "smallest_gap": float((distance - radii[:, None] - radii[None, :]).min()),
        "realised_volume_fraction": float(1 - matrix.mean()),
        "matrix_atoms": int(matrix.sum()),
        "matrix_guest_fraction": float(guest.mean()),
        "xi": field.xi,
        "c_eq": field.c_eq,
        "c_inf": field.c_inf,
        "critical_radius": float(field.critical_radius),
        "depleting_precipitates": int((surface_values < field.c_inf).sum()),
        "surface_value_range": [float(surface_values.min()), float(surface_values.max())],
        "field_sd": float(values.std()),
        "field_range": [float(values.min()), float(values.max())],
        "clipped_fraction": float(clipped.mean()),
    }
    arrays = dict(coords=pattern.coords, domain=pattern.domain, labels=new_labels, radii=radii, centers=centres,
                  physics=field.to_dict(), parameters=p)
    return arrays, record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--output-dir", type=Path, default=paths.DIFFUSION_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    protected = {paths.DATA_DIR.resolve(), paths.SHARED_UPP_DIR.resolve(), paths.RANDOM_CENTRES_DIR.resolve()}
    if args.output_dir.resolve() in protected:
        raise SystemExit(f"refusing to write into {args.output_dir}, which holds another dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    low, high = SPLITS[args.split]
    indices = list(range(low, high))[: args.limit]
    records_path = args.output_dir / "patterns.json"
    checksums_path = args.output_dir / "checksums.json"
    records = {r["pattern"]: r for r in json.loads(records_path.read_text())} if records_path.exists() else {}
    checksums = json.loads(checksums_path.read_text()) if checksums_path.exists() else {}

    started, failures = time.perf_counter(), []
    for n, index in enumerate(indices, 1):
        begun = time.perf_counter()
        try:
            arrays, record = generate(index)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            failures.append((index, f"{type(exc).__name__}: {exc}"))
            print(f"  pattern {index} failed: {failures[-1][1]}", flush=True)
            continue
        record["split"] = args.split
        record["seconds"] = round(time.perf_counter() - begun, 1)
        np.savez(args.output_dir / f"clust_pattern_{index}.npz", **arrays)
        checksums[str(index)] = content_hash(arrays["coords"], arrays["labels"], arrays["radii"], arrays["centers"])
        records[index] = record
        print(f"  {n}/{len(indices)} pattern {index}: {record['precipitates']} precipitates, "
              f"xi {record['xi']:.2f}, clipped {record['clipped_fraction']:.4f}, {record['seconds']} s", flush=True)

    records_path.write_text(json.dumps([records[k] for k in sorted(records)], indent=1) + "\n")
    checksums_path.write_text(json.dumps(dict(sorted(checksums.items(), key=lambda kv: int(kv[0]))), indent=1) + "\n")
    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance.update({
        "generated_by": "experiments/reconstruction/generate_diffusion_patterns.py",
        "design": "clustersim with min_gap and r_min; matrix relabelled from geom_mesh_net.fields.physics",
        "box_side": SIDE, "prior": PRIOR, "r_min": R_MIN, "min_gap": MIN_GAP, "opp_oversample": OPP_OVERSAMPLE,
        "entropy": ENTROPY, "splits": SPLITS, "checksum": "sha256 of coords, labels, radii and centres arrays",
    })
    provenance.setdefault("runs", []).append({
        "split": args.split, "patterns": len(indices), "prior": PRIOR,
        "failed": len(failures), "failures": failures[:20],
        "seconds": round(time.perf_counter() - started, 1),
    })
    provenance_path.write_text(json.dumps(provenance, indent=1) + "\n")

    # Freeze the dataset's description where it is tracked, as for the benchmark, so a
    # regenerated dataset can be told apart from the one results were scored on.
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    for source in (records_path, checksums_path, provenance_path):
        (FROZEN_DIR / f"dataset_{source.name}").write_text(source.read_text())
    print(f"wrote {len(indices) - len(failures)} patterns to {args.output_dir} ({len(failures)} failed) "
          f"in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
