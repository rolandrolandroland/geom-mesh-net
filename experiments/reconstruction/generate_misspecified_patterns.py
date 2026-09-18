"""Relabel Stage 5 patterns with matrix fields that break the diffusion law (Gate 5.2(iii) controls).

Each misspecified pattern is a Stage 5 pattern from ``generate_diffusion_patterns.py`` with
only its matrix labels redrawn, so it keeps the precipitates, their labels and the replay
oracle inside them. Three matrix fields replace the diffusion field
(``geom_mesh_net.fields.misspecified``), each with the pattern's own geometry and matrix mean:

- ``inverse_square``: (R/r)^2 profiles whose surface means keep the Gibbs-Thomson values of
  the pattern's constants and supersaturation. The boundary condition holds; the equation
  does not. The best screened-diffusion fit comes within a few nats of this field over a
  whole pattern, so no held-out comparison can reject it; it is kept as a record of that
  finding, not as a Gate 5.2 control.
- ``shuffled_surface``: the screened diffusion field with the pattern's constants and
  screening length, but with each precipitate given the Gibbs-Thomson value of another's
  radius, by a random permutation. The equation holds; the capillarity law does not.
- ``smoothed_noise``: white noise smoothed by a Gaussian of ``NOISE_LENGTH``, with the mean
  and standard deviation of the pattern's diffusion field over its matrix atoms. Neither
  the equation nor the boundary condition holds.

Files hold the full label array and the field's value at every matrix atom, in atom order,
which is the matrix part of the oracle. Coordinates stay in the base pattern.

Usage
-----
    python -m experiments.reconstruction.generate_misspecified_patterns --kind inverse_square --split development
    python -m experiments.reconstruction.generate_misspecified_patterns --kind smoothed_noise --split development
    python -m experiments.reconstruction.generate_misspecified_patterns --kind shuffled_surface --indices 50,51
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.reconstruction.generate_diffusion_patterns import FROZEN_DIR, SPLITS
from geom_mesh_net import paths
from geom_mesh_net.fields import misspecified

KINDS = ("inverse_square", "smoothed_noise", "shuffled_surface")
ENTROPY = {"inverse_square": 20260921, "smoothed_noise": 20260922, "noise_field": 20260923,
           "shuffled_surface": 20260925, "permutation": 20260926}
NOISE_LENGTH = 3.0
MISSPECIFIED_DIR = paths.DIFFUSION_DIR / "misspecified"


def pattern_path(kind, index, root=MISSPECIFIED_DIR):
    return Path(root) / kind / f"clust_pattern_{index}.npz"


def generate(kind, index, records, data_dir=paths.DIFFUSION_DIR):
    with np.load(Path(data_dir) / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        coords, labels = d["coords"].item(), d["labels"]
        truth, parameters = d["physics"].item(), d["parameters"].item()
    x = np.column_stack([coords[axis] for axis in "xyz"])
    matrix = np.isin(labels, (0, 3))
    if kind == "inverse_square":
        field, values = misspecified.inverse_square_to_matrix_mean(
            np.asarray(truth["centres"]), np.asarray(truth["radii"]), x[matrix], ell=parameters["ell"],
            supersaturation=parameters["supersaturation"], matrix_mean=parameters["matrix_mean"])
    elif kind == "shuffled_surface":
        seed = int(np.random.SeedSequence([ENTROPY["permutation"], index]).generate_state(1)[0])
        field, values = misspecified.shuffled_surface_to_matrix_mean(
            np.asarray(truth["centres"]), np.asarray(truth["radii"]), x[matrix], ell=parameters["ell"],
            supersaturation=parameters["supersaturation"], matrix_mean=parameters["matrix_mean"], volume=None,
            seed=seed, xi=truth["xi"])
    elif kind == "smoothed_noise":
        seed = int(np.random.SeedSequence([ENTROPY["noise_field"], index]).generate_state(1)[0])
        field, values = misspecified.smoothed_noise_to_matrix(seed, NOISE_LENGTH, x[matrix], mean=parameters["matrix_mean"],
                                                              sd=records[index]["field_sd"])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    clipped = (values < 0) | (values > 1)
    generator = np.random.default_rng(np.random.SeedSequence([ENTROPY[kind], index]))
    guest = generator.random(int(matrix.sum())) < np.clip(values, 0.0, 1.0)
    new_labels = labels.copy()
    new_labels[matrix] = np.where(guest, 3, 0).astype(labels.dtype)
    q = np.clip(values, 1e-12, 1 - 1e-12)
    mean = q.mean()
    record = {
        "pattern": index, "kind": kind,
        "matrix_atoms": int(matrix.sum()),
        "matrix_guest_fraction": float(guest.mean()),
        "field_mean": float(values.mean()), "field_sd": float(values.std()),
        "diffusion_field_sd": float(records[index]["field_sd"]),
        "field_range": [float(values.min()), float(values.max())],
        "clipped_fraction": float(clipped.mean()),
        "signal_nats_all_atoms": float(np.sum(q * np.log(q / mean) + (1 - q) * np.log((1 - q) / (1 - mean)))),
    }
    arrays = dict(labels=new_labels, matrix_values=values, field=field.to_dict())
    return arrays, record


def content_hash(labels, values):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels).tobytes())
    digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def load(kind, index, root=MISSPECIFIED_DIR):
    """(labels, matrix values, field description) of one misspecified pattern."""
    with np.load(pattern_path(kind, index, root), allow_pickle=True) as d:
        return d["labels"], d["matrix_values"], d["field"].item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--split", choices=tuple(SPLITS), default=None)
    parser.add_argument("--indices", default=None, help="comma-separated pattern indices, instead of a split")
    parser.add_argument("--output-dir", type=Path, default=MISSPECIFIED_DIR)
    args = parser.parse_args()
    if (args.split is None) == (args.indices is None):
        raise SystemExit("give exactly one of --split and --indices")
    indices = list(range(*SPLITS[args.split])) if args.split else [int(i) for i in args.indices.split(",")]

    protected = {paths.DATA_DIR.resolve(), paths.SHARED_UPP_DIR.resolve(), paths.RANDOM_CENTRES_DIR.resolve()}
    if args.output_dir.resolve() in protected:
        raise SystemExit(f"refusing to write into {args.output_dir}, which holds another dataset")
    out = args.output_dir / args.kind
    out.mkdir(parents=True, exist_ok=True)
    base_records = {r["pattern"]: r for r in json.loads((paths.DIFFUSION_DIR / "patterns.json").read_text())}
    records_path, checksums_path = out / "patterns.json", out / "checksums.json"
    records = {r["pattern"]: r for r in json.loads(records_path.read_text())} if records_path.exists() else {}
    checksums = json.loads(checksums_path.read_text()) if checksums_path.exists() else {}

    started = time.perf_counter()
    for n, index in enumerate(indices, 1):
        arrays, record = generate(args.kind, index, base_records)
        np.savez_compressed(pattern_path(args.kind, index, args.output_dir), **arrays)
        checksums[str(index)] = content_hash(arrays["labels"], arrays["matrix_values"])
        records[index] = record
        print(f"  {n}/{len(indices)} {args.kind} pattern {index}: sd {record['field_sd']:.4f} "
              f"(diffusion {record['diffusion_field_sd']:.4f}), clipped {record['clipped_fraction']:.4f}", flush=True)

    records_path.write_text(json.dumps([records[k] for k in sorted(records)], indent=1) + "\n")
    checksums_path.write_text(json.dumps(dict(sorted(checksums.items(), key=lambda kv: int(kv[0]))), indent=1) + "\n")
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance.update({"generated_by": "experiments/reconstruction/generate_misspecified_patterns.py", "kind": args.kind,
                       "base_dataset": "data_diffusion (generate_diffusion_patterns.py)", "entropy": ENTROPY,
                       "noise_length": NOISE_LENGTH, "checksum": "sha256 of labels and matrix_values arrays"})
    provenance.setdefault("runs", []).append({"indices": [indices[0], indices[-1]], "patterns": len(indices),
                                              "seconds": round(time.perf_counter() - started, 1)})
    provenance_path.write_text(json.dumps(provenance, indent=1) + "\n")
    if args.output_dir.resolve() == MISSPECIFIED_DIR.resolve():  # a smoke run elsewhere freezes nothing
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        for source in (records_path, checksums_path, provenance_path):
            (FROZEN_DIR / f"misspecified_{args.kind}_{source.name}").write_text(source.read_text())
    print(f"wrote {len(indices)} {args.kind} patterns to {out} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
