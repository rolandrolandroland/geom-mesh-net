"""Where is the capillary length recoverable at all? A map over radius spread, efficiency and interface width.

The capillary length enters only through the Gibbs-Thomson surface concentration
c_eq exp(ell / R). A population of equal-sized precipitates therefore says nothing about ell: it
fixes one surface value, which c_eq alone can explain. Only a spread of radii separates the two,
so the spread is this map's main axis.

Nothing here needs labels or a fitted model. The Cramer-Rao bound is a property of the geometry,
the field and the atom positions, so the map is computed on geometries drawn directly, with the
model and machinery of ``check_geometry_bound.py``: the screened field outside precipitates, an
interior profile inside, an interface of width w, and every radius an unknown nuisance parameter.

Axes:

- **radius spread**: the standard deviation of the radii over their mean, from nearly uniform
  precipitates to the widest the Stage 5 prior draws;
- **detection efficiency**: the share of atoms observed;
- **interface width**: how far a measurement mixes the boundary.

The volume fraction, the mean radius and the constants are held at the middle of the Stage 5
prior (``generate_diffusion_patterns.PRIOR``), so a row of the map is comparable with the
development patterns.

Usage
-----
    python -m experiments.reconstruction.pilot.check_identifiability_map
    python -m experiments.reconstruction.pilot.check_identifiability_map --spreads 0.1,0.4 --widths 0.3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.reconstruction.pilot.check_geometry_bound import (CONSTANTS, bounds_from, jacobian,
                                                                    nearest_precipitate, probabilities)
from geom_mesh_net import paths
from geom_mesh_net.fields import physics

SIDE = 60.0
VOLUME = SIDE ** 3
DENSITY = 1.0                 # atoms per cubic nanometre, as in the Stage 5 patterns
SPREADS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6)
EFFICIENCIES = (0.37, 0.1)
WIDTHS = (0.3, 1.0)
MEAN_RADIUS = 3.5             # the middle of the prior's cr
VOLUME_FRACTION = 0.10
MATRIX_MEAN = 0.10
ELL = 3.5
CRITICAL_FRACTION = 0.7       # supersaturation = exp(ell / (fraction * mean radius))
R_MIN, MIN_GAP = 2.0, 1.0
INTERIOR = (0.03, 3.8)        # rho_edge and exponent, the median of the development patterns' oracle fits
ATOMS = 60_000                # a subsample; the information is scaled back up


def draw_geometry(spread, seed):
    """Separated spheres with the given radius spread, filling ``VOLUME_FRACTION`` of the box."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(spread * 1000)]))
    target = VOLUME_FRACTION * VOLUME
    centres, radii, filled = [], [], 0.0
    for _ in range(200_000):
        if filled >= target:
            break
        radius = max(R_MIN, rng.normal(MEAN_RADIUS, MEAN_RADIUS * spread))
        centre = rng.uniform(radius, SIDE - radius, size=3)
        if all(np.linalg.norm(centre - c) >= radius + r + MIN_GAP for c, r in zip(centres, radii)):
            centres.append(centre)
            radii.append(radius)
            filled += 4 / 3 * np.pi * radius ** 3
    return np.array(centres), np.array(radii)


def analyse(spread, seed, widths, efficiencies):
    centres, radii = draw_geometry(spread, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed + 1, int(spread * 1000)]))
    x = rng.uniform(0, SIDE, size=(ATOMS, 3))
    scale = DENSITY * VOLUME / ATOMS   # the box holds this many atoms per one drawn here
    supersaturation = float(np.exp(ELL / (CRITICAL_FRACTION * MEAN_RADIUS)))
    outside = x[nearest_precipitate(x, centres, radii)[0] > radii[nearest_precipitate(x, centres, radii)[1]]]
    field, _ = physics.fit_to_matrix_mean(centres, radii, outside, ell=ELL, supersaturation=supersaturation,
                                          matrix_mean=MATRIX_MEAN, volume=VOLUME)
    theta = np.log([field.c_eq, field.ell, field.xi, field.c_inf])
    distance, index = nearest_precipitate(x, centres, radii)
    row = {"radius_spread": spread, "precipitates": int(len(radii)),
           "realised_spread": float(radii.std() / radii.mean()), "radius_range": [float(radii.min()), float(radii.max())],
           "volume_fraction": float(np.sum(4 / 3 * np.pi * radii ** 3) / VOLUME), "xi": field.xi, "c_eq": field.c_eq,
           "atoms": ATOMS, "widths": {}}
    for width in widths:
        names, jac = jacobian(theta, centres, radii, INTERIOR, x, width, False)
        p = probabilities(theta, (centres, radii), x, distance, index, width, INTERIOR)
        row["widths"][str(width)] = {str(eta): bounds_from(names, jac, p, eta * scale) for eta in efficiencies}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spreads", default=",".join(str(s) for s in SPREADS))
    parser.add_argument("--efficiencies", default=",".join(str(e) for e in EFFICIENCIES))
    parser.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    parser.add_argument("--repeats", type=int, default=2, help="geometries per spread")
    parser.add_argument("--output", type=Path,
                        default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "identifiability_map.json")
    args = parser.parse_args()
    spreads = [float(s) for s in args.spreads.split(",")]
    widths = [float(w) for w in args.widths.split(",")]
    efficiencies = [float(e) for e in args.efficiencies.split(",")]

    rows, started = [], time.perf_counter()
    for spread in spreads:
        for repeat in range(args.repeats):
            row = analyse(spread, 1000 + repeat, widths, efficiencies)
            rows.append({**row, "repeat": repeat})
            for width in widths:
                for eta in efficiencies:
                    b = row["widths"][str(width)][str(eta)]
                    print(f"spread {spread} (realised {row['realised_spread']:.2f}, {row['precipitates']} precipitates) "
                          f"w {width} eta {eta}: ell {b['geometry_known']['ell']:.3f} known, "
                          f"{b['radii_unknown']['ell']:.3f} radii unknown | xi "
                          f"{b['geometry_known']['xi']:.3f} -> {b['radii_unknown']['xi']:.3f}", flush=True)
            args.output.write_text(json.dumps({"design": {"spreads": spreads, "widths": widths,
                                                          "efficiencies": efficiencies, "repeats": args.repeats,
                                                          "mean_radius": MEAN_RADIUS, "volume_fraction": VOLUME_FRACTION,
                                                          "ell": ELL, "matrix_mean": MATRIX_MEAN, "interior": INTERIOR,
                                                          "atoms": ATOMS, "density": DENSITY},
                                               "per_geometry": rows,
                                               "seconds": round(time.perf_counter() - started)}, indent=1) + "\n")
    print(f"wrote {args.output} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
