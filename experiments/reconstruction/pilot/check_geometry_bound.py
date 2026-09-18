"""What can the capillary length be worth when the precipitates are not known either?

Gate 5.2(ii) compares a fit that must find its own precipitates against the Cramer-Rao bound
of ``pilot/check_diffusion_prior.py``, which assumes the true centres and radii
(``experiments/reconstruction/ROADMAP.md``, Stage 5). That bound is unreachable in principle:
nobody has the geometry. This pilot recomputes it with the geometry as unknown nuisance
parameters, so the gate's threshold can be read against something a method could reach.

**The model.** Every atom is a guest with a probability that depends on where it sits:

    p(x) = c(x) + [rho(r_k / R_k) - c(x)] * S((R_k - r_k) / w),

for the nearest precipitate k, its distance r_k, the screened diffusion field c(x) of
``fields/physics.py``, and the logistic S. Two pieces are new:

- ``rho(u)``, the guest probability inside a precipitate as a share of its radius. The
  simulator does not fill precipitates uniformly: it samples guests with a weight that falls
  with distance from the centre, so the oracle's profile saturates near 1 at the centre and
  drops at the rim. It is fitted per pattern as rho(u) = 1 - (1 - rho_edge) u^m.
- ``w``, the width over which the interface is mixed. The simulator's own boundary is a step:
  a guest probability of about 0.5 just inside and the matrix value just outside. A step
  carries unbounded information about a radius, so the bound is not defined without a
  measurement model. Real atom probe data mix the interface over a nanometre or two through
  local magnification and trajectory overlap. ``w`` stands for that, and the bound is reported
  across a range of it.

**The bound.** The Fisher information of independent Bernoulli labels at detection efficiency
eta is eta * sum_i grad p_i grad p_i^T / (p_i (1 - p_i)), over *all* atoms: the ones inside
precipitates carry the information about the radii. Derivatives are taken by central
differences in log parameters. The bound on a constant is the square root of the matching
diagonal of the inverse, which in log space is a relative error. Three nuisance sets are
compared:

- **geometry known**: only c_eq, ell, xi and c_inf are unknown (the Stage 5 bound);
- **radii unknown**: plus every precipitate's radius, and the interior shape. ``--interior-knots``
  replaces the two-parameter profile with a piecewise-linear one through that many segments,
  which is what the joint fit of ``fields/joint_fit.py`` actually frees;
- **radii and centres unknown**: plus every centre, three parameters each.

Usage
-----
    python -m experiments.reconstruction.pilot.check_geometry_bound
    python -m experiments.reconstruction.pilot.check_geometry_bound --patterns 0 --widths 0.5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.reconstruction import stage5_simulator
from geom_mesh_net import paths
from geom_mesh_net.fields import physics

PATTERNS = (0, 1, 2, 3, 4, 5, 6, 7)
EFFICIENCIES = (0.37, 0.1)
WIDTHS = (0.5, 1.0, 2.0)          # nm of interface mixing
STEP = 1e-3                       # central-difference step in log parameters
CONSTANTS = ("c_eq", "ell", "xi", "c_inf")
CHUNK = 20000


def nearest_precipitate(x, centres, radii):
    """Distance to each atom's nearest precipitate centre, and that precipitate's index."""
    distance = np.empty(len(x))
    index = np.empty(len(x), dtype=int)
    for start in range(0, len(x), CHUNK):
        chunk = x[start:start + CHUNK]
        gap = np.sqrt(((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)) - radii[None, :]
        index[start:start + CHUNK] = gap.argmin(axis=1)
        distance[start:start + CHUNK] = np.linalg.norm(chunk - centres[index[start:start + CHUNK]], axis=1)
    return distance, index


def interior_shape(u, p_star, inside):
    """Fit rho(u) = 1 - (1 - rho_edge) u^m to the oracle inside precipitates."""
    from scipy.optimize import curve_fit

    sel = inside & (u <= 1.0)
    shape = lambda uu, rho_edge, m: 1 - (1 - rho_edge) * np.clip(uu, 0, 1) ** m  # noqa: E731
    (rho_edge, m), _ = curve_fit(shape, u[sel], p_star[sel], p0=[0.3, 3.0], bounds=([1e-3, 0.5], [0.999, 20.0]),
                                 maxfev=20000)
    predicted = shape(u[sel], rho_edge, m)
    return float(rho_edge), float(m), float(np.sqrt(np.mean((predicted - p_star[sel]) ** 2)))


def interior_levels(u, p_star, inside, knots, minimum_atoms=20):
    """The oracle's interior profile at ``knots + 1`` equally spaced fractional radii.

    Each atom counts toward its nearest knot. A sphere holds almost nothing near its centre, so
    the innermost knots can go unmeasured; those take the value of the nearest knot that is.
    """
    sel = inside & (u <= 1.0)
    nearest = np.clip(np.round(u[sel] * knots).astype(int), 0, knots)
    levels = np.full(knots + 1, np.nan)
    for j in range(knots + 1):
        take = nearest == j
        if take.sum() >= minimum_atoms:
            levels[j] = float(np.median(p_star[sel][take]))
    measured = np.flatnonzero(~np.isnan(levels))
    return np.interp(np.arange(knots + 1), measured, levels[measured])


def interior_profile(u, shape):
    """``rho(u)``: the two-parameter power law, or a piecewise-linear profile through its knots."""
    u = np.clip(u, 0, 1)
    if len(shape) == 2:
        rho_edge, m = shape
        return 1 - (1 - rho_edge) * u ** m
    levels = np.asarray(shape, dtype=float)
    return np.interp(u, np.linspace(0, 1, len(levels)), levels)


def probabilities(theta, geometry, x, distance, index, width, shape):
    """The model's guest probability at every atom, for parameters in log space."""
    c_eq, ell, xi, c_inf = np.exp(theta)
    centres, radii = geometry
    field = physics.solve_field(centres, radii, xi, c_eq, ell, c_inf)
    matrix = field(x)
    interior = interior_profile(distance / radii[index], shape)
    blend = 1 / (1 + np.exp((distance - radii[index]) / width))
    return np.clip(matrix + (interior - matrix) * blend, 1e-9, 1 - 1e-9)


def jacobian(theta, centres, radii, shape, x, width, free_centres):
    """Central differences of p with respect to every parameter, in log space (or nm for centres)."""
    names, columns = [], []
    distance, index = nearest_precipitate(x, centres, radii)
    base = dict(x=x, width=width)

    def evaluate(t, c, r, s):
        d, i = (distance, index) if c is centres else nearest_precipitate(x, c, r)
        return probabilities(t, (c, r), x=base["x"], distance=d, index=i, width=base["width"], shape=s)

    def difference(up, down, step):
        return ((up - down) / (2 * step)).astype(np.float32)

    for position, name in enumerate(CONSTANTS):
        step = STEP
        up, down = theta.copy(), theta.copy()
        up[position] += step
        down[position] -= step
        columns.append(difference(evaluate(up, centres, radii, shape), evaluate(down, centres, radii, shape), step))
        names.append(name)
    for k in range(len(radii)):
        step = STEP
        up, down = radii.copy(), radii.copy()
        up[k] *= np.exp(step)
        down[k] *= np.exp(-step)
        columns.append(difference(evaluate(theta, centres, up, shape), evaluate(theta, centres, down, shape), step))
        names.append(f"radius_{k}")
    shape_names = ("rho_edge", "m") if len(shape) == 2 else tuple(f"level_{j}" for j in range(len(shape)))
    for position, name in enumerate(shape_names):
        step = STEP
        up, down = list(shape), list(shape)
        up[position] *= np.exp(step)
        down[position] *= np.exp(-step)
        columns.append(difference(evaluate(theta, centres, radii, tuple(up)),
                                  evaluate(theta, centres, radii, tuple(down)), step))
        names.append(name)
    if free_centres:
        for k in range(len(radii)):
            for axis in range(3):
                step = 1e-3 * radii[k]
                up, down = centres.copy(), centres.copy()
                up[k, axis] += step
                down[k, axis] -= step
                columns.append(difference(evaluate(theta, up, radii, shape), evaluate(theta, down, radii, shape), step))
                names.append(f"centre_{k}_{axis}")
    return names, np.column_stack(columns)


def bounds_from(names, jac, p, efficiency):
    """Relative bounds on the constants, with the geometry known and with it unknown."""
    weight = efficiency / (p * (1 - p))
    information = (jac * weight[:, None].astype(np.float32)).T @ jac
    information = np.asarray(information, dtype=float)
    out = {}
    blocks = {"geometry_known": [names.index(k) for k in CONSTANTS],
              "radii_unknown": [i for i, n in enumerate(names) if not n.startswith("centre_")],
              "radii_and_centres_unknown": list(range(len(names)))}
    has_centres = any(n.startswith("centre_") for n in names)
    for label, keep in blocks.items():
        if label == "radii_and_centres_unknown" and not has_centres:
            continue
        block = information[np.ix_(keep, keep)]
        try:
            inverse = np.linalg.inv(block + np.eye(len(block)) * 1e-12 * np.trace(block) / len(block))
        except np.linalg.LinAlgError:
            out[label] = {k: float("inf") for k in CONSTANTS}
            continue
        out[label] = {k: float(np.sqrt(max(inverse[keep.index(names.index(k)), keep.index(names.index(k))], 0.0)))
                      for k in CONSTANTS}
    return out


def analyse(index, widths, efficiencies, free_centres, subsample=None, seed=0, knots=0):
    with np.load(paths.DIFFUSION_DIR / f"clust_pattern_{index}.npz", allow_pickle=True) as d:
        coords, labels = d["coords"].item(), d["labels"]
        truth, parameters = d["physics"].item(), d["parameters"].item()
    x = np.column_stack([coords[a] for a in "xyz"])
    field = physics.DiffusionField.from_dict(truth)
    p_star, inside = stage5_simulator.load_oracle(index)
    scale = 1.0
    if subsample and subsample < len(x):
        # The information is a sum over atoms, so a random subset scaled by 1/share is unbiased.
        keep = np.random.default_rng(np.random.SeedSequence([seed, index])).choice(len(x), subsample, replace=False)
        scale = len(x) / subsample
        x, p_star, inside = x[keep], p_star[keep], inside[keep]
    distance, nearest = nearest_precipitate(x, field.centres, field.radii)
    rho_edge, m, residual = interior_shape(distance / field.radii[nearest], p_star, inside)
    shape = (rho_edge, m) if not knots else tuple(interior_levels(distance / field.radii[nearest],
                                                                 p_star, inside, knots))
    theta = np.log([field.c_eq, field.ell, field.xi, field.c_inf])
    row = {"pattern": index, "precipitates": int(len(field.radii)),
           "radius_mean": float(field.radii.mean()), "radius_spread": float(field.radii.std() / field.radii.mean()),
           "rb": parameters["rb"], "ell": field.ell, "xi": field.xi,
           "interior_shape": {"rho_edge": rho_edge, "exponent": m, "residual_rms": residual,
                              "knots": knots or None, "levels": list(shape) if knots else None},
           "atoms_used": int(len(x)), "atom_scale": scale, "widths": {}}
    for width in widths:
        started = time.perf_counter()
        names, jac = jacobian(theta, field.centres, field.radii, shape, x, width, free_centres)
        p = probabilities(theta, (field.centres, field.radii), x, distance, nearest, width, shape)
        # Each retained atom stands for ``scale`` of them, so the efficiency is scaled up, not down.
        row["widths"][str(width)] = {str(eta): bounds_from(names, jac, p, eta * scale) for eta in efficiencies}
        row["widths"][str(width)]["seconds"] = round(time.perf_counter() - started, 1)
        row["widths"][str(width)]["parameters"] = len(names)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", default=",".join(map(str, PATTERNS)))
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    parser.add_argument("--efficiencies", default=",".join(map(str, EFFICIENCIES)))
    parser.add_argument("--free-centres", action="store_true", help="also treat every centre as unknown")
    parser.add_argument("--interior-knots", type=int, default=0,
                        help="price a piecewise-linear interior with this many segments instead of the power law")
    parser.add_argument("--subsample", type=int, default=None,
                        help="use this many atoms and scale the information up; the bound is a property of the "
                             "pattern, not of its labels, so a subset only costs precision")
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "geometry_bound.json")
    args = parser.parse_args()
    patterns = [int(i) for i in args.patterns.split(",")]
    # Test patterns are allowed: the bound is computed from the stored geometry and constants and
    # from atom positions, never from a label, exactly as the oracle is.
    widths = [float(w) for w in args.widths.split(",")]
    efficiencies = [float(e) for e in args.efficiencies.split(",")]

    rows, started = [], time.perf_counter()
    earlier = {}
    if args.output.exists():   # earlier runs are kept: a run adds its widths to the patterns it recomputes
        for r in json.loads(args.output.read_text()).get("per_pattern", []):
            earlier[r["pattern"]] = r
            if r["pattern"] not in patterns:
                rows.append(r)
    for index in patterns:
        row = analyse(index, widths, efficiencies, args.free_centres, args.subsample, knots=args.interior_knots)
        row["widths"] = {**earlier.get(index, {}).get("widths", {}), **row["widths"]}
        rows.append(row)
        for width in widths:
            for eta in efficiencies:
                b = row["widths"][str(width)][str(eta)]
                known, unknown = b["geometry_known"], b["radii_unknown"]
                extra = b.get("radii_and_centres_unknown")
                print(f"pattern {index} w {width} eta {eta}: ell bound {known['ell']:.3f} known, "
                      f"{unknown['ell']:.3f} radii unknown" + (f", {extra['ell']:.3f} with centres" if extra else "")
                      + f" | xi {known['xi']:.3f} -> {unknown['xi']:.3f}", flush=True)
        rows.sort(key=lambda r: r["pattern"])
        args.output.write_text(json.dumps({"design": {"patterns": sorted({r["pattern"] for r in rows}),
                                                      "widths": sorted({float(w) for r in rows for w in r["widths"]}),
                                                      "efficiencies": efficiencies, "step": STEP,
                                                      "free_centres": bool(args.free_centres),
                                                      "subsample": args.subsample,
                                                      "interior_knots": args.interior_knots or None},
                                           "per_pattern": rows,
                                           "seconds": round(time.perf_counter() - started)}, indent=1) + "\n")
    print(f"wrote {args.output} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
