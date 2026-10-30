"""Does the diffusion field as first written in ROADMAP Stage 5.1 obey its own boundary condition?

Stage 5.1 first specified the matrix field as a sum of single-precipitate profiles,

    naive    c = c_inf + sum_k (c_k - c_inf) g_k(x),     g_k(x) = (R_k / r_k) exp(-(r_k - R_k) / xi),

with c_k the Gibbs-Thomson value of precipitate k. Each g_k solves the screened diffusion
equation, but the neighbours' tails shift every surface value away from c_k. The
alternative solves the amplitudes so that every surface mean is exact,

    solved   c = c_inf + sum_k a_k g_k(x),    a_k + sum_{j != k} a_j g_j(x_k) sinh(R_k/xi)/(R_k/xi) = c_k - c_inf,

using the mean-value property of the equation, which holds while no centre lies inside
another sphere. Where one does, the mean is averaged numerically instead.

On the benchmark geometry of development patterns 0-49, for capillary lengths 0.75, 1.5
and 3 and two regimes (critical radius at the mean radius, "coarsening"; or c_eq half
the matrix mean, "growth"), this measures: boundary-condition error at surface points;
the share of matrix atoms outside [0, 1]; the expected log-likelihood signal of the field
over a constant matrix; and the Cramer-Rao bound on log ell, log xi and log c_eq for a fit
that knows the true geometry. Recorded in ROADMAP Stage 5, correction of 2026-09-16.

Usage
-----
    python -m experiments.reconstruction.pilot.check_diffusion_field
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.reconstruction import benchmark as bm
from geom_mesh_net import paths

R_MIN = 1.0
BOX = 60.0
VOLUME = BOX ** 3
N_SURFACE = 64
FD = 1e-3
CHUNK = 8000
# (ell, regime, matrix mean). None means the pattern's own rho_b.
CONFIGS = [(ell, regime, None) for ell in (0.75, 1.5, 3.0) for regime in ("coarsening", "growth")]
CONFIGS.append((3.0, "growth", 0.10))


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def kernel_apply(points, centres, radii, xi, columns):
    out = np.empty((len(points), columns.shape[1]))
    c2 = (centres ** 2).sum(1)
    for s in range(0, len(points), CHUNK):
        x = points[s:s + CHUNK]
        r = np.sqrt(np.maximum((x ** 2).sum(1)[:, None] + c2[None, :] - 2 * x @ centres.T, 1e-24))
        out[s:s + CHUNK] = ((radii[None, :] / r) * np.exp(-(r - radii[None, :]) / xi)) @ columns
    return out


def coupling(centres, radii, xi, distances):
    d = distances.copy()
    np.fill_diagonal(d, 1.0)
    m = (radii[None, :] / d) * np.exp(-(d - radii[None, :]) / xi) * (np.sinh(radii / xi) / (radii / xi))[:, None]
    np.fill_diagonal(m, 1.0)
    pts = fibonacci_sphere(400)
    for k, j in np.argwhere((distances <= radii[:, None]) & ~np.eye(len(radii), dtype=bool)):
        r = np.linalg.norm(centres[k] + radii[k] * pts - centres[j], axis=1)
        m[k, j] = np.mean((radii[j] / np.maximum(r, 1e-12)) * np.exp(-(r - radii[j]) / xi))
    return m


def analyse(index):
    started = time.perf_counter()
    theta = bm.load_theta()[index]
    rho_b, cr, rb = float(theta[1]), float(theta[2]), float(theta[3])
    pattern = bm.load_pattern(index)
    _, n_spheres = bm.load_oracle(index)
    matrix_x = pattern["coords"][n_spheres == 0]
    all_radii = pattern["radii"]
    keep = all_radii >= R_MIN
    centres, radii = pattern["centres"][keep], all_radii[keep]
    k = len(radii)
    row = {"pattern": index, "rho_b": rho_b, "cr": cr, "rb": rb, "clusters": int(k),
           "matrix_fraction": float(len(matrix_x) / len(n_spheres))}
    if k < 2:
        row["skipped"] = "fewer than two clusters"
        return row

    xi = (4 * np.pi * all_radii.sum() / VOLUME) ** -0.5
    dist = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1))
    off = ~np.eye(k, dtype=bool)
    overlapping = (dist < radii[:, None] + radii[None, :]) & off
    isolated = ~overlapping.any(axis=1)
    row.update(xi=float(xi), rbar=float(radii.mean()), radius_cv=float(radii.std() / radii.mean()),
               overlapping_pair_fraction=float(overlapping.sum() / max(off.sum(), 1)),
               spheres_overlapping_another=float(1 - isolated.mean()),
               centre_inside_other_sphere_pairs=int(((dist <= radii[:, None]) & off).sum()))

    unit = fibonacci_sphere(N_SURFACE)
    surf_all = (centres[:, None, :] + (radii[:, None, None] + 1e-6) * unit[None, :, :]).reshape(-1, 3)
    owner_all = np.repeat(np.arange(k), N_SURFACE)
    d_surf = np.sqrt(((surf_all[:, None, :] - centres[None, :, :]) ** 2).sum(-1))
    exposed = (np.all((surf_all >= 0) & (surf_all <= BOX), axis=1)
               & np.all((d_surf > radii[None, :]) | (np.arange(k)[None, :] == owner_all[:, None]), axis=1))
    row["exposed_surface_fraction"] = float(exposed.mean())

    xis = {0: xi, 1: xi * np.exp(FD), -1: xi * np.exp(-FD)}
    Ms = {key: coupling(centres, radii, x, dist) for key, x in xis.items()}
    row["coupling_condition_number"] = float(np.linalg.cond(Ms[0]))
    ells = sorted({c[0] for c in CONFIGS})
    unit_cols = np.column_stack([np.linalg.solve(Ms[0], np.ones(k))]
                                + [np.linalg.solve(Ms[0], np.exp(e / radii)) for e in ells])
    unit_mean = kernel_apply(matrix_x, centres, radii, xi, unit_cols).mean(axis=0)
    mean_ones = unit_mean[0]
    mean_gt = {e: unit_mean[1 + i] for i, e in enumerate(ells)}

    def amps(ell, c_eq, c_inf, key):
        return np.linalg.solve(Ms[key], c_eq * np.exp(ell / radii) - c_inf)

    settings, columns = [], {0: [], 1: [], -1: []}
    for ell, regime, mean in CONFIGS:
        m = rho_b if mean is None else mean
        if regime == "coarsening":
            c_eq = m / (np.exp(ell / radii.mean()) * (1 - mean_ones) + mean_gt[ell])
            c_inf = c_eq * np.exp(ell / radii.mean())
        else:
            c_eq = 0.5 * m
            c_inf = (m - c_eq * mean_gt[ell]) / (1 - mean_ones)
        c_vec = c_eq * np.exp(ell / radii)
        s = {"ell": ell, "regime": regime, "matrix_mean": float(m), "c_eq": float(c_eq), "c_inf": float(c_inf),
             "amplitude_median": float(np.median(np.abs(c_vec - c_inf)))}
        if not (m > 0 and 0 < c_eq and 0 < c_inf < 1 and np.all(c_vec < 1)):
            s["invalid"] = True
            settings.append((s, None))
            continue
        base = len(columns[0])
        columns[0] += [amps(ell, c_eq, c_inf, 0), amps(ell * np.exp(FD), c_eq, c_inf, 0),
                       amps(ell * np.exp(-FD), c_eq, c_inf, 0), amps(ell, c_eq * np.exp(FD), c_inf, 0),
                       amps(ell, c_eq * np.exp(-FD), c_inf, 0), amps(ell, c_eq, c_inf * np.exp(FD), 0),
                       amps(ell, c_eq, c_inf * np.exp(-FD), 0), c_vec - c_inf]
        base_xi = len(columns[1])
        columns[1].append(amps(ell, c_eq, c_inf, 1))
        columns[-1].append(amps(ell, c_eq, c_inf, -1))
        settings.append((s, (base, base_xi, c_eq, c_inf, c_vec)))

    f0 = kernel_apply(matrix_x, centres, radii, xis[0], np.column_stack(columns[0])) if columns[0] else None
    fp = kernel_apply(matrix_x, centres, radii, xis[1], np.column_stack(columns[1])) if columns[1] else None
    fm = kernel_apply(matrix_x, centres, radii, xis[-1], np.column_stack(columns[-1])) if columns[-1] else None
    bc_cols = []
    for s, info in settings:
        if info is not None:
            bc_cols += [columns[0][info[0]], columns[0][info[0] + 7]]
    fs = kernel_apply(surf_all, centres, radii, xi, np.column_stack(bc_cols)) if bc_cols else None

    results, bc_i = [], 0
    for s, info in settings:
        if info is None:
            results.append(s)
            continue
        base, base_xi, c_eq, c_inf, c_vec = info
        fd = lambda j: f0[:, base + j]  # noqa: E731
        c = c_inf + fd(0)
        naive = c_inf + fd(7)
        jac = np.column_stack([
            (fd(1) - fd(2)) / (2 * FD),
            ((c_inf + fp[:, base_xi]) - (c_inf + fm[:, base_xi])) / (2 * FD),
            (fd(3) - fd(4)) / (2 * FD),
            ((c_inf * np.exp(FD) + fd(5)) - (c_inf * np.exp(-FD) + fd(6))) / (2 * FD),
        ])
        ok = (c > 1e-6) & (c < 1 - 1e-6)
        info_matrix = (jac[ok] / (c[ok] * (1 - c[ok]))[:, None]).T @ jac[ok]
        mc = c[ok].mean()
        kl = float((c[ok] * np.log(c[ok] / mc) + (1 - c[ok]) * np.log((1 - c[ok]) / (1 - mc))).sum())
        for eta in (0.37, 0.1):
            s[f"signal_nats_eta{eta}"] = eta * kl
            try:
                cov = np.linalg.inv(eta * info_matrix)
                s[f"crb_rel_ell_eta{eta}"] = float(np.sqrt(cov[0, 0]))
                s[f"crb_rel_xi_eta{eta}"] = float(np.sqrt(cov[1, 1]))
                s[f"crb_rel_ceq_eta{eta}"] = float(np.sqrt(cov[2, 2]))
            except np.linalg.LinAlgError:
                pass
        s["field_sd"] = float(c.std())
        s["solved_matrix_mean"] = float(c.mean())
        s["naive_matrix_mean"] = float(naive.mean())
        s["solved_clipped_fraction"] = float(np.mean((c < 0) | (c > 1)))
        s["naive_clipped_fraction"] = float(np.mean((naive < 0) | (naive > 1)))
        amp = s["amplitude_median"]
        for name, col in (("solved", bc_i), ("naive", bc_i + 1)):
            dev = c_inf + fs[:, col] - c_vec[owner_all]
            mean_all = np.bincount(owner_all, weights=dev, minlength=k) / N_SURFACE
            e = exposed
            mean_exposed = (np.bincount(owner_all[e], weights=dev[e], minlength=k)
                            / np.maximum(np.bincount(owner_all[e], minlength=k), 1))
            iso = isolated[owner_all] & e
            s[f"{name}_bc_full_sphere_mean_rel"] = float(np.median(np.abs(mean_all)) / amp)
            s[f"{name}_bc_exposed_sphere_mean_rel"] = float(np.median(np.abs(mean_exposed[np.bincount(owner_all[e], minlength=k) > 0])) / amp)
            s[f"{name}_bc_pointwise_rel_median"] = float(np.median(np.abs(dev[e])) / amp)
            s[f"{name}_bc_pointwise_rel_median_isolated"] = (float(np.median(np.abs(dev[iso])) / amp) if iso.any() else None)
        bc_i += 2
        results.append(s)
    row["configs"] = results
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def summarise(rows):
    """Per-setting counts and medians over patterns with at least two clusters, as quoted in the ROADMAP."""
    used = [r for r in rows if "configs" in r]
    out = {
        "patterns_analysed": len(used),
        "clusters_quartiles": np.quantile([r["clusters"] for r in used], [0.25, 0.5, 0.75]).tolist(),
        "patterns_with_fewer_than_10_clusters": int(sum(r["clusters"] < 10 for r in used)),
        "share_of_spheres_overlapping_another_median": float(np.median([r["spheres_overlapping_another"] for r in used])),
        "patterns_with_a_centre_inside_another_sphere": int(sum(r["centre_inside_other_sphere_pairs"] > 0 for r in used)),
        "settings": [],
    }
    for ell, regime, mean in CONFIGS:
        items = [(r, c) for r in used for c in r["configs"]
                 if c["ell"] == ell and c["regime"] == regime
                 and (abs(c["matrix_mean"] - r["rho_b"]) < 1e-12 if mean is None else c["matrix_mean"] == mean)]
        valid = [c for _, c in items if not c.get("invalid")]
        deep = [c for r, c in items if not c.get("invalid") and r["centre_inside_other_sphere_pairs"] > 0]
        shallow = [c for r, c in items if not c.get("invalid") and r["centre_inside_other_sphere_pairs"] == 0]
        median = lambda key, cs=valid: float(np.median([c[key] for c in cs if c.get(key) is not None]))
        out["settings"].append({
            "ell": ell, "regime": regime, "matrix_mean": "own rho_b" if mean is None else mean, "patterns": len(valid),
            "naive_clipped_over_0.1pct": int(sum(c["naive_clipped_fraction"] > 0.001 for c in valid)),
            "solved_clipped_over_0.1pct": int(sum(c["solved_clipped_fraction"] > 0.001 for c in valid)),
            "solved_clipped_over_0.1pct_with_a_centre_inside_another_sphere": [int(sum(c["solved_clipped_fraction"] > 0.001 for c in deep)), len(deep)],
            "solved_clipped_over_0.1pct_without": [int(sum(c["solved_clipped_fraction"] > 0.001 for c in shallow)), len(shallow)],
            "naive_bc_exposed_sphere_mean_rel_median": median("naive_bc_exposed_sphere_mean_rel"),
            "solved_bc_exposed_sphere_mean_rel_median": median("solved_bc_exposed_sphere_mean_rel"),
            "naive_bc_pointwise_rel_median_isolated": median("naive_bc_pointwise_rel_median_isolated"),
            "solved_bc_pointwise_rel_median_isolated": median("solved_bc_pointwise_rel_median_isolated"),
            "signal_over_10_nats_eta0.37": int(sum(c["signal_nats_eta0.37"] > 10 for c in valid)),
            "crb_rel_ell_eta0.37_below_0.25": int(sum((c.get("crb_rel_ell_eta0.37") or np.inf) < 0.25 for c in valid)),
            "crb_rel_xi_eta0.37_below_0.25": int(sum((c.get("crb_rel_xi_eta0.37") or np.inf) < 0.25 for c in valid)),
            "crb_rel_ell_eta0.37_median": median("crb_rel_ell_eta0.37"),
            "crb_rel_xi_eta0.37_median": median("crb_rel_xi_eta0.37"),
        })
    return out


def rounded(value):
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", type=int, default=50, help="development patterns 0..N-1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "diffusion_field.json")
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(analyse, range(args.patterns)), key=lambda r: r["pattern"])
    kept = ("ell", "regime", "matrix_mean", "invalid", "c_eq", "c_inf", "amplitude_median", "signal_nats_eta0.37",
            "crb_rel_ell_eta0.37", "crb_rel_xi_eta0.37", "naive_clipped_fraction", "solved_clipped_fraction",
            "naive_bc_exposed_sphere_mean_rel", "solved_bc_exposed_sphere_mean_rel", "solved_bc_full_sphere_mean_rel")
    compact = [{**{k: v for k, v in r.items() if k != "configs"},
                "configs": [{k: c[k] for k in kept if k in c} for c in r.get("configs", [])]} for r in rows]
    report = {"summary": summarise(rows), "per_pattern": compact}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rounded(report), indent=1) + "\n")
    print(json.dumps(rounded(report["summary"]), indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
