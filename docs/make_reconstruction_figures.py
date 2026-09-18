"""Generate the figures used by the reconstruction walkthroughs (E10 onward) in ``docs/``.

Every figure is drawn from a tracked results file under
``experiments/reconstruction/``, so the documentation can be rebuilt without
rerunning any experiment. A figure whose
results file is missing is skipped with a message.

    PYTHONPATH=. python docs/make_reconstruction_figures.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

from geom_mesh_net import paths

OUT = paths.FIGURES_DIR
OUT.mkdir(parents=True, exist_ok=True)
PILOT = paths.RECONSTRUCTION_DIR / "pilot" / "results"
RESULTS = paths.RECONSTRUCTION_DIR / "results"

INK = "#12161f"
ACCENT = "#2d5d7c"
PASS = "#2e7d5b"
OPEN = "#b07419"
FAINT = "#6b7488"
RULE = "#d9dee6"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": FAINT,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": FAINT,
    "ytick.color": FAINT,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


def load(path):
    if not path.exists():
        print(f"  skipped: {path} not found")
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# E10: the yardstick
# --------------------------------------------------------------------------


def figure_e10_grid_profile():
    data = load(PILOT / "density_grid.json")
    if data is None:
        return
    profiles = data["single_cluster"]["profiles"]
    fig, axes = plt.subplots(1, len(profiles), figsize=(9, 2.8), sharey=True)
    for ax, profile in zip(axes, profiles):
        mid = [(b["u_lo"] + b["u_hi"]) / 2 for b in profile["bins"]]
        ax.plot(mid, [b["simulator"] for b in profile["bins"]], "-o", color=PASS, ms=3.5, lw=1.6,
                label="simulator (replayed)")
        ax.plot(mid, [b["grid"] for b in profile["bins"]], "--s", color=OPEN, ms=3.5, lw=1.4,
                label="generate_density_grid")
        ax.set_title(f"rho_c = {profile['rho_c']}   (RMSE {profile['rmse']:.3f})", fontsize=9)
        ax.set_xlabel("distance from centre / radius")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("guest probability")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "e10_grid_profile.png")


def figure_e10_reliability():
    grid = load(PILOT / "density_grid.json")
    gate = load(RESULTS / "stage0_gate.json")
    if grid is None or gate is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    g = grid["region_check"]["reliability"]
    ax.plot([b["grid"] for b in g], [b["observed"] - b["grid"] for b in g], "-s", color=OPEN, ms=4,
            label="generate_density_grid (60 patterns of data/)")
    for name, colour, marker in (("benchmark", PASS, "o"), ("lattice", ACCENT, "^")):
        bins = gate["datasets"][name]["reliability"]
        ax.plot([b["oracle"] for b in bins], [b["gap"] for b in bins], "-" + marker, color=colour, ms=4,
                label=f"replay oracle ({name}, development patterns)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhspan(-0.005, 0.005, color=PASS, alpha=0.08, lw=0)
    ax.set_xlabel("predicted guest probability (bin mean)")
    ax.set_ylabel("observed − predicted")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "e10_reliability.png")


def figure_e10_lattice():
    data = load(PILOT / "lattice.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    example = data["example_centres"]["10"]
    for ax, key, title in ((axes[0], "lattice", "data/: pattern 10"), (axes[1], "random", "data_random_centres/: pattern 10")):
        c = np.array(example[key])
        ax.scatter(c[:, 0], c[:, 1], s=9, color=ACCENT if key == "lattice" else PASS, alpha=0.7, lw=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y (centres projected along z)")
    n_lat = np.array(data["n_clusters_lattice"])
    n_ran = np.array(data["n_clusters_random"])
    lost = np.array(data["lost_shell"]["indices"])
    ax = axes[2]
    ax.scatter(n_ran, n_lat, s=7, color=FAINT, alpha=0.5, lw=0, label="all patterns")
    ax.scatter(n_ran[lost], n_lat[lost], s=16, color=OPEN, lw=0, label=f"lost outer shell ({len(lost)})")
    lim = [1, max(n_ran.max(), n_lat.max()) * 1.2]
    ax.plot(lim, lim, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clusters intended (random centres)")
    ax.set_ylabel("clusters received (lattice)")
    ax.set_title("cluster counts", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "e10_lattice.png")


# --------------------------------------------------------------------------
# E11: baselines and the headroom map
# --------------------------------------------------------------------------


def stage1_cells():
    data = load(RESULTS / "stage1_headroom.json")
    if data is None:
        return None, None
    return data, [c for r in data["results"] for c in r["cells"]]


def figure_e11_headroom_map():
    data, cells = stage1_cells()
    if data is None:
        return
    etas = data["efficiencies"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    band_labels = ["cr 3–6", "cr 6–10", "cr 10–15"]
    rng = np.random.default_rng(0)
    for ax, eta in zip(axes, etas):
        for band in range(3):
            sel = [c for c in cells if c["eta"] == eta and c["cr_band"] == band]
            closed = np.array([c["gap_closed"]["B1"] for c in sel])
            head = np.array([c["headroom"] for c in sel])
            x = band + rng.uniform(-0.22, 0.22, size=len(sel))
            ax.scatter(x[~head], closed[~head], s=11, color=FAINT, alpha=0.6, lw=0)
            ax.scatter(x[head], closed[head], s=13, color=OPEN, lw=0)
            ax.hlines(np.median(closed), band - 0.3, band + 0.3, color=INK, lw=1.4)
        ax.axhline(0.85, color=INK, lw=0.7, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels(band_labels)
        ax.set_title(f"η = {eta}  ({np.mean([c['headroom'] for c in cells if c['eta'] == eta]):.0%} with headroom)", fontsize=9)
    axes[0].set_ylabel("share of gap closed by B1")
    axes[0].set_ylim(0, 1.02)
    save(fig, "e11_headroom_map.png")


def figure_e11_regions():
    data, cells = stage1_cells()
    if data is None:
        return
    regions = [r for r in data["regions"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, method, colour in ((axes[0], "B1", ACCENT), (axes[1], "B2", PASS)):
        width = 0.25
        for j, eta in enumerate(data["efficiencies"]):
            values = [np.median([c["excess_by_region"][method][r] for c in cells
                                 if c["eta"] == eta and r in c["excess_by_region"][method]]) for r in regions]
            ax.bar(np.arange(len(regions)) + (j - 1) * width, values, width=width * 0.92,
                   color=colour, alpha=0.45 + 0.25 * j, label=f"η = {eta}")
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_title(method, fontsize=9)
    axes[0].set_ylabel("median excess log loss (nats/atom)")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "e11_excess_by_region.png")


def figure_e11_b1_vs_b2():
    data, cells = stage1_cells()
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    colours = {0.1: OPEN, 0.37: ACCENT, 0.8: PASS}
    for eta, colour in colours.items():
        sel = [c for c in cells if c["eta"] == eta]
        ax.scatter([c["excess"]["B1"] for c in sel], [c["excess"]["B2"] for c in sel], s=9, color=colour,
                   alpha=0.75, lw=0, label=f"η = {eta}")
    lim = [min(min(c["excess"]["B1"], c["excess"]["B2"]) for c in cells) * 0.8,
           max(max(c["excess"]["B1"], c["excess"]["B2"]) for c in cells) * 1.2]
    ax.plot(lim, lim, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("B1 excess log loss")
    ax.set_ylabel("B2 excess log loss")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "e11_b1_vs_b2.png")


def figure_e11_bandwidth():
    data, cells = stage1_cells()
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    offsets = {0.1: -0.06, 0.37: 0.0, 0.8: 0.06}
    colours = {0.1: OPEN, 0.37: ACCENT, 0.8: PASS}
    for eta in data["efficiencies"]:
        sel = [c for c in cells if c["eta"] == eta]
        ax.scatter([c["cr"] for c in sel], np.array([c["bandwidth"] for c in sel]) * (1 + offsets[eta]), s=10,
                   color=colours[eta], alpha=0.75, lw=0, label=f"η = {eta}")
    ax.set_yscale("log")
    ax.set_yticks(data["bandwidths"])
    ax.set_yticklabels([str(h) for h in data["bandwidths"]])
    ax.set_xlabel("mean cluster radius cr")
    ax.set_ylabel("B1 bandwidth chosen by CV")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "e11_bandwidth.png")


def figure_e11_predictive_check():
    data, _ = stage1_cells()
    if data is None or "predictive_check_median_abs_z" not in data["summary"]:
        return
    z = data["summary"]["predictive_check_median_abs_z"]
    families = list(next(iter(z.values())).keys())
    methods = list(z.keys())
    colours = {"B0": FAINT, "B1": ACCENT, "B2": PASS, "oracle": INK}
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    width = 0.2
    for j, m in enumerate(methods):
        ax.bar(np.arange(len(families)) + (j - 1.5) * width, [z[m][f] for f in families], width=width * 0.92,
               color=colours.get(m, ACCENT), label=m)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_ylabel("median |z| of relabelled features")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    save(fig, "e11_predictive_check.png")


# --------------------------------------------------------------------------
# E15: a simulator with a diffusion law (Stage 5.1)
# --------------------------------------------------------------------------


def figure_e15_field():
    data = load(RESULTS / "stage5_field_slice.json")
    if data is None:
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    values = np.array([[np.nan if v is None else v for v in row] for row in data["values"]])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.grid(False)
    image = ax.imshow(np.ma.masked_invalid(values), origin="lower", extent=[0, 60, 0, 60], cmap="Blues",
                      interpolation="nearest")
    for circle in data["circles"]:
        enriched = circle["surface_value"] > data["c_inf"]
        ax.add_patch(Circle((circle["x"], circle["y"]), circle["r"], facecolor="white",
                            edgecolor=OPEN if enriched else ACCENT, lw=1.4))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"matrix guest probability, plane z = {data['z']:.0f} (development pattern {data['pattern']})",
                 fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="c(x)")
    ax.legend(handles=[Line2D([], [], color=ACCENT, lw=1.4, label="growing precipitate (c_k < c∞)"),
                       Line2D([], [], color=OPEN, lw=1.4, label="dissolving precipitate (c_k > c∞)")],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    ax = axes[1]
    for profile, colour, word in zip(data["profiles"], (ACCENT, OPEN), ("largest", "smallest")):
        ax.plot(profile["distance_from_surface"], profile["shell_mean"], "-o", color=colour, ms=2.8, lw=1.5,
                label=f"{word} precipitate, R = {profile['radius']:.1f}")
    ax.axhline(data["c_inf"], color=FAINT, ls="--", lw=1)
    ax.text(14.8, data["c_inf"] + 0.002, "far field c∞", color=FAINT, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("distance from the precipitate's surface")
    ax.set_ylabel("mean c over the shell")
    ax.set_title(f"ℓ = {data['ell']:.2f}, ξ = {data['xi']:.2f}, critical radius {data['critical_radius']:.2f}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "e15_field.png")


def figure_e15_boundary_condition():
    data = load(PILOT / "diffusion_field.json")
    if data is None:
        return
    settings = [s for s in data["summary"]["settings"] if s["matrix_mean"] == "own rho_b"]
    settings.sort(key=lambda s: (s["regime"], s["ell"]))
    labels = [f"{s['regime']}\nℓ = {s['ell']}" for s in settings]
    x = np.arange(len(settings))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    ax = axes[0]
    ax.semilogy(x - 0.08, [s["naive_bc_exposed_sphere_mean_rel_median"] for s in settings], "s", color=OPEN, ms=6,
                label="profiles added (as first written)")
    ax.semilogy(x + 0.08, [s["solved_bc_exposed_sphere_mean_rel_median"] for s in settings], "o", color=PASS, ms=6,
                label="amplitudes solved")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("surface-mean error / depletion amplitude")
    ax.set_title("boundary condition, median over 47 patterns", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, [s["naive_clipped_over_0.1pct"] for s in settings], width, color=OPEN,
           label="profiles added")
    deep = [s["solved_clipped_over_0.1pct_with_a_centre_inside_another_sphere"][0] for s in settings]
    ax.bar(x + width / 2, deep, width, color=PASS, label="amplitudes solved (every case has a centre inside a sphere)")
    ax.axhline(47, color=FAINT, lw=0.8, ls=":")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("patterns with > 0.1% of atoms clipped")
    ax.set_title("values outside [0, 1], of 47 patterns", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "e15_boundary_condition.png")


def figure_e15_priors():
    data = load(PILOT / "diffusion_priors.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for name, s in data["summary"].items():
        letter = name.split()[0]
        growing = s["depleting_share_median"]
        chosen = "(chosen)" in name
        colour = PASS if chosen else (ACCENT if growing >= 0.5 else (FAINT if growing >= 0.1 else OPEN))
        ax.plot(s["crb_rel_ell_eta0.37_median"], s["crb_rel_xi_eta0.37_median"], "o", color=colour,
                ms=9 if chosen else 6, zorder=3)
        ax.annotate(letter, (s["crb_rel_ell_eta0.37_median"], s["crb_rel_xi_eta0.37_median"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.axvline(0.25, color=FAINT, lw=0.8, ls=":")
    ax.axhline(0.25, color=FAINT, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    percent = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    ax.set_xticks([0.1, 0.15, 0.2, 0.3, 0.45], minor=False)
    ax.set_yticks([0.2, 0.3, 0.45, 0.7, 1.0], minor=False)
    ax.xaxis.set_major_formatter(percent)
    ax.yaxis.set_major_formatter(percent)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("median bound on ℓ, relative (η = 0.37)")
    ax.set_ylabel("median bound on ξ, relative (η = 0.37)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ACCENT, label="most precipitates grow"),
                       Line2D([], [], marker="o", ls="", color=FAINT, label="most precipitates dissolve"),
                       Line2D([], [], marker="o", ls="", color=OPEN, label="every precipitate dissolves"),
                       Line2D([], [], marker="o", ls="", color=PASS, ms=9, label="chosen prior (most grow)")],
              frameon=False, fontsize=8, loc="upper left")
    ax.set_title("nine candidate priors, 12 patterns each", fontsize=9)
    save(fig, "e15_priors.png")


def figure_e15_bounds():
    data = load(PILOT / "diffusion_prior.json")
    if data is None:
        return
    rows = data["per_pattern"]
    groups = [("ℓ", "crb_rel_ell_eta0.37", 0.37), ("ℓ", "crb_rel_ell_eta0.1", 0.1),
              ("ξ", "crb_rel_xi_eta0.37", 0.37), ("ξ", "crb_rel_xi_eta0.1", 0.1)]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    rng = np.random.default_rng(0)
    for i, (symbol, key, eta) in enumerate(groups):
        values = np.array([r[key] for r in rows])
        colour = ACCENT if symbol == "ℓ" else PASS
        ax.plot(i + rng.uniform(-0.18, 0.18, len(values)), values, "o", color=colour, ms=3.2, alpha=0.75)
        ax.plot([i - 0.28, i + 0.28], [np.median(values)] * 2, color=INK, lw=1.6)
    ax.axhline(0.25, color=OPEN, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.25, 0.5, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(groups)), [f"{s}, η = {e}" for s, _, e in groups])
    ax.set_ylabel("Cramér–Rao bound, relative")
    ax.set_title("best-case recovery with the true geometry, 50 development patterns", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "e15_bounds.png")


def figure_e15_gate():
    data = load(RESULTS / "stage5_simulator.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    ax = axes[0]
    bins = data["in_sphere"]["reliability"]
    ax.plot([b["oracle"] for b in bins], [b["gap"] for b in bins], "-o", color=PASS, ms=4)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhspan(-0.005, 0.005, color=PASS, alpha=0.08, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.012, 0.012)
    ax.set_xlabel("replay oracle, bin mean")
    ax.set_ylabel("observed − oracle")
    ax.set_title(f"inside precipitates: slope {data['in_sphere']['slope']:.4f} ± {data['in_sphere']['slope_se']:.4f}",
                 fontsize=9)
    ax = axes[1]
    rows = data["per_pattern"]
    ax.add_patch(plt.Rectangle((-3, -3), 6, 6, facecolor=PASS, alpha=0.08, lw=0))
    ax.plot([r["matrix_fraction_z"] for r in rows], [r["log_likelihood_gain_z"] for r in rows], "o", color=ACCENT,
            ms=4)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("guest fraction, z")
    ax.set_ylabel("log-likelihood gain, z")
    ax.set_title(f"matrix: {data['matrix']['patterns_within_3se']} of {len(rows)} patterns within ±3", fontsize=9)
    save(fig, "e15_gate.png")


# --------------------------------------------------------------------------
# E16: the soft-constraint PINN
# --------------------------------------------------------------------------

E16_HIGHLIGHT = {"scaled residual, lambda 10": (OPEN, "-"), "lambda 1": (ACCENT, "-"), "lambda 0.1": (ACCENT, "--")}
E16_LABELS = {"scaled residual, lambda 10": "ξ²-scaled residual, λ = 10", "lambda 1": "λ = 1", "lambda 0.1": "λ = 0.1"}


def figure_e16_constants():
    data = load(PILOT / "soft_pinn.json")
    if data is None:
        return
    patterns = data["patterns"]
    fig, axes = plt.subplots(2, len(patterns), figsize=(3.2 * len(patterns), 5.0), sharex=True, squeeze=False)
    bounds = {r["pattern"]: r for r in load(PILOT / "diffusion_prior.json")["per_pattern"]}
    for column, row in enumerate(patterns):
        for line, name in enumerate(("ell", "xi")):
            ax = axes[line, column]
            truth = row["truth"][name]
            crb = bounds[row["pattern"]][f"crb_rel_{name}_eta{data['design']['efficiency']}"]
            ax.axhspan(1 - crb, 1 + crb, color=PASS, alpha=0.10, lw=0)
            ax.axhline(1, color=INK, lw=0.8)
            for setting, result in row["settings"].items():
                if setting.startswith("unconstrained"):
                    continue
                steps = [h["step"] for h in result["history"]]
                ratio = [h[name] / truth for h in result["history"]]
                colour, style = E16_HIGHLIGHT.get(setting, (FAINT, "-"))
                highlighted = setting in E16_HIGHLIGHT
                ax.plot(steps, ratio, style, color=colour, lw=1.6 if highlighted else 0.8, alpha=1 if highlighted else 0.6,
                        label=E16_LABELS.get(setting, "other settings"))
            ax.set_yscale("log")
            low, high = ax.get_ylim()
            ticks = [v for v in (0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100) if low <= v <= high]
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if column == 0:
                ax.set_ylabel(("capillary length ℓ" if name == "ell" else "screening length ξ") + ", fitted / true")
            if line == 0:
                ax.set_title(f"pattern {row['pattern']}: {row['precipitates']} precipitates", fontsize=9)
            else:
                ax.set_xlabel("training step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=len(unique), frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, -0.01, "shaded: within the Cramér–Rao bound of a fit that knows the geometry", ha="center",
             color=FAINT, fontsize=8)
    save(fig, "e16_constants.png")


def figure_e16_excess():
    data = load(PILOT / "soft_pinn.json")
    if data is None:
        return
    patterns = data["patterns"]
    rows = list(patterns[0]["settings"]) + ["analytic family, true geometry"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, name in enumerate(rows):
        shares = []
        for row in patterns:
            excess = (row["analytic_family"]["matrix_excess"] if name == "analytic family, true geometry"
                      else row["settings"][name]["matrix_excess"])
            shares.append(excess / row["constant_matrix_excess"])
        colour = PASS if name.startswith("analytic") else ACCENT
        ax.plot(shares, [i] * len(shares), "o", color=colour, ms=4.5, alpha=0.85)
        ax.plot([np.median(shares)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4)
    ax.axvline(1, color=FAINT, lw=1, ls="--")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [r.replace("lambda", "λ") for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("matrix excess loss ÷ the constant matrix's (0 = oracle)")
    ax.set_title(f"development patterns {', '.join(str(r['pattern']) for r in patterns)}, η = "
                 f"{data['design']['efficiency']}; tick: median", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "e16_excess.png")


def figure_e16_memorisation():
    data = load(PILOT / "soft_pinn.json")
    if data is None:
        return
    patterns = data["patterns"]
    fig, axes = plt.subplots(1, len(patterns), figsize=(3.3 * len(patterns), 3.3), sharey=True, squeeze=False)
    for ax, row in zip(axes[0], patterns):
        constant = row["held_out_loss"]["constant"]
        for setting, colour in (("unconstrained, w0 10", ACCENT), ("unconstrained, w0 3", OPEN)):
            history = row["settings"][setting]["history"]
            steps = [h["step"] for h in history]
            label = setting.split(", ")[1].replace("w0", "w₀ =")
            ax.plot(steps, [h["train_bce"] - constant for h in history], "-", color=colour, lw=1.4,
                    label=f"{label}: training atoms")
            ax.plot(steps, [h["valid_bce"] - constant for h in history], "--", color=colour, lw=1.4,
                    label=f"{label}: held-out atoms")
        ax.axhline(0, color=FAINT, lw=1)
        ax.axhline(row["held_out_loss"]["oracle"] - constant, color=PASS, lw=1, ls=":")
        ax.set_xlabel("training step")
        ax.set_title(f"pattern {row['pattern']}", fontsize=9)
    axes[0, 0].set_ylabel("log loss − constant matrix's, nats per atom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.02, "dotted: the true field on the held-out atoms, the best any field can do on average",
             ha="center", fontsize=8, color=FAINT)
    save(fig, "e16_memorisation.png")


def figure_e16_objective():
    data = load(PILOT / "soft_pinn.json")
    if data is None:
        return
    labels, margins, gaps, costs = [], [], [], []
    for row in data["patterns"]:
        for lam, pair in row["fixed_constants"].items():
            labels.append(f"pattern {row['pattern']}, λ = {float(lam):g}")
            margins.append(row["data_margin_per_atom"])
            gaps.append(pair["true"]["objective_last"] - pair["flat"]["objective_last"])
            costs.append(float(lam) * (pair["true"]["last"]["pde"] + pair["true"]["last"]["bc"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(labels))
    for i in y:
        ax.plot([margins[i], max(gaps[i], 1e-6)], [i, i], color=RULE, lw=2, zorder=1)
    ax.plot(margins, y, "o", color=PASS, ms=6, label="data margin: the true field's advantage over a constant", zorder=2)
    ax.plot([max(g, 1e-6) for g in gaps], y, "s", color=OPEN, ms=6,
            label="loss at the true constants − loss at the flat ones", zorder=3)
    ax.plot(costs, y, "|", color=INK, ms=10, mew=1.4, label="penalties at the true constants", zorder=3)
    for i in y:
        ax.annotate(f"{gaps[i] / margins[i]:.0f}× the margin", (max(gaps[i], 1e-6), i), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.5, color=OPEN)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.5, -1.0)
    ax.set_xlabel("nats per training atom")
    ax.set_title("with the constants held fixed, the loss prefers the flat ones", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax.grid(axis="y", visible=False)
    save(fig, "e16_objective.png")


# --------------------------------------------------------------------------
# E17: the law as a hard constraint, Gate 5.2
# --------------------------------------------------------------------------

E17_KINDS = {"diffusion": (ACCENT, "o", "obeys the law"), "smoothed_noise": (OPEN, "s", "smoothed noise"),
             "shuffled_surface": (INK, "^", "shuffled surface values")}


def figure_e17_selection_bias():
    data = load(PILOT / "matrix_domain.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    rng = np.random.default_rng(1)
    groups = []
    for eta in (0.37, 0.1):
        cells = [r for r in data["per_cell"] if r["efficiency"] == eta]
        for rule, label in (("voxel_rule", "voxel rule"), ("own_label_left_out", "own label left out")):
            groups.append((f"{label}\nη = {eta}", [c["methods"]["significance"][rule]["bias_z"] for c in cells], rule))
    for i, (label, values, rule) in enumerate(groups):
        colour = OPEN if rule == "voxel_rule" else ACCENT
        ax.plot(i + rng.uniform(-0.18, 0.18, len(values)), values, "o", color=colour, ms=4, alpha=0.8)
        ax.plot([i - 0.28, i + 0.28], [np.median(values)] * 2, color=INK, lw=1.5)
    ax.axhspan(-3, 3, color=PASS, alpha=0.10, lw=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("admitted guests − expected, z")
    ax.set_title("choosing matrix atoms from their own labels, 16 development patterns", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "e17_selection_bias.png")


def figure_e17_detectability():
    data = load(PILOT / "misspecification.json")
    if data is None:
        return
    kinds = [("inverse_square", "(R/r)² profiles"), ("shuffled_surface", "shuffled surface values"),
             ("smoothed_noise", "smoothed noise")]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    share = data["design"]["held_out_share"]["0.37"]
    for i, (kind, label) in enumerate(kinds):
        gaps = [row[kind]["gap_nats"] for row in data["per_pattern"]]
        ax.plot(gaps, [i] * len(gaps), "o", color=OPEN if kind == "inverse_square" else ACCENT, ms=5, alpha=0.8)
        ax.plot([np.median(gaps)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.5)
    ax.axvline(2 / share, color=FAINT, lw=1, ls="--")
    ax.annotate("2 nats on held-out atoms at η = 0.37", (2 / share, 2.45), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=FAINT, va="center")
    ax.set_xscale("log")
    ax.set_yticks(range(len(kinds)), [label for _, label in kinds])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("divergence from the best diffusion fit, whole pattern, nats")
    ax.set_title("could any comparison see the misspecification? 8 development patterns", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "e17_detectability.png")


def figure_e17_constants(split="test"):
    data = load(RESULTS / f"stage5_physics_fit_{split}.json")
    if data is None:
        return
    correct = [c for c in data["cells"] if c["kind"] == "diffusion" and "error" not in c]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    rng = np.random.default_rng(2)
    for ax, name, symbol in zip(axes, ("ell", "xi"), ("ℓ", "ξ")):
        labels = []
        for i, (eta, geometry) in enumerate(((0.37, "detected"), (0.37, "true"), (0.1, "detected"), (0.1, "true"))):
            cells = [c for c in correct if c["efficiency"] == eta]
            if geometry == "detected":
                values = [c["physics"]["normalised_error"][name] for c in cells if "normalised_error" in c["physics"]]
            else:
                values = [abs(c["true_geometry"]["relative_error"][name]) / c["bound"][name] for c in cells]
            ax.plot(i + rng.uniform(-0.18, 0.18, len(values)), np.maximum(values, 1e-2), "o",
                    color=ACCENT if geometry == "detected" else PASS, ms=4, alpha=0.8)
            ax.plot([i - 0.28, i + 0.28], [np.median(values)] * 2, color=INK, lw=1.5)
            labels.append(f"{geometry}\nη = {eta}")
        ax.axhline(data["design"]["gate"]["normalised_error"], color=OPEN, lw=1, ls="--")
        ax.axhline(0.674, color=FAINT, lw=1, ls=":")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(range(4), labels)
        ax.set_title(f"{symbol}: |relative error| ÷ Cramér–Rao bound", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("normalised error (tick: median)")
    fig.text(0.5, -0.02, "dashed: Gate 5.2(ii) threshold on the median; dotted: an efficient estimator's median",
             ha="center", fontsize=8, color=FAINT)
    save(fig, f"e17_constants{'' if split == 'test' else '_' + split}.png")


def figure_e17_acceptance(split="test"):
    data = load(RESULTS / f"stage5_physics_fit_{split}.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=False, sharey=False)
    for ax, eta in zip(axes, (0.37, 0.1)):
        for kind, (colour, marker, label) in E17_KINDS.items():
            cells = [c for c in data["cells"] if c["kind"] == kind and c["efficiency"] == eta and "error" not in c]
            for c in cells:
                a = c["acceptance"]
                n = c["atoms"]["acceptance"]
                x = (a["physics_loss"] - a["constant_loss"]) * n
                y = (a["physics_loss"] - a["network_loss"]) * n
                ax.plot(x, y, marker, color=colour, ms=6, mfc=colour if not a["at_bound"] else "white", mew=1.2)
            ax.plot([], [], marker, color=colour, ms=6, label=label)
        ax.plot([], [], "o", color=FAINT, mfc="white", ms=6, label="a fitted constant at a bound")
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.axvline(0, color=FAINT, lw=0.8)
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        ax.add_patch(plt.Rectangle((low_x, low_y), -low_x, -low_y, facecolor=PASS, alpha=0.08, lw=0))
        ax.set_xlim(low_x, high_x)
        ax.set_ylim(low_y, high_y)
        ax.set_xlabel("law − constant matrix, held-out nats")
        ax.set_ylabel("law − network, held-out nats")
        ax.set_title(f"η = {eta}", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.5, -0.02, "the law is accepted only in the shaded quadrant, and only if no fitted constant is at a bound "
             "(open markers)", ha="center", fontsize=8, color=FAINT)
    save(fig, f"e17_acceptance{'' if split == 'test' else '_' + split}.png")


def figure_e17_field(split="test"):
    data = load(RESULTS / f"stage5_physics_fit_{split}.json")
    if data is None:
        return
    correct = [c for c in data["cells"] if c["kind"] == "diffusion" and "error" not in c]
    models = [("true_geometry", "law, true geometry", PASS), ("physics", "law, detected geometry", ACCENT),
              ("network", "unconstrained network", OPEN), ("b1", "B1 smoothing", FAINT)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, eta in zip(axes, (0.37, 0.1)):
        cells = [c for c in correct if c["efficiency"] == eta]
        for i, (key, label, colour) in enumerate(models):
            shares = [(c["true_geometry"]["matrix_excess"] if key == "true_geometry" else c["matrix_excess"][key])
                      / c["matrix_excess"]["constant"] for c in cells]
            ax.plot(shares, [i] * len(shares), "o", color=colour, ms=4, alpha=0.8)
            ax.plot([np.median(shares)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.5)
        ax.axvline(1, color=FAINT, lw=1, ls="--")
        ax.axvline(0, color=INK, lw=0.8)
        largest = max(max((c["true_geometry"]["matrix_excess"] if k == "true_geometry" else c["matrix_excess"][k])
                          / c["matrix_excess"]["constant"] for c in cells) for k, _, _ in models)
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.5)
        ax.set_xlim(-0.03, largest * 1.4)
        ax.set_xticks([t for t in (0, 0.1, 0.3, 1, 3, 10, 30, 100) if t <= largest * 1.4])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks(range(len(models)), [m[1] for m in models])
        ax.invert_yaxis()
        ax.set_xlabel("matrix excess loss ÷ the constant matrix's")
        ax.set_title(f"η = {eta}, {len(cells)} cells; dashed: the constant matrix", fontsize=9)
        ax.grid(axis="y", visible=False)
    save(fig, f"e17_field{'' if split == 'test' else '_' + split}.png")


if __name__ == "__main__":
    print("E10")
    figure_e10_grid_profile()
    figure_e10_reliability()
    figure_e10_lattice()
    print("E11")
    figure_e11_headroom_map()
    figure_e11_regions()
    figure_e11_b1_vs_b2()
    figure_e11_bandwidth()
    figure_e11_predictive_check()
    print("E15")
    figure_e15_field()
    figure_e15_boundary_condition()
    figure_e15_priors()
    figure_e15_bounds()
    figure_e15_gate()
    print("E16")
    figure_e16_constants()
    figure_e16_excess()
    figure_e16_memorisation()
    figure_e16_objective()
    print("E17")
    figure_e17_selection_bias()
    figure_e17_detectability()
    figure_e17_constants()
    figure_e17_acceptance()
    figure_e17_field()
