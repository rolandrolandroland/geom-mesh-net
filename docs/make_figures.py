"""Generate the figures used by the experiment walkthroughs in ``docs/``.

Every figure is drawn from a committed results file so the documentation can be
rebuilt without rerunning any experiment. Where a figure needs the raw feature
cache or posterior samples -- both gitignored because they are large and
regenerable -- the script says so and skips that panel rather than failing.

    PYTHONPATH=. python docs/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference.recover_ground_truth import PARAMETER_NAMES

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

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


# --------------------------------------------------------------------------
# E2: the k_r_max scan that fixed the K features
# --------------------------------------------------------------------------

def figure_k_radius_scan():
    """Interior-extremum rate against k_r_max, both transforms.

    Measured on 16 patterns by truncating a single fine-grid K curve, which is
    exact: K(r) is a cumulative sum over pairs and does not depend on the grid
    maximum.
    """
    radii = [10, 15, 20, 25, 30, 35, 40, 50]
    sqrt_rm = [5, 8, 9, 11, 13, 15, 16, 16]
    sqrt_rdm = [4, 5, 9, 11, 12, 12, 12, 12]
    sqrt_rddm = [3, 4, 4, 4, 6, 8, 8, 12]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for values, label, colour, marker in (
        (sqrt_rm, "Rm", ACCENT, "o"),
        (sqrt_rdm, "Rdm", PASS, "s"),
        (sqrt_rddm, "Rddm", OPEN, "^"),
    ):
        left.plot(radii, [v / 16 for v in values], marker=marker, color=colour,
                  label=label, lw=1.6, ms=5)
    left.axvline(40, color=INK, ls=":", lw=1.2)
    left.text(40.6, 0.13, "adopted\nk_r_max = 40", fontsize=8, color=INK)
    left.axhline(0.8, color=FAINT, ls="--", lw=1)
    left.text(10.5, 0.82, "Stage 1 gate on Rm", fontsize=7.5, color=FAINT)
    left.set_xlabel("k_r_max  (domain side = 60)")
    left.set_ylabel("fraction with an interior extremum")
    left.set_ylim(0, 1.05)
    left.set_title("Extrema recovered vs. K radius (sqrt transform)", fontsize=10)
    left.legend(frameon=False, fontsize=8)

    # Transform comparison at the adopted radius, over all 1000 patterns.
    names = ["Rm", "Rdm", "Rddm"]
    sqrt_rate = [0.964, 0.713, 0.366]
    cube_rate = [1.000, 0.753, 0.466]
    x = np.arange(3)
    right.bar(x - 0.19, sqrt_rate, 0.36, label="sqrt (adopted)", color=ACCENT)
    right.bar(x + 0.19, cube_rate, 0.36, label="cube_root", color=PASS)
    for position, (a, b) in enumerate(zip(sqrt_rate, cube_rate)):
        right.text(position - 0.19, a + 0.02, f"{a:.3f}", ha="center", fontsize=7.5)
        right.text(position + 0.19, b + 0.02, f"{b:.3f}", ha="center", fontsize=7.5)
    right.set_xticks(x)
    right.set_xticklabels(names)
    right.set_ylim(0, 1.12)
    right.set_ylabel("fraction interior, all 1000 patterns")
    right.set_title("Transform comparison at k_r_max = 40", fontsize=10)
    right.legend(frameon=False, fontsize=8, loc="upper right")

    save(fig, "e2_k_radius_scan.png")


# --------------------------------------------------------------------------
# E2: which feature carries which parameter
# --------------------------------------------------------------------------

def figure_feature_parameter_map():
    path = Path("inference/features/global_features.npz")
    if not path.exists():
        print("  skipping feature map: run inference/extract_features.py first")
        return
    with np.load(path, allow_pickle=False) as cached:
        values = cached["values"]
        names = [str(n) for n in cached["feature_names"]]
    theta = np.load("inference/ground_truth/theta.npy")[: len(values)]

    matrix = np.array([
        [np.corrcoef(values[:, j], theta[:, k])[0, 1] for k in range(4)]
        for j in range(len(names))
    ])

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    ax.grid(False)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(PARAMETER_NAMES)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for j in range(len(names)):
        for k in range(4):
            value = matrix[j, k]
            ax.text(k, j, f"{value:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(value) > 0.55 else INK)
    # Bracket the four summary-function families.
    for start, stop, label in ((0, 4, "G"), (4, 6, "F"), (6, 11, "K"), (11, 14, "cross-G")):
        ax.add_patch(plt.Rectangle((-0.5, start - 0.5), 4, stop - start,
                                   fill=False, edgecolor=INK, lw=1.6))
        ax.text(3.62, (start + stop - 1) / 2, label, fontsize=9, color=INK,
                va="center", rotation=270)
    ax.set_title("Correlation of each feature with each parameter", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.55, label="Pearson r")
    save(fig, "e2_feature_parameter_map.png")


# --------------------------------------------------------------------------
# E3: training history
# --------------------------------------------------------------------------

def figure_training_history():
    path = Path("inference/posterior/training_history.csv")
    if not path.exists():
        print("  skipping training history: run inference/fit_posterior.py first")
        return
    rows = [line.split(",") for line in path.read_text().splitlines()[1:]]
    epoch = [int(r[0]) for r in rows]
    train = [float(r[1]) for r in rows]
    validation = [float(r[2]) for r in rows]
    metadata = json.load(open("inference/posterior/fit_metadata.json"))
    prior = metadata["prior_log_prob"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(epoch, train, color=ACCENT, lw=1.5, label="training")
    ax.plot(epoch, validation, color=PASS, lw=1.5, label="validation")
    ax.axhline(prior, color=OPEN, ls="--", lw=1.3,
               label=f"uniform prior ({prior:.2f} nats)")
    best = int(np.argmax(validation))
    ax.plot(epoch[best], validation[best], "o", color=PASS, ms=7,
            markerfacecolor="white", markeredgewidth=1.8)
    ax.annotate(f"best {validation[best]:.2f}",
                (epoch[best], validation[best]), textcoords="offset points",
                xytext=(8, -14), fontsize=8, color=PASS)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean log-density (nats)")
    ax.set_title("Stage 2: the flow beating the prior", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "e3_training_history.png")


# --------------------------------------------------------------------------
# E5: ablation
# --------------------------------------------------------------------------

def figure_ablation():
    path = Path("inference/posterior/ablation_sqrt.json")
    if not path.exists():
        print("  skipping ablation: run inference/ablate_features.py first")
        return
    data = json.load(open(path))
    results, families = data["results"], data["families"]
    base = results["all 14 features"]["contraction_mean"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    order = list(families)
    x = np.arange(len(PARAMETER_NAMES))
    width = 0.2
    palette = [ACCENT, PASS, OPEN, "#7a4a8c"]
    for index, family in enumerate(order):
        entry = results[f"{family} family only"]["contraction_mean"]
        left.bar(x + (index - 1.5) * width,
                 [entry[n] for n in PARAMETER_NAMES], width,
                 label=f"{family} only", color=palette[index])
    left.axhline(0, color=FAINT, lw=0.8)
    left.set_xticks(x)
    left.set_xticklabels(PARAMETER_NAMES)
    left.set_ylabel("posterior contraction")
    left.set_title("What each summary function alone recovers", fontsize=10)
    left.legend(frameon=False, fontsize=8, ncol=2)

    for index, family in enumerate(order):
        entry = results[f"drop {family} family"]["contraction_mean"]
        right.bar(x + (index - 1.5) * width,
                  [entry[n] - base[n] for n in PARAMETER_NAMES], width,
                  label=f"drop {family}", color=palette[index])
    right.axhline(0, color=FAINT, lw=0.8)
    right.set_xticks(x)
    right.set_xticklabels(PARAMETER_NAMES)
    right.set_ylabel("change in contraction")
    right.set_title("Cost of removing each family", fontsize=10)
    right.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    right.set_ylim(-0.28, 0.06)

    save(fig, "e5_ablation.png")


# --------------------------------------------------------------------------
# E6: observation noise
# --------------------------------------------------------------------------

def figure_augmentation():
    path = Path("inference/features/augmented_features.json")
    if not path.exists():
        print("  skipping augmentation: run inference/augment_features.py first")
        return
    data = json.load(open(path))
    ratios = data["within_over_between_sd"]
    order = sorted(ratios, key=ratios.get, reverse=True)

    comparison = json.load(open("inference/posterior/augmentation_comparison.json"))
    control = comparison["control"]["parameters"]
    augmented = comparison["augmented"]["parameters"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    left.barh(range(len(order)), [ratios[n] for n in order], color=ACCENT)
    left.set_yticks(range(len(order)))
    left.set_yticklabels(order, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("within-pattern sd / between-pattern sd")
    left.set_title(f"Observation noise at {data['retention']:.0%} retention",
                   fontsize=10)
    left.axvline(1.0, color=OPEN, ls="--", lw=1)
    left.text(0.30, len(order) - 1.4,
              "features barely move between\nobservations of one structure",
              fontsize=8, color=FAINT)

    x = np.arange(len(PARAMETER_NAMES))
    left_vals = [control[n]["coverage_90"] for n in PARAMETER_NAMES]
    right_vals = [augmented[n]["coverage_90"] for n in PARAMETER_NAMES]
    right.bar(x - 0.19, left_vals, 0.36, label="one observation", color=FAINT)
    right.bar(x + 0.19, right_vals, 0.36, label="four observations", color=ACCENT)
    right.axhline(0.9, color=OPEN, ls="--", lw=1.2, label="nominal 0.90")
    right.set_xticks(x)
    right.set_xticklabels(PARAMETER_NAMES)
    right.set_ylim(0.80, 0.95)
    right.set_ylabel("90% coverage")
    right.set_title("Effect of augmentation on coverage", fontsize=10)
    right.legend(frameon=False, fontsize=8, loc="lower right")

    save(fig, "e6_augmentation.png")


# --------------------------------------------------------------------------
# E7: where rho_b's calibration fails
# --------------------------------------------------------------------------

def figure_rho_b():
    path = Path("inference/posterior/calibration.npz")
    if not path.exists():
        print("  skipping rho_b: run inference/validate_posterior.py first")
        return
    with np.load(path, allow_pickle=False) as cached:
        ranks = cached["ranks"]
        draws = int(cached["n_posterior_samples"])
    theta = np.load("inference/ground_truth/theta.npy")[: len(ranks)]
    position = PARAMETER_NAMES.index("rho_b")
    normalized = (ranks[:, position] + 0.5) / (draws + 1)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.7))

    left.hist(normalized, bins=25, color=ACCENT, edgecolor="white", linewidth=0.6)
    left.axhline(len(normalized) / 25, color=OPEN, ls="--", lw=1.4,
                 label="uniform expectation")
    left.set_xlabel("normalized SBC rank")
    left.set_ylabel("patterns")
    left.set_title(f"rho_b rank histogram (mean {normalized.mean():.3f}, "
                   f"uniform 0.500)", fontsize=10)
    left.legend(frameon=False, fontsize=8)

    edges = np.quantile(theta[:, position], np.linspace(0, 1, 6))
    centres, means = [], []
    for index in range(5):
        mask = ((theta[:, position] >= edges[index])
                & (theta[:, position] <= edges[index + 1]))
        centres.append((edges[index] + edges[index + 1]) / 2)
        means.append(normalized[mask].mean())
    right.plot(centres, means, "o-", color=ACCENT, lw=1.8, ms=7)
    right.axhline(0.5, color=OPEN, ls="--", lw=1.4, label="calibrated")
    # Binomial-style band on a mean of 200 uniform draws.
    band = np.sqrt(1 / 12 / 200)
    right.fill_between([edges[0], edges[-1]], 0.5 - 2 * band, 0.5 + 2 * band,
                       color=RULE, alpha=0.6, label="±2 se")
    right.set_xlabel("true rho_b")
    right.set_ylabel("mean normalized rank")
    right.set_title("The bias is local, not uniform", fontsize=10)
    right.legend(frameon=False, fontsize=8)

    save(fig, "e7_rho_b.png")


if __name__ == "__main__":
    print("Generating walkthrough figures")
    figure_k_radius_scan()
    figure_feature_parameter_map()
    figure_training_history()
    figure_ablation()
    figure_augmentation()
    figure_rho_b()
    print("done")
