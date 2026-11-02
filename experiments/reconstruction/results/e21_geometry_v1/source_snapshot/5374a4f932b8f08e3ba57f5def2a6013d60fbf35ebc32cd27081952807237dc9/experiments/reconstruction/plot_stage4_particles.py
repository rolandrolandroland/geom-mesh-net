"""Plot every completed confirmation case; no fitting or selection is performed.

Run from the repository root as
``python -m experiments.reconstruction.plot_stage4_particles``.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "experiments/reconstruction/results/stage4_particles_confirmation.json"
DEFAULT_OUTPUT = ROOT / "docs/figures/E20_particle_pilot.png"


def plot(report_path=DEFAULT_REPORT, output=DEFAULT_OUTPUT):
    report = json.loads(Path(report_path).read_text())
    expected = report["provenance"]["design"]["scope"]["confirmation"]
    cells = sorted(report["cells"], key=lambda c: c["pattern"])
    if sorted(c["pattern"] for c in cells) != sorted(expected) or any("error" in c for c in cells):
        raise ValueError("the comparison figure requires every prescribed confirmation case without a run error")
    reference = report["selection"]["reference"]
    names = {"legacy_b1": "Gaussian detector", "b1": "Gaussian + extraction",
             "dbscan": "DBSCAN", "maximum_separation": "Separation connectivity"}
    reference_name = names[reference]
    colors = plt.get_cmap("tab10")(np.arange(len(cells)))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    panels = [
        ("Field prediction error", "Excess log loss (nats / atom) · lower is better",
         ["Gaussian field", "Fourier field"],
         lambda c: [c["field_scores"]["removed"]["methods"][m]["excess_loss"] for m in ("b1", "fourier")]),
        ("Particles recovered", "Recall (%) · higher is better",
         [reference_name, "Fourier detector"],
         lambda c: [100 * c["methods"][m]["metrics"]["recall"] for m in (reference, "fourier")]),
        ("Radius error on shared detections", "Mean absolute relative error (%) · lower is better",
         [reference_name, "Fourier detector"],
         lambda c: [100 * c["common_matched_radius"][m] if c["common_matched_radius"][m] is not None else np.nan
                    for m in ("reference", "fourier")]),
        ("Capillary-length recovery", "Absolute relative error (%) · lower is better",
         [reference_name, "Fourier detector"],
         lambda c: [100 * abs(c["physics"][m]["relative_error"]["ell"])
                    if "relative_error" in c["physics"][m] else np.nan for m in (reference, "fourier")]),
    ]
    invalid = False
    missing = []
    for panel_index, (ax, (title, ylabel, labels, values)) in enumerate(zip(axes.flat, panels)):
        for color, cell in zip(colors, cells):
            pair = values(cell)
            ax.plot([0, 1], pair, color=color, linewidth=1.8, marker="o", markersize=6,
                    label=f"Pattern {cell['pattern']}")
            if not np.isfinite(pair).all():
                missing.append(f"{title}: pattern {cell['pattern']}")
            if panel_index == 3:
                for side, method in enumerate((reference, "fourier")):
                    if not cell["physics"][method]["reliable"] and np.isfinite(pair[side]):
                        invalid = True
                        ax.scatter([side], [pair[side]], marker="x", s=110, linewidth=2.5,
                                   color="black", zorder=5)
        ax.set_title(title, loc="left", weight="bold", pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.15, 1.15)
        ax.grid(axis="y", alpha=0.2)
        ax.margins(y=0.14)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    axes[0, 1].set_ylim(0, 105)
    fig.suptitle("A better field must also find the particles", fontsize=18, weight="bold", y=0.99)
    foot = "Four development-confirmation patterns · 37% detection · approximately 29.6% of original atoms used for fitting"
    if invalid:
        foot += "\nBlack ×: fit failed a validity check; its estimate remains shown."
    if missing:
        foot += "\nUnavailable values: " + "; ".join(missing)
    fig.text(0.5, 0.015, foot, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.07 if invalid or missing else 0.045, 1, 0.96), h_pad=2.5, w_pad=2.8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(plot(args.report, args.output))


if __name__ == "__main__":
    main()
