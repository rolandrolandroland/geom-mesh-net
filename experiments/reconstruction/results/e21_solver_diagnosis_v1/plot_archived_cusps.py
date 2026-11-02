"""Plot saved guarded probes; this script performs no model evaluations."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    source = folder / "archived_v3_knots.json"
    report = json.loads(source.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.7))
    for ax, particle in zip(axes.flat, report["particles"]):
        crossing = next(c for c in particle["crossings"]
                        if c["assignment_modes"]["fresh"]["coordinate_cusp_minimum"])
        mode = crossing["assignment_modes"]["fresh"]
        points = mode["points"]
        assert mode["verified"] and all(p["valid"] for p in points)
        assert all(p["assignment_changes"] == 0 for p in points)
        xx = [(p["radius"] - crossing["radius"]) * 1e6 for p in points]
        yy = [p["scaled_radius_derivative"] * 1e5 for p in points]
        for indices in ((0, 1), (2, 3)):
            ax.plot([xx[i] for i in indices], [yy[i] for i in indices],
                    "o-", color="#17627c", linewidth=2, markersize=5,
                    label="Guarded derivative probes" if indices == (0, 1) else None)
        endpoint = (particle["endpoint_radius"] - crossing["radius"]) * 1e6
        ax.scatter([endpoint], [particle["raw_radius_gradient"] * 1e5],
                   marker="D", s=33, color="#9b4d93", zorder=4, label="Saved fit endpoint")
        ax.axvline(0, color="#777777", linestyle=":", linewidth=1)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.axhspan(-1, 1, alpha=.12, color="#17627c", label="± declared gradient tolerance")
        ax.set_title(particle["candidate_id"], loc="left", fontsize=11)
        ax.set_xlabel("Radius offset from crossing (model units × 10⁻⁶)", fontsize=9)
        ax.set_ylabel("Mean-NLL derivative in R/5 (× 10⁻⁵)", fontsize=9)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle("Radius derivatives jump at profile and surface joins", x=.07, ha="left", fontsize=15)
    fig.text(.07, .92, "Archived E21 v3 · development case 0 · start 0 · all other parameters fixed", fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3,
               frameon=False, fontsize=9)
    fig.text(.07, .025, "Sign changes support nearby one-coordinate minima; they do not certify joint optimality.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.855, bottom=.20, wspace=.29, hspace=.55)
    output = folder / "archived_radius_cusps.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    provenance = {"scope": "Rendering existing probes only; no new evaluations",
                  "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "output": output.name, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (folder / "archived_radius_cusps_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
