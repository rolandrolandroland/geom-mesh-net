"""Calculate and visualize global paper features for saved point patterns."""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from geom_mesh_net.core_functions import paper_spatial_features as psf


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "global_paper_feature_validation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern-count", type=int, default=10)
    parser.add_argument("--relabelings", type=int, default=99)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--g-r-max", type=float, default=10.0)
    parser.add_argument("--k-r-max", type=float, default=70.0)
    parser.add_argument("--cross-g-r-max", type=float, default=8.0)
    parser.add_argument("--k-max-points", type=int, default=3000)
    return parser.parse_args()


def validate_curves(result):
    tolerance = 1e-10
    cdf_curves = [
        result.observed.guest_g,
        result.observed.guest_f,
        result.observed.guest_to_host_g,
        result.expected.guest_g,
        result.expected.guest_f,
        result.expected.guest_to_host_g,
    ]
    k_curves = [
        result.observed.guest_k,
        result.expected.guest_k,
    ]
    cdf_finite = all(np.all(np.isfinite(curve)) for curve in cdf_curves)
    cdf_bounded = all(
        np.all((curve >= -tolerance) & (curve <= 1.0 + tolerance))
        for curve in cdf_curves
    )
    cdf_monotonic = all(
        np.all(np.diff(curve) >= -tolerance)
        for curve in cdf_curves
    )
    k_finite = all(np.all(np.isfinite(curve)) for curve in k_curves)
    k_nonnegative = all(
        np.all(curve >= -tolerance)
        for curve in k_curves
    )
    k_monotonic = all(
        np.all(np.diff(curve) >= -tolerance)
        for curve in k_curves
    )
    features_finite = bool(np.all(np.isfinite(result.values)))
    feature_map = dict(zip(result.names, result.values))
    k_peak_at_boundary = bool(
        np.isclose(feature_map["Rm"], result.radii["k"][0])
        or np.isclose(feature_map["Rm"], result.radii["k"][-1])
    )
    k_derivative_at_boundary = bool(
        np.isclose(feature_map["Rdm"], result.radii["k"][0])
        or np.isclose(feature_map["Rdm"], result.radii["k"][-1])
    )
    guest_g_endpoint = float(result.observed.guest_g[-1])
    guest_f_endpoint = float(result.observed.guest_f[-1])
    cross_g_endpoint = float(result.observed.guest_to_host_g[-1])
    cross_g_reaches_95 = cross_g_endpoint >= 0.95
    numerical_checks = [
        cdf_finite,
        cdf_bounded,
        cdf_monotonic,
        k_finite,
        k_nonnegative,
        k_monotonic,
        features_finite,
    ]
    return {
        "cdf_finite": cdf_finite,
        "cdf_bounded": cdf_bounded,
        "cdf_monotonic": cdf_monotonic,
        "k_finite": k_finite,
        "k_nonnegative": k_nonnegative,
        "k_monotonic": k_monotonic,
        "features_finite": features_finite,
        "k_peak_at_boundary": k_peak_at_boundary,
        "k_derivative_at_boundary": k_derivative_at_boundary,
        "guest_g_endpoint": guest_g_endpoint,
        "guest_f_endpoint": guest_f_endpoint,
        "cross_g_endpoint": cross_g_endpoint,
        "cross_g_reaches_95": cross_g_reaches_95,
        "range_status": (
            "PASS"
            if (
                not k_peak_at_boundary
                and not k_derivative_at_boundary
                and cross_g_reaches_95
            )
            else "WARNING"
        ),
        "numerical_status": "PASS" if all(numerical_checks) else "FAIL",
    }


def save_curve_data(output_dir, pattern_id, result):
    np.savez_compressed(
        output_dir / f"pattern_{pattern_id}_summary_curves.npz",
        g_r=result.radii["g"],
        g_observed=result.observed.guest_g,
        g_expected=result.expected.guest_g,
        f_observed=result.observed.guest_f,
        f_expected=result.expected.guest_f,
        k_r=result.radii["k"],
        k_observed=result.observed.guest_k,
        k_expected=result.expected.guest_k,
        cross_g_r=result.radii["cross_g"],
        cross_g_observed=result.observed.guest_to_host_g,
        cross_g_expected=result.expected.guest_to_host_g,
        feature_names=np.asarray(result.names),
        feature_values=result.values,
    )


def plot_pattern_summary(output_dir, pattern_id, result):
    features = dict(zip(result.names, result.values))
    transformed_k = (
        np.sqrt(result.observed.guest_k)
        - np.sqrt(result.expected.guest_k)
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    figure.suptitle(f"Pattern {pattern_id}: Global spatial summaries")

    axes[0, 0].plot(
        result.radii["g"],
        result.observed.guest_g,
        label="Observed",
        linewidth=2,
    )
    axes[0, 0].plot(
        result.radii["g"],
        result.expected.guest_g,
        label="Random-label median",
        color="black",
        linestyle="--",
    )
    axes[0, 0].axvline(
        features["G_max_diff_r"],
        color="tab:red",
        alpha=0.6,
        label="G max-difference radius",
    )
    axes[0, 0].axvline(
        features["G_zero_diff_r"],
        color="tab:green",
        alpha=0.6,
        label="G zero-difference radius",
    )
    axes[0, 0].set_title("Guest nearest-neighbor G")
    axes[0, 0].set_xlabel("Radius")
    axes[0, 0].set_ylabel("CDF")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        result.radii["g"],
        result.observed.guest_f,
        label="Observed",
        linewidth=2,
    )
    axes[0, 1].plot(
        result.radii["g"],
        result.expected.guest_f,
        label="Random-label median",
        color="black",
        linestyle="--",
    )
    axes[0, 1].set_title("Guest empty-space F")
    axes[0, 1].set_xlabel("Radius")
    axes[0, 1].set_ylabel("CDF")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        result.radii["k"],
        transformed_k,
        linewidth=2,
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].scatter(
        [
            features["Rm"],
            features["Rdm"],
            features["Rddm"],
        ],
        [
            np.interp(features["Rm"], result.radii["k"], transformed_k),
            np.interp(features["Rdm"], result.radii["k"], transformed_k),
            np.interp(features["Rddm"], result.radii["k"], transformed_k),
        ],
        color=["tab:red", "tab:orange", "tab:purple"],
        label="Rm, Rdm, Rddm",
        zorder=3,
    )
    axes[1, 0].set_title("Transformed guest K anomaly")
    axes[1, 0].set_xlabel("Radius")
    axes[1, 0].set_ylabel("sqrt(K observed) - sqrt(K expected)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        result.radii["cross_g"],
        result.observed.guest_to_host_g,
        label="Observed",
        linewidth=2,
    )
    axes[1, 1].plot(
        result.radii["cross_g"],
        result.expected.guest_to_host_g,
        label="Random-label median",
        color="black",
        linestyle="--",
    )
    axes[1, 1].axvline(
        features["GXGH_95diff_r"],
        color="tab:red",
        alpha=0.6,
        label="Observed 95% radius",
    )
    axes[1, 1].set_title("Guest-to-host cross G")
    axes[1, 1].set_xlabel("Radius")
    axes[1, 1].set_ylabel("CDF")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.ravel():
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        output_dir / f"pattern_{pattern_id}_summary_functions.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_aggregate_summaries(output_dir, results):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Global summary-function anomalies for patterns 0-9")
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(results)))

    for color, (pattern_id, result) in zip(colors, results.items()):
        axes[0, 0].plot(
            result.radii["g"],
            result.observed.guest_g - result.expected.guest_g,
            color=color,
            label=f"Pattern {pattern_id}",
        )
        axes[0, 1].plot(
            result.radii["g"],
            result.observed.guest_f - result.expected.guest_f,
            color=color,
        )
        axes[1, 0].plot(
            result.radii["k"],
            np.sqrt(result.observed.guest_k)
            - np.sqrt(result.expected.guest_k),
            color=color,
        )
        axes[1, 1].plot(
            result.radii["cross_g"],
            result.observed.guest_to_host_g
            - result.expected.guest_to_host_g,
            color=color,
        )

    titles = [
        "Guest G anomaly",
        "Guest F anomaly",
        "Transformed guest K anomaly",
        "Guest-to-host G anomaly",
    ]
    y_labels = [
        "Observed - expected",
        "Observed - expected",
        "sqrt(K observed) - sqrt(K expected)",
        "Observed - expected",
    ]
    for axis, title, y_label in zip(axes.ravel(), titles, y_labels):
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Radius")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)

    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(1.05, 1.3),
        ncol=5,
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "summary_function_anomalies_10_patterns.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_csv(path, rows, columns):
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_feature_table(path, rows):
    columns = ["pattern"] + list(psf.PAPER_FEATURE_NAMES)
    with path.open("w") as output_file:
        output_file.write("| " + " | ".join(columns) + " |\n")
        output_file.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [str(row["pattern"])]
            values.extend(f"{row[name]:.6g}" for name in psf.PAPER_FEATURE_NAMES)
            output_file.write("| " + " | ".join(values) + " |\n")


def plot_feature_table(output_dir, rows):
    row_labels = [f"Pattern {row['pattern']}" for row in rows]
    cell_text = [
        [f"{row[name]:.4g}" for name in psf.PAPER_FEATURE_NAMES]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(27, 6))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=psf.PAPER_FEATURE_NAMES,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)
    figure.suptitle("Global paper features for patterns 0-9")
    figure.tight_layout()
    figure.savefig(
        output_dir / "global_paper_feature_table.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_feature_heatmap(output_dir, rows):
    values = np.asarray(
        [
            [row[name] for name in psf.PAPER_FEATURE_NAMES]
            for row in rows
        ],
        dtype=float,
    )
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)
    standardized = np.divide(
        values - means,
        standard_deviations,
        out=np.zeros_like(values),
        where=standard_deviations > 0.0,
    )

    figure, axis = plt.subplots(figsize=(16, 7))
    image = axis.imshow(
        standardized,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )
    axis.set_xticks(np.arange(len(psf.PAPER_FEATURE_NAMES)))
    axis.set_xticklabels(
        psf.PAPER_FEATURE_NAMES,
        rotation=55,
        ha="right",
    )
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels([f"Pattern {row['pattern']}" for row in rows])
    axis.set_title("Standardized global paper features")
    figure.colorbar(image, ax=axis, label="Z-score across 10 patterns")
    figure.tight_layout()
    figure.savefig(
        output_dir / "global_paper_feature_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_validation_readme(
    output_dir,
    config,
    pattern_count,
    diagnostic_rows,
):
    numerical_failures = [
        row["pattern"]
        for row in diagnostic_rows
        if row["numerical_status"] != "PASS"
    ]
    range_warnings = [
        row["pattern"]
        for row in diagnostic_rows
        if row["range_status"] != "PASS"
    ]
    with (output_dir / "README.md").open("w") as output_file:
        output_file.write("# Global Paper Feature Validation\n\n")
        output_file.write(
            "This directory contains global spatial-summary features "
            f"calculated from original patterns 0-{pattern_count - 1}.\n\n"
        )
        output_file.write("## Configuration\n\n")
        output_file.write("- Guest labels: `2` and `3`\n")
        output_file.write(
            f"- Random relabelings per pattern: `{config.n_relabelings}`\n"
        )
        output_file.write(
            f"- Guest G and F maximum radius: `{config.g_r_max}`\n"
        )
        output_file.write(
            f"- Guest K maximum radius: `{config.k_r_max}`\n"
        )
        output_file.write(
            "- Guest-to-host G maximum radius: "
            f"`{config.cross_g_r_max}`\n"
        )
        output_file.write(
            "- Maximum guest points used for each K estimate: "
            f"`{config.k_max_points}`\n\n"
        )
        output_file.write(
            "All original points are used for G, F, and guest-to-host G. "
            "The K point cap is a random guest-point sample used to keep "
            "large-radius translation correction tractable.\n\n"
        )
        output_file.write("## Results\n\n")
        output_file.write(
            f"- Numerical-check failures: `{numerical_failures}`\n"
        )
        output_file.write(
            f"- Radius-range warnings: `{range_warnings}`\n\n"
        )
        output_file.write("## Main Artifacts\n\n")
        output_file.write(
            "- `global_paper_features_10_patterns.csv`: feature table\n"
        )
        output_file.write(
            "- `validation_diagnostics.csv`: numerical and range checks\n"
        )
        output_file.write(
            "- `summary_function_anomalies_10_patterns.png`: "
            "aggregate curves\n"
        )
        output_file.write(
            "- `pattern_*_summary_functions.png`: per-pattern curves\n"
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = psf.PaperFeatureConfig(
        g_r_max=args.g_r_max,
        g_num_radii=int(round(args.g_r_max * 200)) + 1,
        k_r_max=args.k_r_max,
        k_num_radii=int(round(args.k_r_max * 20)) + 1,
        cross_g_r_max=args.cross_g_r_max,
        cross_g_num_radii=int(round(args.cross_g_r_max * 333)) + 1,
        n_relabelings=args.relabelings,
        workers=args.workers,
        k_max_points=args.k_max_points,
    )
    with (args.output_dir / "validation_config.json").open("w") as output_file:
        json.dump(asdict(config), output_file, indent=2)

    results = {}
    feature_rows = []
    diagnostic_rows = []

    for pattern_id in range(args.pattern_count):
        data_path = DATA_DIR / f"clust_pattern_{pattern_id}.npz"
        data = np.load(data_path, allow_pickle=True)
        coords = data["coords"].item()
        labels = data["labels"]
        domain = data["domain"].item()
        radii = np.asarray(data["radii"], dtype=float)

        start = time.perf_counter()
        result = psf.calculate_global_paper_features(
            coords,
            labels,
            domain,
            guest_marks=(2, 3),
            config=config,
        )
        elapsed = time.perf_counter() - start
        diagnostics = validate_curves(result)
        guest_count = int(np.isin(labels, (2, 3)).sum())

        feature_row = {
            "pattern": pattern_id,
            "point_count": len(labels),
            "guest_count": guest_count,
            "guest_fraction": guest_count / len(labels),
            "cluster_count": len(radii),
            "mean_cluster_radius": float(np.mean(radii)),
            "sd_cluster_radius": float(np.std(radii)),
            "calculation_seconds": elapsed,
        }
        feature_row.update(dict(zip(result.names, result.values)))
        feature_rows.append(feature_row)

        diagnostic_row = {
            "pattern": pattern_id,
            "calculation_seconds": elapsed,
        }
        diagnostic_row.update(diagnostics)
        diagnostic_rows.append(diagnostic_row)

        results[pattern_id] = result
        save_curve_data(args.output_dir, pattern_id, result)
        plot_pattern_summary(args.output_dir, pattern_id, result)
        print(
            f"Pattern {pattern_id}: {diagnostics['numerical_status']} "
            f"in {elapsed:.2f}s"
        )

    feature_columns = [
        "pattern",
        "point_count",
        "guest_count",
        "guest_fraction",
        "cluster_count",
        "mean_cluster_radius",
        "sd_cluster_radius",
        "calculation_seconds",
        *psf.PAPER_FEATURE_NAMES,
    ]
    diagnostic_columns = [
        "pattern",
        "calculation_seconds",
        "cdf_finite",
        "cdf_bounded",
        "cdf_monotonic",
        "k_finite",
        "k_nonnegative",
        "k_monotonic",
        "features_finite",
        "k_peak_at_boundary",
        "k_derivative_at_boundary",
        "guest_g_endpoint",
        "guest_f_endpoint",
        "cross_g_endpoint",
        "cross_g_reaches_95",
        "range_status",
        "numerical_status",
    ]
    write_csv(
        args.output_dir / "global_paper_features_10_patterns.csv",
        feature_rows,
        feature_columns,
    )
    write_csv(
        args.output_dir / "validation_diagnostics.csv",
        diagnostic_rows,
        diagnostic_columns,
    )
    write_markdown_feature_table(
        args.output_dir / "global_paper_features_10_patterns.md",
        feature_rows,
    )
    plot_aggregate_summaries(args.output_dir, results)
    plot_feature_table(args.output_dir, feature_rows)
    plot_feature_heatmap(args.output_dir, feature_rows)
    write_validation_readme(
        args.output_dir,
        config,
        args.pattern_count,
        diagnostic_rows,
    )


if __name__ == "__main__":
    main()
