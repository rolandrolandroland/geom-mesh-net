import os
import sys
import numpy as np

from geom_mesh_net.core_functions import compare_benchmarks as cb


def run_benchmark_report_suite(checkpoints_dir="checkpoints", output_dir="checkpoints/figures", num_patterns=5):
    """
    Main orchestration routine to load metrics from checkpoints and execute all plotting functions
    in compare_benchmarks for every pattern, ensuring unique filenames with pattern identifiers.
    """
    print("=" * 65)
    print("      EXECUTING COMPREHENSIVE BENCHMARK PLOTTING SUITE")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)
    model_names = ["Model_1_Baseline", "Model_2_Dense", "Model_3_Spatial_Stats"]
    param_lookup = {
        "Model_1_Baseline": 18432,
        "Model_2_Dense": 148900,
        "Model_3_Spatial_Stats": 34305
    }

    pattern_metrics_dict = {}
    found_any = False

    for pattern_idx in range(num_patterns):
        pattern_dir = os.path.join(checkpoints_dir, f"pattern_{pattern_idx}")
        if not os.path.exists(pattern_dir):
            print(f"[Notice] Pattern directory '{pattern_dir}' does not exist. Skipping.")
            continue

        pattern_metrics_dict[pattern_idx] = {}
        models_summary = {}

        for model_name in model_names:
            csv_path = os.path.join(pattern_dir, model_name, "training_metrics.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(pattern_dir, model_name, "training_benchmarks_log.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(pattern_dir, "training_metrics.csv")

            metrics = cb.load_csv_metrics(csv_path) if os.path.exists(csv_path) else None

            if metrics and len(metrics.get('mae', [])) > 0:
                found_any = True
                pattern_metrics_dict[pattern_idx][model_name] = metrics

                final_iou = metrics['iou'][-1] if len(metrics['iou']) > 0 else 0.0
                final_mae = metrics['mae'][-1] if len(metrics['mae']) > 0 else 0.0
                time_array = metrics.get('time', [1.0])
                total_time = time_array[-1] if len(time_array) > 0 else 1.0
                num_epochs = len(metrics['mae'])
                epoch_time = total_time / max(1, num_epochs)

                throughput_vals = metrics.get('throughput', [])
                avg_throughput = float(np.mean(throughput_vals)) if len(throughput_vals) > 0 else None

                models_summary[model_name] = {
                    'throughput': avg_throughput,
                    'epoch_time_sec': epoch_time,
                    'iou': final_iou,
                    'mae': final_mae,
                    'params': param_lookup.get(model_name, 25000)
                }

        if pattern_metrics_dict[pattern_idx]:
            print(f"\n--- Generating Benchmark Plots for Pattern {pattern_idx} ---")

            # 1. Convergence vs Wall-Clock Time (MAE, PSNR, IoU)
            cb.plot_convergence_vs_time(pattern_metrics_dict[pattern_idx], pattern_id=pattern_idx,
                                        output_dir=output_dir, show=False)

            # 2. Pareto Efficiency Bubble Curve (skipped if no throughput in dataset)
            if models_summary:
                cb.plot_pareto_efficiency(models_summary, pattern_id=pattern_idx, output_dir=output_dir, show=False)

                # 3. Speed & Cost Bar Breakdown (omits throughput panel if missing)
                cb.plot_speed_and_cost_bars(models_summary, pattern_id=pattern_idx, output_dir=output_dir, show=False)

    if not found_any:
        print("\n[Notice] No training CSV logs found under 'checkpoints/' for any pattern.")
        print("Please ensure checkpoints/pattern_{idx}/model_name/training_metrics.csv exist.")
        return

    # 4. Per-Pattern Performance Comparison Figures
    print("\n--- Generating Multi-Model Performance Metric Curves ---")
    cb.generate_benchmark_plots_per_pattern(pattern_metrics_dict, output_dir=output_dir, show=False)

    print("\n" + "=" * 65)
    print(f" SUCCESS: All unique benchmark figures generated and saved to: '{output_dir}'")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark_report_suite(
        checkpoints_dir="checkpoints",
        output_dir="checkpoints/figures",
        num_patterns=5
    )