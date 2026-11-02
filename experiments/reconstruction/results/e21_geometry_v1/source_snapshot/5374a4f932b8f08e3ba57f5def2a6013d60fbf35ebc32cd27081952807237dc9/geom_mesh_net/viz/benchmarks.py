import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def generate_benchmark_plots(metrics_list, model_names):
    """
    metrics_list: List of dictionaries containing keys ['mae', 'psnr', 'iou']
    model_names: List of strings
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(model_names):
        m = metrics_list[i]
        axes[0].plot(m['mae'], label=name)
        axes[1].plot(m['psnr'], label=name)
        axes[2].plot(m['iou'], label=name)

    axes[0].set_title("Mean Absolute Error")
    axes[1].set_title("PSNR (dB)")
    axes[2].set_title("IoU")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def generate_benchmark_plots_per_pattern(pattern_metrics_dict, output_dir="checkpoints/figures", show=False):
    """
    Generates and saves separate benchmark metric plots (MAE, PSNR, IoU) for each pattern
    with unique filenames including the pattern name.
    """
    os.makedirs(output_dir, exist_ok=True)

    for pattern_id, models_data in pattern_metrics_dict.items():
        if not models_data:
            print(f"Skipping Pattern {pattern_id}: No model data provided.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Benchmark Performance Comparison — Pattern {pattern_id}", fontsize=16, fontweight='bold')

        for model_name, m in models_data.items():
            epochs = m.get('epoch', list(range(1, len(m['mae']) + 1)))

            axes[0].plot(epochs, m['mae'], label=model_name, linewidth=2)
            axes[1].plot(epochs, m['psnr'], label=model_name, linewidth=2)
            axes[2].plot(epochs, m['iou'], label=model_name, linewidth=2)

        axes[0].set_title("Mean Absolute Error (Lower is Better)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MAE")

        axes[1].set_title("PSNR in dB (Higher is Better)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PSNR (dB)")

        axes[2].set_title("Intersection over Union (Higher is Better)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("IoU")

        for ax in axes:
            ax.legend(loc="best")
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(output_dir, f"pattern_{pattern_id}_benchmark_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure for Pattern {pattern_id} -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def load_csv_metrics(csv_path):
    """
    Parses logged training metrics from a CSV file.
    Throughput/Speed is only logged if explicitly present in the CSV headers.
    """
    metrics = {'epoch': [], 'mae': [], 'psnr': [], 'iou': [], 'time': [], 'throughput': []}
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        cumulative_time = 0.0
        for i, row in enumerate(reader):
            epoch_val = int(row.get('Epoch', len(metrics['epoch']) + 1))
            mae_val = float(row.get('MAE', 0))
            psnr_val = float(row.get('PSNR', 0) or row.get('PSNR (dB)', 0))
            iou_val = float(row.get('IoU', 0))

            # Time tracking
            if 'Cumulative Time (s)' in row or 'Cumulative Time' in row:
                time_val = float(row.get('Cumulative Time (s)', row.get('Cumulative Time')))
            elif 'Epoch Time (s)' in row or 'Time per Epoch' in row or 'Epoch Time' in row:
                step_time = float(row.get('Epoch Time (s)', row.get('Time per Epoch', row.get('Epoch Time'))))
                cumulative_time += step_time
                time_val = cumulative_time
            elif 'Time' in row or 'Time (s)' in row:
                raw_time = float(row.get('Time', row.get('Time (s)')))
                time_val = raw_time
            else:
                cumulative_time += 1.0
                time_val = cumulative_time

            # Speed / Throughput tracking (only append if column exists in CSV)
            raw_speed = row.get('Speed (pts/sec)', row.get('Speed', row.get('Throughput', None)))
            if raw_speed is not None and raw_speed != '':
                metrics['throughput'].append(float(raw_speed))

            metrics['epoch'].append(epoch_val)
            metrics['mae'].append(mae_val)
            metrics['psnr'].append(psnr_val)
            metrics['iou'].append(iou_val)
            metrics['time'].append(time_val)

    return metrics


def plot_convergence_vs_time(models_data, pattern_id=0, output_dir="checkpoints/figures", show=False):
    """
    Plots training metrics (MAE, PSNR, & IoU) against cumulative wall-clock seconds for a specific pattern.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pattern_{pattern_id}_convergence_vs_wall_time.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Convergence vs. Cumulative Wall-Clock Time — Pattern {pattern_id}", fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (model_name, m) in enumerate(models_data.items()):
        color = colors[i % len(colors)]
        times = m.get('time', list(range(1, len(m['mae']) + 1)))

        axes[0].plot(times, m['mae'], label=model_name, linewidth=2, color=color)
        axes[1].plot(times, m['psnr'], label=model_name, linewidth=2, color=color)
        axes[2].plot(times, m['iou'], label=model_name, linewidth=2, color=color)

    axes[0].set_title("Mean Absolute Error (Lower is Better)")
    axes[0].set_xlabel("Cumulative Time (Seconds)")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="best")

    axes[1].set_title("PSNR in dB (Higher is Better)")
    axes[1].set_xlabel("Cumulative Time (Seconds)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best")

    axes[2].set_title("Intersection over Union (Higher is Better)")
    axes[2].set_xlabel("Cumulative Time (Seconds)")
    axes[2].set_ylabel("IoU")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pareto_efficiency(models_summary, pattern_id=0, output_dir="checkpoints/figures", show=False):
    """
    Creates a Pareto Efficiency Scatter Plot for a specific pattern with a unique filename.
    Skipped if no throughput values are available in the dataset.
    """
    names = list(models_summary.keys())
    if not names:
        return

    # Check if throughput values exist
    has_throughput = any(models_summary[k].get('throughput') is not None for k in names)
    if not has_throughput:
        print(f"Skipping Pareto Efficiency Plot for Pattern {pattern_id}: No throughput (pts/sec) data found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pattern_{pattern_id}_pareto_efficiency_bubble.png")

    fig, ax = plt.subplots(figsize=(9, 6.5))

    throughputs = [models_summary[k]['throughput'] for k in names]
    ious = [models_summary[k]['iou'] for k in names]
    params = [models_summary[k]['params'] for k in names]

    max_p = max(params) if max(params) > 0 else 1
    param_sizes = [max(150, (p / max_p) * 1200) for p in params]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))

    ax.scatter(throughputs, ious, s=param_sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

    for i, name in enumerate(names):
        ax.annotate(
            f"{name}\n({params[i]:,} params)",
            (throughputs[i], ious[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8, alpha=0.8)
        )

    ax.set_title(f"Model Efficiency Pareto Curve — Pattern {pattern_id}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Throughput — Processing Speed (Points / Second) → [Higher is Better]", fontsize=11)
    ax.set_ylabel("Final IoU — Boundary Accuracy → [Higher is Better]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    ax.text(0.02, 0.03, "Bubble area represents total parameter count",
            transform=ax.transAxes, fontsize=9, fontstyle='italic',
            bbox=dict(boxstyle="square", fc="white", ec="none", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_speed_and_cost_bars(models_summary, pattern_id=0, output_dir="checkpoints/figures", show=False):
    """
    Generates comparative bar charts for Performance & Computational Cost.
    Omits the throughput (pts/sec) bar panel if no throughput values exist in the dataset.
    """
    names = list(models_summary.keys())
    if not names:
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pattern_{pattern_id}_speed_cost_bar_breakdown.png")

    has_throughput = any(models_summary[k].get('throughput') is not None for k in names)

    x = np.arange(len(names))
    width = 0.55
    epoch_times = [models_summary[k]['epoch_time_sec'] for k in names]
    params_k = [models_summary[k]['params'] / 1000.0 for k in names]

    if has_throughput:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Computational Cost & Performance Benchmark Summary — Pattern {pattern_id}", fontsize=14,
                     fontweight='bold')

        throughputs = [models_summary[k]['throughput'] for k in names]

        # 1. Throughput
        bars1 = axes[0].bar(x, throughputs, width, color='#2b5c8f')
        axes[0].set_title("Processing Throughput (pts/sec)")
        axes[0].set_ylabel("Points / Sec (Higher is Better)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15, ha='right')
        axes[0].bar_label(bars1, fmt='{:,.0f}', padding=3)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Time Per Epoch
        bars2 = axes[1].bar(x, epoch_times, width, color='#d95f02')
        axes[1].set_title("Time per Epoch (Seconds)")
        axes[1].set_ylabel("Seconds / Epoch (Lower is Better)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15, ha='right')
        axes[1].bar_label(bars2, fmt='{:.2f}s', padding=3)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # 3. Model Complexity
        bars3 = axes[2].bar(x, params_k, width, color='#7570b3')
        axes[2].set_title("Model Complexity (Parameters)")
        axes[2].set_ylabel("Parameters in Thousands (k)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=15, ha='right')
        axes[2].bar_label(bars3, fmt='{:.1f}k', padding=3)
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        # 2-Panel Plot: Time per Epoch & Model Complexity (leaving out throughput)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Computational Cost & Performance Benchmark Summary — Pattern {pattern_id}", fontsize=14,
                     fontweight='bold')

        # 1. Time Per Epoch
        bars1 = axes[0].bar(x, epoch_times, width, color='#d95f02')
        axes[0].set_title("Time per Epoch (Seconds)")
        axes[0].set_ylabel("Seconds / Epoch (Lower is Better)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15, ha='right')
        axes[0].bar_label(bars1, fmt='{:.2f}s', padding=3)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Model Complexity
        bars2 = axes[1].bar(x, params_k, width, color='#7570b3')
        axes[1].set_title("Model Complexity (Parameters)")
        axes[1].set_ylabel("Parameters in Thousands (k)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15, ha='right')
        axes[1].bar_label(bars2, fmt='{:.1f}k', padding=3)
        axes[2].grid(axis='y', linestyle='--', alpha=0.7) if len(axes) > 2 else axes[1].grid(axis='y', linestyle='--',
                                                                                             alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)