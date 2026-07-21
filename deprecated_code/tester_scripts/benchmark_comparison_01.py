import matplotlib.pyplot as plt


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

# Example Usage:
# generate_benchmark_plots([baseline_metrics, dense_metrics], ["Baseline", "Dense"])