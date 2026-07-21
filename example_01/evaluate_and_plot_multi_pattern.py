import os

# Fix for "libomp already initialized" error on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import core modules
from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import cluster_visualizer as cv

# Ensure figures directory exists
figures_dir = os.path.join("checkpoints", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Dataset configuration matching pattern training setup
size = 5  # Number of patterns trained in train_pattern_models.py
probs = 0.1
resolution = 0.5
barcode_bins = 5

dataset = dl.LoadData(
    size=size,
    data_file_name="clust_pattern_",
    params_file_name="pattern_stats",
    probs=probs,
    resolution=resolution,
    pcp_ind=1, rho_c_ind=4, rho_b_ind=7,
    data_prefix="../data/",
    params_prefix="../data/",
    marks="all",
    prob_function="Gaussian_decay",
    prob_exp=-3,
    selection='sampled',
    overlap_prob="highest",
    barcode_bins=barcode_bins,
    barcode_r_max=15.0,
    barcode_sample_size=500
)

style_key = {
    "marks": [0, 1, 2, 3],
    "colors": ["blue", "mediumblue", "brown", "firebrick"],
    "names": ["Host Background", "Host Clusters", "Guest Clusters", "Guest Background"],
    "sizes": [4, 4, 4, 4],
    "opacities": [0.001, 0.1, 0.8, 0.8],
}

plots_to_make = ["thinned_pp3", "original_vox", "predicted_volume"]

print("\nGenerating 3D Visualizations across Pattern Models and Epoch Checkpoints...")

for pattern_idx in range(size):
    print(f"\n{'=' * 20} PROCESSING PATTERN {pattern_idx} {'=' * 20}")

    # Create pattern-specific output directory for figures
    pattern_fig_dir = os.path.join(figures_dir, f"pattern_{pattern_idx}")
    os.makedirs(pattern_fig_dir, exist_ok=True)

    # Extract data corresponding specifically to this pattern
    eval_batch = [dataset[pattern_idx]]
    coords, domain, labs, xx, yy, zz, full_upp_probs, barcode = dl.point_cloud_collate(eval_batch)

    # Prepare inputs (XYZ coordinates + expanded Spatial Barcode)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    z_flat = zz.flatten()
    xyz_inputs = torch.stack([x_flat, y_flat, z_flat], dim=1).float()

    barcode_expanded = barcode.repeat(xyz_inputs.shape[0], 1).float()
    inputs = torch.cat([xyz_inputs, barcode_expanded], dim=1)

    grid_origin = (xx.min().item(), yy.min().item(), zz.min().item())
    original_vox = full_upp_probs.squeeze(0).numpy()

    # Instantiate dictionary of models corresponding to subdirectories
    models = {
        "Model_1_Baseline": dl.ContinuousNeuralField(),
        "Model_2_Dense": dl.ContinuousNeuralField2(),
        "Model_3_Spatial_Stats": dl.ContinuousNeuralFieldspatstat_01(barcode_bins=barcode_bins)
    }

    # Iterate over each trained model architecture
    for model_name, model in models.items():
        print(f"\n--- Evaluating {model_name} on Pattern {pattern_idx} ---")

        # Select correct input features (Spatial Stats requires appended spatial barcode)
        if "Spatial_Stats" in model_name:
            inputs = torch.cat([xyz_inputs, barcode_expanded], dim=1)
        else:
            inputs = xyz_inputs

        # Model-specific figure output subfolder
        model_fig_dir = os.path.join(pattern_fig_dir, model_name)
        os.makedirs(model_fig_dir, exist_ok=True)

        for epoch in range(100, 1001, 100):
            # Updated path matching: checkpoints/pattern_{idx}/{model_name}/epoch_{epoch}.pth
            checkpoint_path = os.path.join("checkpoints", f"pattern_{pattern_idx}", model_name, f"epoch_{epoch}.pth")

            if not os.path.exists(checkpoint_path):
                print(
                    f"Skipping Pattern {pattern_idx} | {model_name} | Epoch {epoch}: Checkpoint not found at {checkpoint_path}")
                continue

            # Load weights for this specific model, pattern, & epoch
            model.load_state_dict(torch.load(checkpoint_path))
            model.eval()

            with torch.no_grad():
                pred = model(inputs)
                predicted_volume = pred.reshape(xx.squeeze(0).shape).numpy()

                # Save figure per pattern, model, and epoch
                save_path = os.path.join(model_fig_dir, f"epoch_{epoch}.png")

                # Render and save 3D comparison plot off-screen
                cv.plot_side_by_side_comparison(
                    plots_to_make=plots_to_make,
                    thinned_coords_dict=coords[0],
                    thinned_labels=labs[0],
                    original_vox=original_vox,
                    predicted_volume=predicted_volume,
                    grid_origin=grid_origin,
                    style_key=style_key,
                    resolution=resolution,
                    screenshot_path=save_path,
                    off_screen=True,
                    show=False
                )
                print(f"Saved figure: Pattern {pattern_idx} | {model_name} | Epoch {epoch} -> {save_path}")

print("\nVisualization generation complete for all patterns, models, and epoch checkpoints!")