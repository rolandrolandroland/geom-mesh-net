import os

# Fix for "libomp already initialized" error on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your core modules
from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import cluster_visualizer as cv

# Ensure figures directory exists
figures_dir = os.path.join("checkpoints", "figures")
os.makedirs(figures_dir, exist_ok=True)

# 1. Setup
size = 3
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

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dl.point_cloud_collate)

# 2. Initialize Models
models = {
    "Model_1_Baseline": dl.ContinuousNeuralField(),
    "Model_2_Dense": dl.ContinuousNeuralField2(),
    "Model_3_Spatial_Stats": dl.ContinuousNeuralFieldspatstat_01(barcode_bins=barcode_bins)
}

# 3. Generate 3D Visualizations
print("\nGenerating 3D Visualizations...")
eval_batch = [dataset[0]]
coords, domain, labs, xx, yy, zz, full_upp_probs, barcode = dl.point_cloud_collate(eval_batch)

x_flat = xx.flatten()
y_flat = yy.flatten()
z_flat = zz.flatten()
xyz_inputs = torch.stack([x_flat, y_flat, z_flat], dim=1).float()
grid_origin = (xx.min().item(), yy.min().item(), zz.min().item())

style_key = {
    "marks": [0, 1, 2, 3],
    "colors": ["blue", "mediumblue", "brown", "firebrick"],
    "names": ["Host Background", "Host Clusters", "Guest Clusters", "Guest Background"],
    "sizes": [4, 4, 4, 4],
    "opacities": [0.001, 0.1, 0.8, 0.8],
}

plots_to_make = ["thinned_pp3", "original_vox", "predicted_volume"]

for model_name, model in models.items():
    checkpoint_path = os.path.join("checkpoints", model_name, "epoch_1000.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {model_name}: Checkpoint not found at {checkpoint_path}.")
        continue

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with torch.no_grad():
        if "Spatial_Stats" in model_name:
            barcode_expanded = barcode.repeat(xyz_inputs.shape[0], 1).float()
            inputs = torch.cat([xyz_inputs, barcode_expanded], dim=1)
        else:
            inputs = xyz_inputs

        pred = model(inputs)
        predicted_volume = pred.reshape(xx.squeeze(0).shape).numpy()
        original_vox = full_upp_probs.squeeze(0).numpy()

        # Save to the new figures directory
        save_path = os.path.join(figures_dir, f"prediction_{model_name}.png")

        cv.plot_side_by_side_comparison(
            plots_to_make=plots_to_make,
            thinned_coords_dict=coords[0],
            thinned_labels=labs[0],
            original_vox=original_vox,
            predicted_volume=predicted_volume,
            grid_origin=grid_origin,
            style_key=style_key,
            resolution=0.5,
            screenshot_path=save_path,
            off_screen=True,
            show=False
        )
        print(f"Saved visualization for {model_name} to {save_path}")

print("Done.")