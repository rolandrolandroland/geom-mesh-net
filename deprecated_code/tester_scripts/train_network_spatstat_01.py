import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time  # <-- NEW: Import the time module for speed benchmarking
import torch
import torch.nn as nn
import torch.nn.functional as F
from geom_mesh_net.core_functions import data_loader as dl
from torch.utils.data import DataLoader



# ==========================================
# 1. SETUP AND HYPERPARAMETERS
# ==========================================
# number of training sets
size = 1

# names to look for
data_file_name = "clust_pattern_"
params_file_name = "pattern_stats"

# thinning prob
probs = 0.1

# voxel size
resolution = 0.5

# indices of values within params_stats
pcp_ind = 1
rho_c_ind = 4
rho_b_ind = 7

# path to data
data_prefix = "../data/"
params_prefix = "../data/"

# which types of points will be thinned
marks = "all"

n_points = None
x_weight = 1
y_weight = 1
z_weight = 1
x_exp = 2
y_exp = 2
z_exp = 2
r_weighted_max = None
r_max_weighted_max_ratio = None
prob_function = "Gaussian_decay"
prob_exp = -3
selection = 'sampled'
overlap_prob = "highest"

# spatial parameters
barcode_bins = 5
barcode_r_max = 15.0
barcode_sample_size = 500

# load in saved data
dataset = dl.LoadData(size=size,
                      data_file_name=data_file_name, params_file_name=params_file_name,
                      probs=probs, resolution=resolution,
                      pcp_ind=pcp_ind, rho_c_ind=rho_c_ind, rho_b_ind=rho_b_ind,
                      data_prefix=data_prefix, params_prefix=params_prefix,
                      marks=marks,
                      n_points=n_points,
                      x_weight=x_weight,
                      y_weight=y_weight,
                      z_weight=z_weight,
                      x_exp=x_exp,
                      y_exp=y_exp,
                      z_exp=z_exp,
                      r_weighted_max=r_weighted_max,
                      r_max_weighted_max_ratio=r_max_weighted_max_ratio,
                      prob_function=prob_function,
                      prob_exp=prob_exp,
                      selection=selection,
                      overlap_prob=overlap_prob,
                      barcode_bins=barcode_bins,
                      barcode_r_max=barcode_r_max,
                      barcode_sample_size=barcode_sample_size
                      )

dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=dl.point_cloud_collate)

# initialize model, loss function, and optimizer
model = dl.ContinuousNeuralFieldspatstat_01(barcode_bins=barcode_bins)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- NEW: Calculate Model Size ---
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("=" * 50)
print(f"MODEL INITIALIZED | Total Trainable Parameters: {total_params:,}")
print("=" * 50)

# define number of epochs
epochs = 1000

# ==========================================
# 2. THE TRAINING LOOP
# ==========================================
model.train()
for epoch in range(epochs):

    # Trackers for the epoch's average performance
    epoch_start_time = time.time()  # <-- NEW: Start the stopwatch
    epoch_points_processed = 0  # <-- NEW: Track total points

    epoch_loss = 0.0
    epoch_mae = 0.0
    epoch_psnr = 0.0
    epoch_iou = 0.0
    num_batches = 0

    for batch in dataloader:
        # 1. Unpack the NEW batch with the barcode included
        coords, domain, labs, xx, yy, zz, full_upp_probs, barcode = batch

        # 2. Flatten coords for use in model
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = zz.flatten()

        # Matrix of spatial inputs: Shape (N, 3)
        xyz_inputs = torch.stack([x_flat, y_flat, z_flat], dim=1).float()

        # 3. --- THE CONDITIONAL MERGE ---
        n_points = xyz_inputs.shape[0]
        barcode_expanded = barcode.repeat(n_points, 1).float()

        epoch_points_processed += n_points  # <-- NEW: add to total points processed

        # Concatenate XYZ and Barcode horizontally into shape (N, 8)
        conditional_inputs = torch.cat([xyz_inputs, barcode_expanded], dim=1)

        # 4. Flatten target probs
        targets = full_upp_probs.flatten().unsqueeze(1).float()

        optimizer.zero_grad()

        # 5. Feed the 8-feature conditional inputs to the model!
        preds = model(conditional_inputs)
        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()

        # 6. --- CALCULATE BENCHMARKS ---
        # We turn off gradients here so PyTorch doesn't track this math for backprop
        with torch.no_grad():
            mse = F.mse_loss(preds, targets)
            mae = F.l1_loss(preds, targets)
            # Add 1e-8 to MSE to prevent taking log10 of absolute zero
            psnr = -10.0 * torch.log10(mse + 1e-8)
            iou = calculate_iou(preds, targets, threshold=0.5)

            # Accumulate totals
            epoch_loss += loss.item()
            epoch_mae += mae.item()
            epoch_psnr += psnr.item()
            epoch_iou += iou
            num_batches += 1

    # --- CALCULATE COMPUTATIONAL COST ---
    epoch_duration = time.time() - epoch_start_time
    throughput = epoch_points_processed / epoch_duration

    # --- PRINT THE RESULTS OF THE EPOCH ---
    avg_loss = epoch_loss / num_batches
    avg_mae = epoch_mae / num_batches
    avg_psnr = epoch_psnr / num_batches
    avg_iou = epoch_iou / num_batches

    print(
        f"Epoch [{epoch + 1}/{epochs}] | Time: {epoch_duration:.2f}s | Speed: {throughput:,.0f} pts/sec | MAE: {avg_mae:.4f} | PSNR: {avg_psnr:.2f} dB | IoU: {avg_iou:.4f}")