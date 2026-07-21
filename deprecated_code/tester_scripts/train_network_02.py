import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from geom_mesh_net.core_functions import data_loader as dl
from torch.utils.data import DataLoader


# ==========================================
# BENCHMARKING HELPER FUNCTIONS
# ==========================================
def calculate_iou(preds, targets, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) for 3D volumetric boundaries.
    """
    pred_mask = (preds > threshold).float()
    target_mask = (targets > threshold).float()

    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum() - intersection

    return (intersection / (union + 1e-8)).item()


# ==========================================
# 1. SETUP AND HYPERPARAMETERS
# ==========================================
size = 1
data_file_name = "clust_pattern_"
params_file_name = "pattern_stats"
probs = 0.1
resolution = 0.5
pcp_ind = 1
rho_c_ind = 4
rho_b_ind = 7
data_prefix = "../data/"
params_prefix = "../data/"
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

# Load in saved data
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
                      overlap_prob=overlap_prob)

dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=dl.point_cloud_collate)

# Initialize model, loss function, and optimizer
model = dl.ContinuousNeuralFieldspatstat_01()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 2. FILE SAVING & LOGGING SETUP
# ==========================================
log_file_path = "../walkthroughs/trained_models/training_benchmarks_log.csv"
best_model_path = "../walkthroughs/trained_models/databest_spatial_model_weights.pth"
final_model_path = "../walkthroughs/trained_models/final_spatial_model_weights.pth"

# Initialize CSV and write the headers
with open(log_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Time (s)", "Speed (pts/sec)", "Loss", "MAE", "PSNR (dB)", "IoU"])

# Track the best Mean Absolute Error so we know when to save
best_mae = float('inf')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("=" * 60)
print(f"MODEL INITIALIZED | Total Trainable Parameters: {total_params:,}")
print(f"Logging metrics to: {log_file_path}")
print("=" * 60)

epochs = 100

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
model.train()
for epoch in range(epochs):

    epoch_start_time = time.time()
    epoch_points_processed = 0

    epoch_loss = 0.0
    epoch_mae = 0.0
    epoch_psnr = 0.0
    epoch_iou = 0.0
    num_batches = 0

    for batch in dataloader:
        coords, domain, labs, xx, yy, zz, full_upp_probs, barcode = batch

        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = zz.flatten()
        xyz_inputs = torch.stack([x_flat, y_flat, z_flat], dim=1).float()

        n_points_batch = xyz_inputs.shape[0]
        barcode_expanded = barcode.repeat(n_points_batch, 1).float()

        epoch_points_processed += n_points_batch

        conditional_inputs = torch.cat([xyz_inputs, barcode_expanded], dim=1)
        targets = full_upp_probs.flatten().unsqueeze(1).float()

        optimizer.zero_grad()
        preds = model(conditional_inputs)
        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()

        # Benchmarks
        with torch.no_grad():
            mse = F.mse_loss(preds, targets)
            mae = F.l1_loss(preds, targets)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            iou = calculate_iou(preds, targets, threshold=0.5)

            epoch_loss += loss.item()
            epoch_mae += mae.item()
            epoch_psnr += psnr.item()
            epoch_iou += iou
            num_batches += 1

    # Calculate Averages & Speed
    epoch_duration = time.time() - epoch_start_time
    throughput = epoch_points_processed / epoch_duration
    avg_loss = epoch_loss / num_batches
    avg_mae = epoch_mae / num_batches
    avg_psnr = epoch_psnr / num_batches
    avg_iou = epoch_iou / num_batches

    # Print to Console
    print(
        f"Epoch [{epoch + 1}/{epochs}] | Time: {epoch_duration:.2f}s | Speed: {throughput:,.0f} pts/sec | MAE: {avg_mae:.4f} | PSNR: {avg_psnr:.2f} dB | IoU: {avg_iou:.4f}")

    # --- SAVE LOGS TO CSV ---
    with open(log_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [epoch + 1, round(epoch_duration, 2), round(throughput, 0), avg_loss, avg_mae, avg_psnr, avg_iou])

    # --- SAVE THE BEST MODEL WEIGHTS ---
    if avg_mae < best_mae:
        best_mae = avg_mae
        # It is best practice in PyTorch to save only the state_dict (the weights/biases)
        torch.save(model.state_dict(), best_model_path)
        print(f"   -> 💾 New best model saved! (MAE dropped to {best_mae:.4f})")

# Final wrap-up save
torch.save(model.state_dict(), final_model_path)
print("=" * 60)
print(f"Training Complete! Final model saved to {final_model_path}")
print("=" * 60)