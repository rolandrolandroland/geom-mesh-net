import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your core module
from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import spatial_stats_01 as spst

# ==========================================
# 2. DATA SETUP (Load 3 Patterns)
# ==========================================
print("Loading Data...")
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
epochs = 1000

# ==========================================
# 3. MODEL INITIALIZATIONS
# ==========================================
# Use clean folder-friendly names for the keys
models = {
    "Model_1_Baseline": dl.ContinuousNeuralField(),
    "Model_2_Dense": dl.ContinuousNeuralField2(),
    "Model_3_Spatial_Stats": dl.ContinuousNeuralFieldspatstat_01(barcode_bins=barcode_bins)
}

loss_fn = nn.BCELoss()

# ==========================================
# 4. TRAINING LOOP FOR EACH MODEL
# ==========================================
for model_name, model in models.items():
    print(f"\n{'=' * 50}\nTraining {model_name}\n{'=' * 50}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    start_train_time = time.time()

    # Ensure save directory exists for this specific model
    save_dir = f"checkpoints/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the CSV Logger for this model
    log_file_path = os.path.join(save_dir, "training_metrics.csv")
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Time (s)", "MAE", "PSNR", "IoU"])

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_psnr = 0.0
        epoch_iou = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack batch
            coords, domain, labs, xx, yy, zz, full_upp_probs, barcode = batch

            x_flat = xx.flatten()
            y_flat = yy.flatten()
            z_flat = zz.flatten()
            xyz_inputs = torch.stack([x_flat, y_flat, z_flat], dim=1).float()

            targets = full_upp_probs.flatten().unsqueeze(1).float()

            # FORMAT INPUTS DEPENDING ON WHICH MODEL IS RUNNING
            if "Spatial_Stats" in model_name:
                n_points = xyz_inputs.shape[0]
                barcode_expanded = barcode.repeat(n_points, 1).float()
                inputs = torch.cat([xyz_inputs, barcode_expanded], dim=1)
            else:
                inputs = xyz_inputs

            # Forward and Backward passes
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            # Benchmarks
            with torch.no_grad():
                mse = F.mse_loss(preds, targets)
                epoch_mae += F.l1_loss(preds, targets).item()
                epoch_psnr += (-10.0 * torch.log10(mse + 1e-8)).item()
                epoch_iou += spst.calculate_iou(preds, targets, threshold=0.5)
                epoch_loss += loss.item()
                num_batches += 1

        # Calculate epoch averages
        avg_mae = epoch_mae / num_batches
        avg_psnr = epoch_psnr / num_batches
        avg_iou = epoch_iou / num_batches

        # Track cumulative time for plotting later
        total_elapsed_time = time.time() - start_train_time

        # Log metrics to CSV every epoch
        with open(log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, round(total_elapsed_time, 2), avg_mae, avg_psnr, avg_iou])

        # Save checkpoints every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Epoch [{epoch + 1}/{epochs}] | MAE: {avg_mae:.4f} | PSNR: {avg_psnr:.2f} dB | IoU: {avg_iou:.4f} --> [Saved Checkpoint]")