import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import spatial_stats_01 as spst

# ==========================================
# 1. DATA SETUP
# ==========================================
print("Loading Dataset...")
size = 5  # Ensure size is at least 5 to cover patterns 0-4
dataset = dl.LoadData(
    size=size,
    data_file_name="clust_pattern_",
    params_file_name="pattern_stats",
    probs=0.1,
    resolution=0.5,
    pcp_ind=1, rho_c_ind=4, rho_b_ind=7,
    data_prefix="../data/",
    params_prefix="../data/",
    marks="all",
    barcode_bins=5,
    barcode_r_max=15.0,
    barcode_sample_size=500
)

epochs = 1000
loss_fn = nn.BCELoss()

# ==========================================
# 2. TRAINING LOOP OVER PATTERNS & MODELS
# ==========================================
# Loop through each pattern (0-4)
for pattern_idx in range(5):
    print(f"\n{'#' * 20} PROCESSING PATTERN {pattern_idx} {'#' * 20}")

    # Create subset for this specific pattern
    pattern_dataset = Subset(dataset, [pattern_idx])
    dataloader = DataLoader(pattern_dataset, batch_size=1, shuffle=True, collate_fn=dl.point_cloud_collate)

    # Initialize models for this pattern
    models = {
        "Model_1_Baseline": dl.ContinuousNeuralField(),
        "Model_2_Dense": dl.ContinuousNeuralField2(),
        "Model_3_Spatial_Stats": dl.ContinuousNeuralFieldspatstat_01(barcode_bins=5)
    }

    for model_name, model in models.items():
        print(f"\n--- Training {model_name} on Pattern {pattern_idx} ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

        # Save directory: checkpoints/pattern_X/ModelName/
        save_dir = f"checkpoints/pattern_{pattern_idx}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        log_file_path = os.path.join(save_dir, "training_metrics.csv")
        with open(log_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Time (s)", "MAE", "PSNR", "IoU"])

        start_train_time = time.time()

        for epoch in range(epochs):
            epoch_mae, epoch_psnr, epoch_iou, num_batches = 0, 0, 0, 0

            for batch in dataloader:
                coords, domain, labs, xx, yy, zz, full_upp_probs, barcode = batch

                xyz_inputs = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).float()
                targets = full_upp_probs.flatten().unsqueeze(1).float()

                if "Spatial_Stats" in model_name:
                    n_points = xyz_inputs.shape[0]
                    inputs = torch.cat([xyz_inputs, barcode.repeat(n_points, 1).float()], dim=1)
                else:
                    inputs = xyz_inputs

                optimizer.zero_grad()
                preds = model(inputs)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    mse = F.mse_loss(preds, targets)
                    epoch_mae += F.l1_loss(preds, targets).item()
                    epoch_psnr += (-10.0 * torch.log10(mse + 1e-8)).item()
                    epoch_iou += spst.calculate_iou(preds, targets, threshold=0.5)
                    num_batches += 1

            if (epoch + 1) % 100 == 0:
                avg_mae = epoch_mae / num_batches
                torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))
                print(f"Epoch {epoch + 1} | MAE: {avg_mae:.4f}")

            # Append to metrics log
            with open(log_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, round(time.time() - start_train_time, 2),
                                 epoch_mae / num_batches, epoch_psnr / num_batches, epoch_iou / num_batches])