import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from geom_mesh_net.core_functions import data_loader as dl
from torch.utils.data import DataLoader



# numer of training sets
size = 1

# names to look for
data_file_name = "clust_pattern_"
params_file_name = "pattern_stats"

#thinning prob
probs = 0.1

# voxel size
resolution = 0.5

# indices of values within params_stats
pcp_ind = 1
rho_c_ind = 4
rho_b_ind = 7

# path to data
data_prefix = "data/"
params_prefix = "data/"

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

# load in saved data
dataset = dl.LoadData(size = size,
             data_file_name = data_file_name, params_file_name = params_file_name,
             probs = probs, resolution = resolution,
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
                        batch_size = 1,
                        shuffle = True,
                        collate_fn = dl.point_cloud_collate)

# initialize model, loss function, and optimizer
model = dl.ContinuousNeuralField()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# define number of epochs
epochs = 1000

# run training loop
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        # unpack batch
        thinned_coords, domain, thinned_labs, xx, yy, zz, full_upp_probs = batch


        # flatten coords for use in  model
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = zz.flatten()

        # make matrix of inputs
        inputs = torch.stack([x_flat, y_flat, z_flat], dim = 1).float()

        # flatten target probs
        targets = full_upp_probs.flatten().unsqueeze(1).float()

        # clear memory
        optimizer.zero_grad()

        # make guess and grade it (loss)
        preds = model(inputs)
        loss = loss_fn(preds, targets)

        # backpropogate
        loss.backward()

        # update weights
        optimizer.step()
    # --- PRINT THE RESULTS OF THE EPOCH ---
    # This aligns with the outer 'for epoch in range(epochs):' loop
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../phase1_model.pt")