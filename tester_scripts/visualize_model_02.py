## This script will visualize the data produced in train_network.py

import numpy as np
import torch
import pyvista as pv
from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import cluster_visualizer as cv

from torch.utils.data import DataLoader



# initialize load data
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
# get a single file
batch = next(iter(dataloader))
# unpack batch
thinned_coords, domain, thinned_labs, xx, yy, zz, full_upp_probs = batch

## initialize network
model = dl.ContinuousNeuralField2()
# load in saved data
model.load_state_dict(torch.load("../phase2_model_250.pt"))
# set model to evaluation mode
model.eval()

## run inference
# flatten coords for use in  model
x_flat = xx.flatten()
y_flat = yy.flatten()
z_flat = zz.flatten()

# make matrix of inputs
inputs = torch.stack([x_flat, y_flat, z_flat], dim = 1).float()

# turn off math memory
with torch.no_grad():
    preds = model(inputs)

# reshape predictions into 3d numpy cube
grid_shape = xx.squeeze(0).shape
predicted_volume = preds.reshape(grid_shape).numpy()
#predicted_volume = full_upp_probs.squeeze(0).numpy()
# need to transpose to match coordinates
#predicted_volume = np.transpose(predicted_volume, (1, 0, 2))
# --- RENDER 3D VOLUME ---
print("Rendering 3D Volume...")
# 1. Grab the dictionary of thinned points AND their labels from the batch
thinned_coords_dict = thinned_coords[0]
thinned_labels_array = thinned_labs[0]

# 2. Extract the original points AND original labels from the raw file
raw_data = np.load("../data/clust_pattern_0.npz", allow_pickle=True)
original_coords_dict = raw_data['coords'].item()
original_labels_array = raw_data['labels']

# 3. Define your styling key
style_key = {
    "marks": [0, 1, 2, 3],
    "colors": ["blue", "mediumblue", "brown", "firebrick"],
    "names": ["Host Background", "Host Clusters", "Guest Clusters", "Guest Background"],
    "sizes": [4, 4, 4, 4],
    "opacities": [0.001, 0.1, 0.8, 0.8]
}

# 4. Extract the exact mathematical starting coordinates for PyVista
grid_origin = (xx.min().item(), yy.min().item(), zz.min().item())

# --- RENDER COMPARISON ---
cv.plot_side_by_side_comparison(
    orig_coords_dict=original_coords_dict,
    orig_labels=original_labels_array,
    thinned_coords_dict=thinned_coords_dict,
    thinned_labels=thinned_labels_array,
    predicted_volume=predicted_volume,
    grid_origin=grid_origin,  # Passing the true origin here!
    style_key=style_key,
    resolution=resolution
)