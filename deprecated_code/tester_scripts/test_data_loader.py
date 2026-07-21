from geom_mesh_net import data_loader as dl
from geom_mesh_net import clustersim as csim
from torch.utils.data import DataLoader

size = 10
data_file_name = "clust_pattern_"
params_file_name = "pattern_stats"

probs = {0: 0.1,
         1: 0.1,
         2: 0.1,
         3: 0.1}
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

data = dl.LoadData(size = size,
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


loader = DataLoader(data, batch_size = 1)
thinned_coords, domain, thinned_labs, xx, yy, zz, full_upp_probs = next(iter(loader))
print("--- Grid Info ---")
print(f"Type: {type(full_upp_probs)}")
print(f"Shape: {full_upp_probs.shape}")
print(f"Min probability: {full_upp_probs.min()}")
print(f"Max probability: {full_upp_probs.max()}")

print("\n--- Thinned Atoms Info ---")
print(f"X-coords shape: {thinned_coords['x'].shape}")
print(f"Labels shape: {thinned_labs.shape}")