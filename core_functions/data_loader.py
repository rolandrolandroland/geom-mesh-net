import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from geom_mesh_net.core_functions import clustersim as csim
from geom_mesh_net.core_functions import voxelize_clusters as vc


# class for loading data
class LoadData(Dataset):
    """
    Class for loading data

    Args:
        size (int): number of datasets to read in
        data_file_name (string): prefix of data
        params_file_name (string): name of params file
        probs (float): data will be thinned to

    """
    # initialize with size and prefix for unpacking
    def __init__(self, size,
                 data_file_name, params_file_name,
                 probs, resolution,
                 pcp_ind = 1, rho_c_ind = 4, rho_b_ind = 7,
                 data_prefix="", params_prefix="",
                 marks = "all",
                 n_points=None,
                 x_weight=1,
                 y_weight=1,
                 z_weight=1,
                 x_exp=2,
                 y_exp=2,
                 z_exp=2,
                 r_weighted_max=None,
                 r_max_weighted_max_ratio=None,
                 prob_function="Gaussian_decay",
                 prob_exp=-3,
                 selection='sampled',
                 overlap_prob="highest"
                 ):
        self.data_prefix = data_prefix

        self.size = size
        self.data_file_name = data_file_name

        params_name = params_prefix + params_file_name + ".npy"
        # unpack data
        self.params = np.load(params_name, allow_pickle=True)
        self.pcp_ind = pcp_ind
        self.rho_c_ind = rho_c_ind
        self.rho_b_ind = rho_b_ind
        self.probs = probs
        self.resolution = resolution
        self.marks = marks
        self.n_points = n_points
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.z_weight = z_weight
        self.x_exp = x_exp
        self.y_exp = y_exp
        self.z_exp = z_exp
        self.r_weighted_max = r_weighted_max
        self.r_max_weighted_max_ratio = r_max_weighted_max_ratio
        self.prob_function = prob_function
        self.prob_exp = prob_exp
        self.selection = selection
        self.overlap_prob = overlap_prob

    def __len__(self):
        return self.size
    def __getitem__(self,key):
        # name to unpack
        name = self.data_prefix + self.data_file_name + str(key) + ".npz"
        # unpack data
        data = np.load(name, allow_pickle=True)
        # assign each value
        coords = data["coords"].item()
        domain = data["domain"].item()
        labels = data["labels"]
        radius = data["radii"]
        centers = data["centers"].item()

        # must thin data
        thinned_coords, thinned_labs = csim.thin_cluster(coords, self.probs, labels=labels,
                                                         marks=self.marks)
        grid_size = [domain['x'][1] - domain['x'][0], domain['y'][1] - domain['y'][0], domain['z'][1] - domain['z'][0]]
        rho_c = self.params[key, self.rho_c_ind]
        rho_b = self.params[key, self.rho_b_ind]
        #pcp = self.params[key, self.pcp_ind]
        # generate a density grid based on the parameters for the clustered data
        xx, yy, zz, full_upp_probs = vc.generate_density_grid(grid_size = grid_size, cluster_centers=centers,
                                    radii=radius,
                                    rho_c=rho_c, rho_b=rho_b,
                                       n_points=self.n_points,
                                       resolution=self.resolution,
                                       x_weight=self.x_weight,
                                       y_weight=self.y_weight,
                                       z_weight=self.z_weight,
                                       x_exp=self.x_exp,
                                       y_exp=self.y_exp,
                                       z_exp=self.z_exp,
                                       r_weighted_max=self.r_weighted_max,
                                       r_max_weighted_max_ratio=self.r_max_weighted_max_ratio,
                                       prob_function=self.prob_function,
                                       prob_exp=self.prob_exp,
                                       selection=self.selection,
                                       overlap_prob=self.overlap_prob
                                       )
        return thinned_coords, domain, thinned_labs, xx, yy, zz, full_upp_probs

class ContinuousNeuralField(nn.Module):
    def __init__(self):
        super().__init__()
        # define model: we are using a sequential multi layer perceptron with 3 features, a 128 neurons hidden layer, and 1 output with a ReLU activation function
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # make sure answer is between 0 and 1
        )
class ContinuousNeuralField2(nn.Module):
    def __init__(self):
        super().__init__()
    # need a more complex model
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    # forward pass input through model
    def forward(self, x):
        return self.model(x)


# custom collate function to handle variable-length point clouds
def point_cloud_collate(batch):
    # batch is a list of tuples, where each tuple is the 7 items returned by __getitem__
    # unzip batch into separate lists
    thinned_coords, domains, thinned_labs, xxs, yys, zzs, probs = zip(*batch)
    # Convert grids to tensors and stack them cleanly
    xx_batch = torch.tensor(np.array(xxs))
    yy_batch = torch.tensor(np.array(yys))
    zz_batch = torch.tensor(np.array(zzs))
    probs_batch = torch.tensor(np.array(probs))

    # We can just leave the variable-length items as standard Python lists
    return thinned_coords, domains, thinned_labs, xx_batch, yy_batch, zz_batch, probs_batch