import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from geom_mesh_net.core_functions import clustersim as csim
from geom_mesh_net.core_functions import voxelize_clusters as vc
from geom_mesh_net.core_functions import spatial_stats_01 as spst
from geom_mesh_net.core_functions import point_cloud_fields as pcf
from geom_mesh_net.core_functions import paper_spatial_features as psf


# class for loading data
class LoadData(Dataset):
    """
    Class for loading data

    Args:
        size (int): number of datasets to read in
        data_file_name (string): prefix of data
        params_file_name (string): name of params file
        probs (float): data will be thinned to
        barcode_source (string): point source used for the spatial barcode
        barcode_marks: point labels included in the spatial barcode
        spatial_feature_kind (string): global barcode or paper feature vector
        paper_feature_config: configuration for the paper feature extractor
        paper_guest_marks: labels treated as guest points in paper features
        target_source (string): simulator-generated or point-cloud-derived target
        target_point_source (string): point source used for point-cloud targets
        target_guest_marks: labels treated as guest points in point-cloud targets
        target_bandwidth (float): Gaussian bandwidth in physical coordinate units

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
                 overlap_prob="highest",
                 barcode_bins = 5,
                 barcode_r_max = 15.0,
                 barcode_sample_size = 500,
                 barcode_source = "thinned",
                 barcode_marks = "all",
                 spatial_feature_kind = "barcode",
                 paper_feature_config = None,
                 paper_guest_marks = (2, 3),
                 target_source = "simulation",
                 target_point_source = "original",
                 target_guest_marks = (2, 3),
                 target_bandwidth = 1.5
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
        self.barcode_bins = barcode_bins
        self.barcode_r_max = barcode_r_max
        self.barcode_sample_size = barcode_sample_size
        if barcode_source not in {"thinned", "original"}:
            raise ValueError("barcode_source must be either 'thinned' or 'original'")
        self.barcode_source = barcode_source
        self.barcode_marks = barcode_marks
        if spatial_feature_kind not in {"barcode", "paper"}:
            raise ValueError(
                "spatial_feature_kind must be either 'barcode' or 'paper'"
            )
        self.spatial_feature_kind = spatial_feature_kind
        self.paper_feature_config = (
            paper_feature_config or psf.PaperFeatureConfig()
        )
        self.paper_guest_marks = paper_guest_marks
        if target_source not in {"simulation", "point_cloud"}:
            raise ValueError(
                "target_source must be either 'simulation' or 'point_cloud'"
            )
        self.target_source = target_source
        if target_point_source not in {"thinned", "original"}:
            raise ValueError(
                "target_point_source must be either 'thinned' or 'original'"
            )
        self.target_point_source = target_point_source
        self.target_guest_marks = target_guest_marks
        self.target_bandwidth = target_bandwidth

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
        if self.target_source == "simulation":
            grid_size = [
                domain['x'][1] - domain['x'][0],
                domain['y'][1] - domain['y'][0],
                domain['z'][1] - domain['z'][0],
            ]
            rho_c = self.params[key, self.rho_c_ind]
            rho_b = self.params[key, self.rho_b_ind]
            xx, yy, zz, full_upp_probs = vc.generate_density_grid(
                grid_size=grid_size,
                cluster_centers=centers,
                radii=radius,
                rho_c=rho_c,
                rho_b=rho_b,
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
                overlap_prob=self.overlap_prob,
            )
        else:
            target_coords, target_labels = pcf.choose_point_cloud(
                self.target_point_source,
                coords,
                labels,
                thinned_coords,
                thinned_labs,
            )
            xx, yy, zz, full_upp_probs = (
                pcf.estimate_guest_probability_grid(
                    coords=target_coords,
                    labels=target_labels,
                    domain=domain,
                    resolution=self.resolution,
                    guest_marks=self.target_guest_marks,
                    bandwidth=self.target_bandwidth,
                )
            )

        feature_coords, feature_labels = pcf.choose_point_cloud(
            self.barcode_source,
            coords,
            labels,
            thinned_coords,
            thinned_labs,
        )
        if self.spatial_feature_kind == "paper":
            spatial_features = psf.calculate_global_paper_features(
                feature_coords,
                feature_labels,
                domain,
                guest_marks=self.paper_guest_marks,
                config=self.paper_feature_config,
            ).values
        else:
            feature_coords, _ = pcf.filter_point_cloud(
                feature_coords,
                feature_labels,
                marks=self.barcode_marks,
            )
            spatial_features = spst.calculate_spatial_barcode(
                feature_coords,
                bins=self.barcode_bins,
                r_max=self.barcode_r_max,
                sample_size=self.barcode_sample_size
            )
        return (
            thinned_coords,
            domain,
            thinned_labs,
            xx,
            yy,
            zz,
            full_upp_probs,
            spatial_features,
        )

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

        # forward pass input through model
    def forward(self, x):
        return self.model(x)


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

class ContinuousNeuralFieldspatstat_01(nn.Module):
    def __init__(self, barcode_bins = 5):
        super().__init__()
        input_features = 3 + barcode_bins
    # need a more complex model
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
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


class ContinuousNeuralFieldGlobalFeatures(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than zero")
        input_features = 3 + feature_count
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ContinuousNeuralFieldFeatures(nn.Module):
    def __init__(
        self,
        feature_count=0,
        hidden_width=128,
        hidden_layers=3,
    ):
        super().__init__()
        if feature_count < 0:
            raise ValueError("feature_count must be nonnegative")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be greater than zero")
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be greater than zero")

        layers = []
        input_features = 3 + feature_count
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_features, hidden_width),
                    nn.ReLU(),
                ]
            )
            input_features = hidden_width
        layers.extend([nn.Linear(hidden_width, 1), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# custom collate function to handle variable-length point clouds
def point_cloud_collate(batch):
    # batch is a list of tuples, where each tuple is the 7 items returned by __getitem__
    # unzip batch into separate lists
    thinned_coords, domains, thinned_labs, xxs, yys, zzs, probs, barcodes = zip(*batch)    # Convert grids to tensors and stack them cleanly
    xx_batch = torch.tensor(np.array(xxs))
    yy_batch = torch.tensor(np.array(yys))
    zz_batch = torch.tensor(np.array(zzs))
    probs_batch = torch.tensor(np.array(probs))

    # Convert barcodes to a tensor (Shape: Batch_Size x 5)
    barcode_batch = torch.tensor(np.array(barcodes))

    # We can just leave the variable-length items as standard Python lists
    return thinned_coords, domains, thinned_labs, xx_batch, yy_batch, zz_batch, probs_batch, barcode_batch
