# Data Loader
We must now take our simulated point patterns and load them into our neural network. For this, we use the DataLoader class. 

## LoadData class
We will use this class to create our models.  First, they must load in the point patterns that we made and saved in the `data_factory.py` script as well as the file that contains
the cluster parameters. 

### Thinning data
The data may be thinned using the `thin_cluster` function. Our initial model is used on unthinned data, but more advanced
models can characterize thinned data.

### Density Grid
We create a voxelized density grid using `generate_density_grid`

## Continuous Neural Field
Here are the nuts and bolts of the model itself. We are taking a location in a grid and returning the probability of a guest type atom existing there. 
For patterns with uniform cluster concentration, this will be a number between the background density `rho_b` and the cluster density `rho_c`.



# Training the Network
## Overview
The goal of this is to train a single neural field to overfit a single point pattern. A neural field takes an input of
`(x, y, z)` and outputs a single value for a signal- in our case, a continuous density field.

Once the point pattern has been generated, building this model consists of XX steps
1. Load the data
2. Thin the data (note: the thinned data is not yet used in this model)
3. Voxelize the data

## Loading the data
We unpack our data as a `LoadData` class object. This looks to open a `.npy` file that was created using the `data_factory.py`
script. The object should have been created using a call such as
```
   np.savez(name,
             coords=clust_pattern.coords,
             domain = clust_pattern.domain,
             labels=clust_pattern.labels,
             radii=rads, centers=centers)
```

### Voxelize Data
In order to obtain the data that we will use to train our neural field, we must convert our spatial point pattern into a 
voxelized density field.  This is done using the functions in the `voxelize_clusters.py` script.  In our `LoadData` 
class object, the domain, cluster centers, cluster radii, cluster concentration, and background concentration of the
point pattern are fed into the `generate_density_grid` function.  This creates a voxelized grid where each voxel
has a value for the average density of guest type molecules across that voxel by following these steps:

1. The first step in `generate_density_grid` is initializing each voxel to be equal to the background density $rho_b$.  
The voxelized data are then fed into our neural field to predict density as a function of x, y and z coordinates. 
def assign_clust_probs(dist, weighted_dist, density_grid,
2. The grid values inside each cluster domain are modified using `assign_clust_probs`, a continuous analog 
of `assign_clust_points`. The distribution is determined by the `prob_function` and `selection` parameters.  The 
values are clipped to be between 0 and 1 and then rescaled so that the average density inside the cluster domain is $rho_c$,
the cluster concentration.  
3. Any voxels that fall inside of overlapping clustering domains are handled by default by assigning
the highest of the density values.

### Point-cloud-derived targets
`LoadData` uses the simulator-generated target by default so existing experiments retain their original behavior.
Set `target_source="point_cloud"` to estimate guest probability from labeled points instead. The point-cloud target
voxelizes guest and total point counts, applies Gaussian smoothing, and divides the smoothed guest density by the
smoothed total density.

The target and observation sources are configured independently:

```
dataset = LoadData(
    ...,
    barcode_source="thinned",
    barcode_marks=(2, 3),
    target_source="point_cloud",
    target_point_source="original",
    target_guest_marks=(2, 3),
    target_bandwidth=1.5,
)
```

This configuration uses thinned guest observations for the spatial barcode while estimating the training target from
the complete original point cloud. Change `barcode_source` to `"original"` to use the complete point cloud as model
context, or change `target_source` to `"simulation"` to restore the rule-generated target.

## Network Architecture
Initially, we use a two layer continuous neural field (`ContinuousNeuralField` class).  The first layer takes three spatial coordinates and maps them to 
128 neurons. The second layer maps those 128 neurons to 1 output, the probability.  As you can see in the walkthrough, 
this yields a rather poor fit. So then we expand it to a 4 layer network (`ContinuousNeuralField2`) with
three hidden layers, each with 128 neurons, and an output layer. Each layer uses a ReLU activation function. 


### Loss Function
We are using binary cross entropy

### Optimizer
We are using an adaptive moment estimation (Adam) optimizer

include plots.  loss is low, but image doesn't match. is the problem that the model is training too much on the background
and we need to weight the clusters more, or is it that our model is not complex enough (use ContinuousNeuralField2)
This model does not work well at all. 

## Spatial stats
Once we have developed our base model, we are going to try and improve it by incorporating some spatial statistics summary
functions.  We will start by using Ripley's K function. We define the `calculate_spatial_barcode` function
to sample `sample_size` points from a dictionary of points `coords_dict` and find the distance between each pair of
points.  Then a histogram of `bins` from 0 to `r_max`. The number of points in each bin is then fed into the neural network.

`LoadData` also supports the 14 global features from Bennett et al. (2023). These combine guest nearest-neighbor
$G_g$, guest empty-space $F_g$, transformed guest $K_g$, and guest-to-host $G_{gh}$ summaries. The observed curves
are compared with pointwise medians from random relabelings that preserve the number of guest points.

```
from geom_mesh_net.core_functions import paper_spatial_features as psf

paper_config = psf.PaperFeatureConfig(
    n_relabelings=99,
    random_seed=42,
)

dataset = LoadData(
    ...,
    barcode_source="thinned",
    spatial_feature_kind="paper",
    paper_feature_config=paper_config,
    paper_guest_marks=(2, 3),
)
```

Set `barcode_source="original"` to calculate the global features from the full point cloud. The paper feature vector
contains 14 values, so use `ContinuousNeuralFieldGlobalFeatures(feature_count=14)` for the corresponding model.
The staged experiment runner in `example_01/train_networks_for_compare_multi_pattern_02.py` supports analytical CSR
and whole-pattern random-label null models, global features, sampled local voxel features, shared models, and offline
feature caches. See `example_01/PAPER_FEATURE_EXPERIMENTS.md` for commands and interpretation guidance.


# Model Benchmarking
