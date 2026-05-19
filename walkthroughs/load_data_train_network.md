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