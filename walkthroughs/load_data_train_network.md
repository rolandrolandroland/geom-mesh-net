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

### Loading the data
We unpack our data as a `LoadData` class object. This looks to open a `.npy` file that was created using the `data_factory.py`
script. The object should have been created using a call such as
```
   np.savez(name,
             coords=clust_pattern.coords,
             domain = clust_pattern.domain,
             labels=clust_pattern.labels,
             radii=rads, centers=centers)
```
### Loss Function
We are using binary cross entropy

### Optimizer
We are using an adaptive moment estimation (Adam) optimizer

include plots.  loss is low, but image doesn't match. is the problem that the model is training too much on the background
and we need to weight the clusters more, or is it that our model is not complex enough (use ContinuousNeuralField2)
This model does not work well at all. 