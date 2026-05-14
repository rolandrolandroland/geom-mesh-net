# Data Loader
We must now take our simulated point patterns and load them into our neural network. For this, we use the DataLoader class. 

## LoadData class
We will use this class to create our models.  First, they must load in the point patterns that we made and saved in the `data_factory.py` script as well as the file that contains
the cluster parameters. 

### Thinning data
The data is thinned using the `thin_cluster` function

### Density Grid
We create a voxelized density grid using `generate_density_grid`

## Continuous Neural Field
Here are the nuts and bolts of the model itself. We are taking a location in a grid and returning the probability of a guest type atom existing there. 
For patterns with uniform cluster concentration, this will be a number between the background density `rho_b` and the cluster density `rho_c`.



# Training the Network
## Overview
Again, our goal here is to take a single thinned point pattern and return an isosurface of the original point pattern. 
### Loss Function
We are using binary cross entropy

### Optimizer
We are using an adaptive moment estimation (Adam) optimizer

include plots.  loss is low, but image doesn't match. is the problem that the model is training too much on the background
and we need to weight the clusters more, or is it that it that our model is not complex enough (use ContinuousNeuralField2)
This model does not work well at all. 