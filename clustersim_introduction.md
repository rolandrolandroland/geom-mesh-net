# Cluster simulation algorithm Introduction

Welcome! This is a guide to using the ```clustersim``` function and its accompanying 
helper functions!

## Introduction
The clustersim function here ```clustersim``` is based on the R language `clustersim` function from the R `rapt` package.
The function takes a 3D point pattern as input and relabels it to contain clusters with user defined parameters:  
pcp: overall concentration of guest-type points throughout the entire point pattern   

$\rho_c$ (rho_c): cluster concentration, or percent of points inside a cluster that are guest-type  
$\rho_b$ (rho_b): background concentration, or the percent of points in domains regions that are not inside clusters that are guest-type  
cr: mean cluster radius  
rb: radius dispersity, or mean normalized cluster radius standard deviation  

Additional parameters are discussed later in the document

### Underlying and Overlying Point Patterns
The function requires an underlying point pattern (UPP) in which to generate the clusters.  The function will return an
object with the same points, but new labels to mark which points are guest-type (clustering) and which are host type.
The objects are of the custom `PointPattern3` object class.  The function also takes another `PointPattern3` class
object, the overlying point pattern (OPP).  This pattern will be be used to determine the centers of the clusters.

Before we get into the cluster simulation process, let's talk about this `PointPattern3` class.

###  `PointPattern3` custom class
One way that this differs from the `rapt` package `clustersim` function is by making use of Python's object and method
functionality.  To get started with this class, we need a numpy array of coordinates and a numpy dictionary containing a domain.
To get the coordinates, let's use our `gen_rand_points` function, which takes an intensity and three dimension lengths,
then uses a uniform distribution to generate random coordinates within those dimensions. The number of coordinate
sets generated will be equal to $ intensity \times dim1 \times dim2 \times dim3$. `gen_rand_points` will also give us
vector of labels, or marks, for each point, which we will use later. 
```angular2html
dim1 = 30
dim2 = dim1
dim3 = dim1
intensity = 5
mat, labels = csim.gen_rand_points(intensity, dim1 = dim1, dim2 = dim3, dim3 = dim3)
```
Now let's use those same dimensions to create a dictionary that holds our domain
```angular2html
domain_a = {'x': np.array([0.0, dim1]),
             'y': np.array([0.0, dim2]),
             'z': np.array([0.0, dim3])}
```
Now we can create a `PointPattern3` object with those points and that domain!
```angular2html
pp3_a = csim.PointPattern3(mat, domain_a)
```
In addition to the `mat` (coordinates) and `domain_a` (domain) arguments, there is an optional argument for labels.
We will use this to differentiate between species type in the future, but for now we don't define it.

## Cluster Center Assignment
Alright, so how do we determine where these clusters will be? For that we use overlaying point pattern (OPP).
Ideally, these points will have some minimum distance between them (inhibition) to prevent overlapping clusters, so using a Matern hard-core
point process to generate the points would be ideal.  For simplicity, we are going to use a points on an
equally spaced grid, with a bit of a normal blur applied to add a little irregularity.
The `gen_uniform_points` function used below is simular to the `gen_rand_points` function we just used for the
UPP, except the points are linearly spaced throughout the grid and then each coordinate is blurred using a normal
distribution with user defined mean and standard deviation.

```angular2html
dim1b = 10
dim2b = dim1b
dim3b = dim1b

domain_b = {'x': np.array([0.0, dim1b]),
            'y': np.array([0.0, dim2b]),
            'z': np.array([0.0, dim3b])}

intensityb = 4
blur_mean =0
blur_sd = 0.0001
matb, labelsb = csim.gen_uniform_points(intensityb, dim1 = dim1b, dim2 = dim2b, dim3 = dim3b,
                                   x_blur_mean= blur_mean, x_blur_sd = blur_sd,
                                   y_blur_mean= blur_mean, y_blur_sd = blur_sd,
                                   z_blur_mean= blur_mean, z_blur_sd = blur_sd)
pp3_b = csim.PointPattern3(matb, domain_b, labels = labelsb)
```

### Cluster volume correction
Using irregular cluster shapes will make our cluster volumes no longer equal to the
spherical volumes that we would use to calcualate the number of clusters needed initially.
To solve this, we calculate much more accurate volume estimates for our cluster geometries
using `estimate_cluster_volume` and then update the number of needed cluster accordingly.

### Domain Scaling
`clustersim` finds the total volume of clusters needed to meet the user-defined pcp, rho_c, and rho_b.
Then, it uses the user-defined cluster radius mean and dispersity (cr and rb) to find the number of clusters
required to meet all this criteria.  We then scale and shift the OPP so that it covers the domain of the UPP and after an
optional (but recommended) "buffering" of that domain, there are a number of points that match the needed number of cluster centers.

### Domain Buffering
`clustersim` has the option to "buffer" the domain in which cluster centers can appear.  If the `cut == "buffered"`
argument is used (as it is by default), then the algorithm will not only allow cluster centers to be places within
a distance of `buffer_factor x cr` of the edges.  This prevents clusters from being cut off by the end of the domain.

### Oversampling
In order to ensure that the model includes enough cluster centers, the domain is scaled so that the buffered
domain contains more cluster centers than needed, then randomly samples them to achieve the necessary number.

### Cluster Radius Assignment
After the final cluster centers are determined, a vector of cluster radii is created using a normal distribution
of mean cr and standard deviation cr x rb and each cluster center is assigned a radius.

## Creating a Cluster
A cluster is created from each of these cluster centers using the `assign_clust_points` function.
This takes the following inputs of:  
`dist`: The distance from the cluster center to each point in the pattern
`weighted_dist`: The weighted distance from the cluster center to each point in the pattern (see below)  
`rho_c`: cluster concentration (see above)  
`n_points`: number of guest-type points that the cluster must contain   
`r_max`: cluster radius - the maximum distance a point can be from the cluster center and still be eligible to be part of the cluster  
`r_weighted_max`: the maximum weighted distance a point can be from the cluster center and still be eligible to be part of the cluster  
`prob_function`: How the probability of a point becoming guest-type is calculated from the weighted distance  
`selection`: Determines if guest-type points are selected by sampling using probabilities or if points with smallest weighted distance are chosen

### Distance and Weighted Distance
The `dist` argument is calculated using the Euclidean distance formula
$\sqrt{(x_c -x_i)^2+ (y_c -y_i)^2 + (z_c -z_i)^2}$. The weighted distance is calculated using
the user defined `x_weight, y_weight, z_weight, x_exp, y_exp, z_exp` and the formula  
$\sqrt{(x_{weight}(x_c -x_i))^{x_{exp}}+ (y_{weight}(y_c -y_i))^{y_{exp}} + (z_{weight}(z_c -z_i))^{z_{exp}}}$  
When the weights are at their default value of 1 and the exponents are at their default value of 2,
this is the Euclidean distance formula.  But by changing the weights or exponents,
we can bias the clusters to form in a certain orientation. For example, if the `z_weight`
is set to 0, then clusters will form as vertical (z dimension) stacks. 

The `r_max_weighted_max_ratio` argument allows for the `r_weighted_max` of each cluster center to
be set as a function of that cluster's radius. This allows for cluster boundaries to be automatically defined
as a function of the weighted radius and radius. 

### Probability function
If `selection` is set to the default value of  `sampled`, then guest-type points
are selected from the points within the cluster radius (`r_max`) based on weighted distance ($d_w$) using one of the
following methods  
`Gaussian_decay`: $P = 1- \frac{d_w}{d_w{max}}$  
`inverse`: $P = \frac{1}{d_w + 10^{-10}}$  
`growth`: $P = d_w$  
`constant`: $P = constant$ (for all points within max cluster radius)  
`exponential`: $P = e^{-n*\frac{d_w}{d_w{max}}}$, where $n$ is `prob_frac` argument   
If selection is set to `highest`, then the `n_points` points with the lowest weighted distances
are selected as the guest-type points.

### Using  `rho_c` vs `n_points`
The `clustersim` function calls the `assign_clust_points` function by defining the `rho_c` argument,
not the `n_points` argument. This means that the number of points to be assigned as guest-type
is found by multiplying `rho_c` by the total number of points within the radius of that cluster center.
The option to do this using a set number of points instead (`n_points`) is defined for future development.

## Creating the background
Finally we must assign guest-type points in the background (i.e., the sections of the domain
that aren't within the radius of a cluster center) to get the background concentration
to `rho_b`. We do this by calling the `gen_back_guest` function. 

# `clustersim` Output
Alright, now that the function has run, what is it returning?  The answer is an
object of the `PointPattern3` class (described above) with the same coordinates as the
`upp` that was used, but now labels marking the identity of each point:  
0: host-type point in the background
1: host-type point in clustering domain  
2: guest-type point in clustering domain (aka inside of a cluster)  
3: guest-type point in the background  

We can check if the function assigned the proper number of points for each using
the following  
```
print(f"pcp is {(sum(clust_pattern.labels == 2) + sum(clust_pattern.labels == 3))/ clust_pattern.n_points}")
print(f" rho_c is {sum(clust_pattern.labels == 2) / (sum(clust_pattern.labels == 2) + sum(clust_pattern.labels == 1))}")
print(f" rho_b is {sum(clust_pattern.labels == 3)/ (sum(clust_pattern.labels == 0))}")
```


