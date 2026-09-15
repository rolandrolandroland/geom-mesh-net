# Features I would like to add to cluster simulation algorithm

For `r_max_weighted_max_ratio`, allow for a function input, rather
than just a constant

Allow for a custom `prob_function` input

Clustersim, optimize the dists and weighted dists calculation, 
similar to how dists are optimized in the generate_density_grid

In generate_density_grid, optimize weighted dist calculations

in clustersim, only pass the submatrix like i did in generate_density_grid

add documentation to thin_cluster function and clearly explain how it handles cases where lables != "all"
and make sure that LoadData is compatible with this

could i add features to neural field that are k function trained on point pattern or analog of k function on continuous pattern?

 weight clustered regions greater because currently the majority of the training is done on low intensity space
 
voxelization walkthrough 

there is at least some discrepancy between the rho_c, rho_b, cr, and pcp values used to create clusters and the 
measured values of the resulting point patterns. should the voxelized patterns be created using the expected or observed 
values?

My isosurfaces seem to be normally distrubuted- makes sense, since i used gaussian decay cluster concentration distribution.
However, I need to check that i am indeed plotting the percentile and actual value properly. should check this when
1) clusters are uniform concentration
2) background density is much higher


Feature selection:
which features capture the most information?

we are training on the voxelized version of the point cloud, not the point cloud,
so it might be more accurate if we create the voxelized version by calculating
it from the point cloud (binning the points) rather than by generating it using 
the parameters of the point cloud.

add r_min to spatial barcode
make model able to adjust barcode arguments

restructure project so that it makes sense. perhaps i will create an "examples" directory
currently, "train_networks_for_compare_multi_pattern" creates the 3 models
on 5 patterns, then "evaluate_and_plot_multi_pattern" makes the plots
compare_benchmarks contains the functions for plotting benchmarks and speed
run_benchmark_plotting runs compare_benchmarks

update so that we can calculate the total number of points per epoch
If an epoch processes $B$ batches,
each containing $S$ point-cloud samples, 
and each sample consists of $N$ points, 
the total points processed in that epoch is:
$$\text{Total Points per Epoch} = B \times S \times N$$
Alternatively, if your point-cloud samples have varying point densities,
it is simply the sum of all point counts across every sample in the dataset:
$$\text{Total Points per Epoch} = \sum_{i=1}^{M} N_i$$
where $M$ is the total number of training samples in the dataset and $N_i$ is the point count of sample $i$.
