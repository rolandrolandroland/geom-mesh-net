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