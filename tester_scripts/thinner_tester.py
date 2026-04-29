import geom_mesh_net.clustersim as csim
import numpy as np

import pickle
# make matrix of points
dim1 = 30
dim2 = dim1
dim3 = dim1
intensity = 5
mat, labels = csim.gen_rand_points(intensity, dim1 = dim1, dim2 = dim2, dim3 = dim3)
print(type(mat))
domain_a = {'x': np.array([0.0, dim1]),
             'y': np.array([0.0, dim2]),
             'z': np.array([0.0, dim3])}
print(type(domain_a))
#new_labels = csim.gen_back_guest(labels, back_conc = 0.01)
pp3_a = csim.PointPattern3(mat, domain_a)

dim1b = 10
dim2b = dim1b
dim3b = dim1b

domain_b = {'x': np.array([0.0, dim1b]),
            'y': np.array([0.0, dim2b]),
            'z': np.array([0.0, dim3b])}

#print(test)
intensityb = 4
blur_mean =0
blur_sd = 0.0001
matb, labelsb = csim.gen_uniform_points(intensityb, dim1 = dim1b, dim2 = dim2b, dim3 = dim3b,
                                   x_blur_mean= blur_mean, x_blur_sd = blur_sd,
                                   y_blur_mean= blur_mean, y_blur_sd = blur_sd,
                                   z_blur_mean= blur_mean, z_blur_sd = blur_sd)
print(len(labelsb))
pp3_b = csim.PointPattern3(matb, domain_b, labels = labelsb)


clust_pattern, centers, rads = csim.clustersim(opp = pp3_b,
           upp = pp3_a,
           pcp = 0.1,
           rho_c = 0.5,
           rho_b = 0.01,
           cr =4,
           rb = 0.2,
           cut = "buffered",
           buffer_factor= 1,
           selection='sampled',
           prob_function='constant',
                                )

#print(clust_pattern)
#print(clust_pattern.labels)
print(f"pcp is {(sum(clust_pattern.labels == 2) + sum(clust_pattern.labels == 3))/ clust_pattern.n_points}")
print(f" rho_c is {sum(clust_pattern.labels == 2) / (sum(clust_pattern.labels == 2) + sum(clust_pattern.labels == 1))}")
print(f" rho_b is {sum(clust_pattern.labels == 3)/ ((sum(clust_pattern.labels == 0)) + sum(clust_pattern.labels == 3))}")

probs = {0: 0.6,
         1: 0.6,
         2: 0.6,
         3: 0.6}
print(clust_pattern.n_points)
thinned_coords, thinned_labs = csim.thin_cluster(clust_pattern.coords, probs, labels = clust_pattern.labels, marks = (1, 2, 3, 4))
print(thinned_coords)
print(len(thinned_labs))
print(len(thinned_coords['x']))

thinned_coords = csim.thin_cluster(clust_pattern.coords, probs = 0.6)
print(len(thinned_coords['x']))