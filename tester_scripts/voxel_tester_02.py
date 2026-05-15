from geom_mesh_net.core_functions import voxelize_clusters as vc
from geom_mesh_net.core_functions import clustersim as csim
import numpy as np
grid_size = (10, 10, 10)



domain_b = {'x': np.array([0.0, grid_size[0]]),
            'y': np.array([0.0, grid_size[1]]),
            'z': np.array([0.0, grid_size[2]])}

#print(test)
intensityb = 0.01
blur_mean =0
blur_sd = 0.0001
matb, labelsb = csim.gen_uniform_points(intensityb, dim1 = grid_size[0], dim2 = grid_size[1], dim3 = grid_size[2],
                                   x_blur_mean= blur_mean, x_blur_sd = blur_sd,
                                   y_blur_mean= blur_mean, y_blur_sd = blur_sd,
                                   z_blur_mean= blur_mean, z_blur_sd = blur_sd)
print(len(labelsb))
pp3_centers = csim.PointPattern3(matb, domain_b, labels = labelsb)
print(pp3_centers.coords)
cr = [3, 4, 2, 2, 1, 2, 1, 3]
out = vc.generate_density_grid(grid_size, cluster_centers = pp3_centers.coords, radii = cr, resolution = 0.5,
                            rho_c = 0.5, rho_b = 0.01)

print(out)