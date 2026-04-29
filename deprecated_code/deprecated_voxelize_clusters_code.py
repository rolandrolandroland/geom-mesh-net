if overlap_prob == "highest":
    full_upp_probs = np.maximum(full_upp_probs, new_densities)
elif overlap_prob == "lowest":
    # get overlapping indices
    overlap = (new_densities > 0) & (full_upp_probs > 0)
    # if they overlap, assign to minimum
    full_upp_probs[overlap] = np.minimum(full_upp_probs[overlap], new_densities[overlap])
    # if they don't overlap, assign to max
    full_upp_probs[~overlap] = np.maximum(full_upp_probs[~overlap], new_densities[~overlap])

elif overlap_prob == "combined":
    full_upp_probs = 1 - ((1 - full_upp_probs) * (1 - new_densities))
else:
    raise ValueError("overlap_prob must be either highest, lowest, or combined")


## Below is the original. it uses more memory because each operation in the
# for loop operates on the entire grid. If i want to go back to a smooth drop
# off outside of clusters though, this will be essential

import numpy as np
import clustersim as csim


# function to assign the probability distribution inside of clusters
def assign_clust_probs(dist, weighted_dist, density_grid,
                        rho_c = None,
                        n_points = None,
                        r_max = None, r_weighted_max = None,
                        prob_function = "Gaussian_decay",
                        prob_exp = 3,
                        selection = "sampled"):
    if n_points is None and rho_c is None:
        raise ValueError("Either n_points or rho_c must be provided")
    if n_points and rho_c:
        raise ValueError("Only n_points or rho_c can be provided")
    else:
        pass
    labels = np.zeros_like(dist, dtype = np.int8)

    # if either r_max or r_weighted_max aren't to be used, assign them to the max value
    if r_max is None:
        r_max = dist.max()
    if r_weighted_max is None:
        r_weighted_max = weighted_dist.max()

    # assign all labels that fall within r_max & r_weighted_max
    labels[(dist <= r_max) & (weighted_dist <= r_weighted_max)] = 1
    candidate_weighted_distances = weighted_dist[labels == 1]
    if n_points == 0 or rho_c == 0:
        return density_grid

    if rho_c is not None:
        n_points = int(round((labels == 1).sum()* rho_c,0))


    # if npoints is greater than number of points inside radius
    if n_points >= np.sum(labels == 1):
        # mark all points as guest type (2)
        density_grid[labels == 1] = 1
        return density_grid
    else:
        # assign probability based on weighted distance
        # if all weighted_dists are 0, then make them all 1 so that they are all still equal but
        # we also get a nonzero value when multiplying by labels
        if weighted_dist.sum() == 0:
            weighted_dist = np.ones_like(weighted_dist)

        r_weighted_max = candidate_weighted_distances.max()
        # now we can assign points as guests based on probability
        # Option A: the npoints highest probs
        if selection == "highest":
            inds = np.argpartition(candidate_weighted_distances, 1*n_points)[:n_points]
            temp = np.zeros_like(candidate_weighted_distances)
            temp[inds] = 1
            density_grid[labels == 1] = temp


        # Option B: assign probability based on weighted distance, sample with that probability
        elif selection == "sampled":
            if prob_function == "Gaussian_decay":
                probs = (1 - candidate_weighted_distances / r_weighted_max)
            elif prob_function == "inverse":
                probs = 1 / (candidate_weighted_distances + 1e-10)
            elif prob_function == "growth":
                probs = 1 * candidate_weighted_distances
            elif prob_function == "constant":
                probs = np.ones(len(candidate_weighted_distances), dtype=np.float64)
            elif prob_function == "exponential":
                # Sharpens the difference between low and high weighted distances
                probs = np.exp(-1*prob_exp * (candidate_weighted_distances / r_weighted_max))
            else:
                raise ValueError(
                    "Probability function not recognized: must be either Gaussian_decay, inverse, growth, exponential, or constant. Gaussian_decay is standard")
            probs = probs / sum(probs)
            density_grid[labels==1] = probs


        else:
            raise ValueError("selection must either be sampled or highest")

        # inds are the indices of inside_inds



        return density_grid

def generate_density_grid(grid_size, cluster_centers, radii,
                          rho_c,  # guest concentration inside clusters
                          rho_b,
                          n_points = None,
                          resolution=1,
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
                          overlap_prob = "highest"):

    # axes for each dimension
    x_grid = np.linspace(resolution / 2, grid_size[0] - (resolution / 2), int(grid_size[0] / resolution))
    y_grid = np.linspace(resolution / 2, grid_size[1] - (resolution / 2), int(grid_size[1] / resolution))
    z_grid = np.linspace(resolution / 2, grid_size[2] - (resolution / 2), int(grid_size[2] / resolution))

    # create grid for coordinates
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing = 'ij')

    full_upp_probs = np.ones((len(x_grid), len(y_grid), len(z_grid))) *rho_b
    for rad, center_x, center_y, center_z in zip(enumerate(radii), cluster_centers['x'], cluster_centers['y'], cluster_centers['z']):
        i = rad[0]
        r = rad[1]
        dists = ((((xx - center_x)**2) +((yy - center_y)**2) +((zz - center_z)**2))**(1/2))
        weighted_dists = csim.calc_weighted_dist(xx, yy, zz,
                                                 center = np.array([center_x, center_y, center_z]),
                                                 x_weight=x_weight,
                                                 y_weight=y_weight,
                                                 z_weight=z_weight,
                                                 x_exp=x_exp,
                                                 y_exp=y_exp,
                                                 z_exp=z_exp)
        current_r_weighted_max = r_weighted_max  # Start with the user-provided global override
        # create grid to hold probabilities
        density_grid = np.zeros((len(x_grid), len(y_grid), len(z_grid)))
        if r_max_weighted_max_ratio and not r_weighted_max:
            current_r_weighted_max = r * r_max_weighted_max_ratio
        new_densities = assign_clust_probs(dist=dists,
                                           weighted_dist=weighted_dists,
                                           density_grid=density_grid,
                                           rho_c=rho_c,
                                           n_points = n_points,
                                           r_max=r,
                                           r_weighted_max=current_r_weighted_max,
                                           prob_function=prob_function,
                                           prob_exp=prob_exp,
                                           selection=selection
                                           )
        if overlap_prob == "highest":
            full_upp_probs = np.maximum(full_upp_probs, new_densities)
        elif overlap_prob == "combined":
            full_upp_probs = 1-((1-full_upp_probs) * (1-new_densities))
        else:
            raise ValueError("overlap_prob must be either highest, lowest, or combined")


    return xx, yy, zz, full_upp_probs


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
#print(len(labelsb))
pp3_centers = csim.PointPattern3(matb, domain_b, labels = labelsb)
#print(pp3_centers.coords)
cr = [3, 4, 2, 2, 1, 2, 1, 3]
out = generate_density_grid(grid_size, cluster_centers = pp3_centers.coords, radii = cr, resolution = 0.5,
                            rho_c = 0.5, rho_b = 0.01)

########################################################################
########################################################################


import numpy as np
import clustersim as csim


# function to assign the probability distribution inside of clusters
def assign_clust_probs(dist, weighted_dist, density_grid,
                        rho_c = None,
                        n_points = None,
                        r_max = None, r_weighted_max = None,
                        prob_function = "Gaussian_decay",
                        prob_exp = 3,
                        selection = "sampled"):
    if n_points is None and rho_c is None:
        raise ValueError("Either n_points or rho_c must be provided")
    if n_points and rho_c:
        raise ValueError("Only n_points or rho_c can be provided")
    else:
        pass
    labels = np.zeros_like(dist, dtype = np.int8)

    # if either r_max or r_weighted_max aren't to be used, assign them to the max value
    if r_max is None:
        r_max = dist.max()
    if r_weighted_max is None:
        r_weighted_max = weighted_dist.max()

    # assign all labels that fall within r_max & r_weighted_max
    labels[(dist <= r_max) & (weighted_dist <= r_weighted_max)] = 1
    candidate_weighted_distances = weighted_dist[labels == 1]
    if n_points == 0 or rho_c == 0:
        return density_grid

    if rho_c is not None:
        n_points = int(round((labels == 1).sum()* rho_c,0))


    # if npoints is greater than number of points inside radius
    if n_points >= np.sum(labels == 1):
        # mark all points as guest type (2)
        density_grid[labels == 1] = 1
        return density_grid
    else:
        # assign probability based on weighted distance
        # if all weighted_dists are 0, then make them all 1 so that they are all still equal but
        # we also get a nonzero value when multiplying by labels
        if weighted_dist.sum() == 0:
            weighted_dist = np.ones_like(weighted_dist)

        r_weighted_max = candidate_weighted_distances.max()
        # now we can assign points as guests based on probability
        # Option A: the npoints highest probs
        if selection == "highest":
            inds = np.argpartition(candidate_weighted_distances, 1*n_points)[:n_points]
            temp = np.zeros_like(candidate_weighted_distances)
            temp[inds] = 1
            density_grid[labels == 1] = temp


        # Option B: assign probability based on weighted distance, sample with that probability
        elif selection == "sampled":
            if prob_function == "Gaussian_decay":
                probs = (1 - candidate_weighted_distances / r_weighted_max)
            elif prob_function == "inverse":
                probs = 1 / (candidate_weighted_distances + 1e-10)
            elif prob_function == "growth":
                probs = 1 * candidate_weighted_distances
            elif prob_function == "constant":
                probs = np.ones(len(candidate_weighted_distances), dtype=np.float64)
            elif prob_function == "exponential":
                # Sharpens the difference between low and high weighted distances
                probs = np.exp(-1*prob_exp * (candidate_weighted_distances / r_weighted_max))
            else:
                raise ValueError(
                    "Probability function not recognized: must be either Gaussian_decay, inverse, growth, exponential, or constant. Gaussian_decay is standard")
            probs = probs / sum(probs)
            density_grid[labels==1] = probs


        else:
            raise ValueError("selection must either be sampled or highest")

        # inds are the indices of inside_inds



        return density_grid

def generate_density_grid(grid_size, cluster_centers, radii,
                          rho_c,  # guest concentration inside clusters
                          rho_b,
                          n_points = None,
                          resolution=1,
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
                          overlap_prob = "highest"):

    # axes for each dimension
    x_grid = np.linspace(resolution / 2, grid_size[0] - (resolution / 2), int(grid_size[0] / resolution))
    y_grid = np.linspace(resolution / 2, grid_size[1] - (resolution / 2), int(grid_size[1] / resolution))
    z_grid = np.linspace(resolution / 2, grid_size[2] - (resolution / 2), int(grid_size[2] / resolution))

    # create grid for coordinates
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing = 'ij')

    full_upp_probs = np.ones((len(x_grid), len(y_grid), len(z_grid))) *rho_b
    for rad, center_x, center_y, center_z in zip(enumerate(radii), cluster_centers['x'], cluster_centers['y'], cluster_centers['z']):
        i = rad[0]
        r = rad[1]
        dists = ((((xx - center_x)**2) +((yy - center_y)**2) +((zz - center_z)**2))**(1/2))
        weighted_dists = csim.calc_weighted_dist(xx, yy, zz,
                                                 center = np.array([center_x, center_y, center_z]),
                                                 x_weight=x_weight,
                                                 y_weight=y_weight,
                                                 z_weight=z_weight,
                                                 x_exp=x_exp,
                                                 y_exp=y_exp,
                                                 z_exp=z_exp)
        current_r_weighted_max = r_weighted_max  # Start with the user-provided global override
        # create grid to hold probabilities
        density_grid = np.zeros((len(x_grid), len(y_grid), len(z_grid)))
        if r_max_weighted_max_ratio and not r_weighted_max:
            current_r_weighted_max = r * r_max_weighted_max_ratio
        new_densities = assign_clust_probs(dist=dists,
                                           weighted_dist=weighted_dists,
                                           density_grid=density_grid,
                                           rho_c=rho_c,
                                           n_points = n_points,
                                           r_max=r,
                                           r_weighted_max=current_r_weighted_max,
                                           prob_function=prob_function,
                                           prob_exp=prob_exp,
                                           selection=selection
                                           )
        if overlap_prob == "highest":
            full_upp_probs = np.maximum(full_upp_probs, new_densities)
        elif overlap_prob == "combined":
            full_upp_probs = 1-((1-full_upp_probs) * (1-new_densities))
        else:
            raise ValueError("overlap_prob must be either highest, lowest, or combined")


    return xx, yy, zz, full_upp_probs


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
#print(len(labelsb))
pp3_centers = csim.PointPattern3(matb, domain_b, labels = labelsb)
#print(pp3_centers.coords)
cr = [3, 4, 2, 2, 1, 2, 1, 3]
out = generate_density_grid(grid_size, cluster_centers = pp3_centers.coords, radii = cr, resolution = 0.5,
                            rho_c = 0.5, rho_b = 0.01)


########################################################################
########################################################################

