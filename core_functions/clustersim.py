import numpy as np
import math


rng = np.random.default_rng(42)

class PointPattern3:
    def __init__(self, coords, domain, labels = None):
        self.coords = {
            "x": coords[:, 0].copy(),
            "y": coords[:, 1].copy(),
            "z": coords[:, 2].copy()
        }
        self.domain = {k: v.copy() for k, v in domain.items()}
        self.center = {"x": (self.domain['x'][1] + self.domain['x'][0])/2,
                       "y": (self.domain['y'][1] + self.domain['y'][0])/2,
                       "z": (self.domain['z'][1] + self.domain['z'][0])/2}
        self.volume = (self.domain['x'][1] - self.domain['x'][0]) *( self.domain['y'][1] - self.domain['y'][0]) * (self.domain['z'][1] - self.domain['z'][0])
        self.n_points = len(self.coords['x'])
        self.labels = labels.copy() if labels is not None else np.zeros(len(coords), dtype = np.int8)

    def update_properties(self):
        # Helper to ensure center and volume are always current
        self.center = {
            "x": (self.domain['x'][1] + self.domain['x'][0]) / 2,
            "y": (self.domain['y'][1] + self.domain['y'][0]) / 2,
            "z": (self.domain['z'][1] + self.domain['z'][0]) / 2
        }
        self.volume = (self.domain['x'][1] - self.domain['x'][0]) * \
                      (self.domain['y'][1] - self.domain['y'][0]) * \
                      (self.domain['z'][1] - self.domain['z'][0])
        self.n_points = len(self.coords['x'])

    def scale(self, scale_x, scale_y, scale_z):
        self.domain['x'] = self.domain['x'] * scale_x
        self.domain['y'] = self.domain['y'] * scale_y
        self.domain['z'] = self.domain['z'] * scale_z
        self.coords['x'] = self.coords['x'] * scale_x
        self.coords['y'] = self.coords['y'] * scale_y
        self.coords['z'] = self.coords['z'] * scale_z
        self.update_properties()
        return self

    # method for adding buffer to domain
    def expand_domain(self, expand_x, expand_y, expand_z):
        self.domain['x'][0] =self.domain['x'][0] - expand_x
        self.domain['x'][1] =self.domain['x'][1] + expand_x
        self.domain['y'][0] =self.domain['y'][0] - expand_y
        self.domain['y'][1] =self.domain['y'][1] + expand_y
        self.domain['z'][0] =self.domain['z'][0] - expand_z
        self.domain['z'][1] =self.domain['z'][1] + expand_z
        self.update_properties()
        return self

    def shift(self, shift_x, shift_y, shift_z):
        self.coords['x'] = self.coords['x'] + shift_x
        self.coords['y'] = self.coords['y'] + shift_y
        self.coords['z'] = self.coords['z'] + shift_z
        self.domain['x'] = self.domain['x'] + shift_x
        self.domain['y'] = self.domain['y'] + shift_y
        self.domain['z'] = self.domain['z'] + shift_z
        self.update_properties()
        return self

    def chop(self, win, chop_domain = False):
        keep = ((self.coords['x'] >= win['x'][0]) &
                (self.coords['x'] <= win['x'][1]) &
                (self.coords['y'] >= win['y'][0])&
                (self.coords['y'] <= win['y'][1]) &
                (self.coords['z'] >= win['z'][0]) &
                (self.coords['z'] <= win['z'][1]))
        self.coords['x'] = self.coords['x'][keep]
        self.coords['y'] = self.coords['y'][keep]
        self.coords['z'] = self.coords['z'][keep]
        self.labels = self.labels[keep]

        if chop_domain:
            self.domain = win
        self.update_properties()

        return self

    def scale_domain_only(self, scale_x, scale_y, scale_z):
        self.domain['x']= self.domain['x']* scale_x
        self.domain['y']= self.domain['y']* scale_y
        self.domain['z']= self.domain['z']* scale_z
        self.update_properties()

        return self

    def scale_coords_only(self, scale_x, scale_y, scale_z):
        self.coords['x']= self.coords['x']* scale_x
        self.coords['y']= self.coords['y']* scale_y
        self.coords['z']= self.coords['z']* scale_z
        self.update_properties()

        return self

    def shift_coords_only(self, shift_x, shift_y, shift_z):
        self.coords['x']= self.coords['x']+ shift_x
        self.coords['y']= self.coords['y']+ shift_y
        self.coords['z']= self.coords['z']+ shift_z
        self.update_properties()

        return self

    def shift_domain_only(self, shift_x, shift_y, shift_z):
        self.domain['x']= self.domain['x']+ shift_x
        self.domain['y']= self.domain['y']+ shift_y
        self.domain['z']= self.domain['z']+ shift_z
        self.update_properties()

        return self

# function to create random 3d points
def gen_rand_points(intensity = 1, dim1 = 10, dim2 = 10, dim3 = 10):
    # define dimension lengths
    # domain is a rectangular prism, but not necessarily cubic
    vol = dim1 * dim2 * dim3
    n_points = intensity * vol
    s = rng.uniform(low=0, high=1, size=(n_points, 3))
    s = s * [dim1, dim2, dim3]
    labels = np.zeros(shape = len(s), dtype = np.int8)
    return s, labels

# function to create uniform 3d points
def gen_uniform_points(intensity = 1, dim1 = 10, dim2 = 10, dim3 = 10,
                       x_blur_mean = 0, x_blur_sd =0,
                       y_blur_mean = 0, y_blur_sd =0,
                       z_blur_mean = 0, z_blur_sd =0):
    vol = dim1 * dim2 * dim3
    n_points = intensity * vol
    x_vals = np.linspace(0, dim1, int(round(n_points**(1/3))))
    y_vals = np.linspace(0, dim2, int(round(n_points**(1/3))))
    z_vals = np.linspace(0, dim3, int(round(n_points**(1/3))))

    # Calculate number of points along each dimension
    # (Assuming a roughly cubic aspect ratio for simplicity,
    # but we can do this per-dimension if needed)
    n_side = int(round(n_points ** (1 / 3)))

    # Determine step size
    dx = dim1 / n_side
    dy = dim2 / n_side
    dz = dim3 / n_side

    # Generate Cell-Centered coordinates
    # (0.5 + i) * dx ensures we start at half a step into the domain
    x_vals = (np.arange(n_side) + 0.5) * dx
    y_vals = (np.arange(n_side) + 0.5) * dy
    z_vals = (np.arange(n_side) + 0.5) * dz


    xyz = np.array([(x, y, z) for x in x_vals for y in y_vals for z in z_vals])

    # create blurs fore each point
    x_blur = rng.normal(loc=x_blur_mean, scale=x_blur_sd, size=len(xyz[:,0]))
    y_blur = rng.normal(loc=y_blur_mean, scale=y_blur_sd, size=len(xyz[:,1]))
    z_blur = rng.normal(loc=z_blur_mean, scale=z_blur_sd, size=len(xyz[:,2]))

    # add blur values to coordinates
    xyz[:,0] = xyz[:,0] + x_blur
    xyz[:,1] = xyz[:,1] + y_blur
    xyz[:,2] = xyz[:,2] + z_blur
    labels = np.zeros(shape = len(xyz), dtype = np.int8)

    return xyz, labels
def gen_back_guest(labels, back_conc, assign_val = 1):
    # for each point in labels, assign it a random value between 0 and 1
    vals = rng.uniform(low=0, high=1, size=len(labels))
    # assign numbers less than back_conc as guest (1)
    labels[vals < back_conc] = assign_val
    return labels

def calc_weighted_dist(x, y, z, center, x_weight = 1, y_weight = 1, z_weight = 1, x_exp = 2, y_exp = 2, z_exp = 2):
    weighted_dist = (((x_weight*(x - center[0]))**x_exp) + ((y_weight*(y - center[1]))**y_exp) +  ((z_weight*(z - center[2]))**z_exp)) **(1/2)
    return weighted_dist

def assign_clust_points(dist, weighted_dist,
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
    labels = np.zeros(shape = len(dist), dtype = np.int8)
    # if either r_max or r_weighted_max aren't to be used, assign them to the max value
    if r_max is None:
        r_max = dist.max()
    if r_weighted_max is None:
        r_weighted_max = weighted_dist.max()

    # assign all labels that fall within r_max & r_weighted_max
    labels[(dist <= r_max) & (weighted_dist <= r_weighted_max)] = 1
    inside_inds = np.where(labels == 1)[0]
    if rho_c:
        n_points = int(round(len(inside_inds)* rho_c,0))


    # if npoints is greater than number of points inside radius
    if n_points >= len(inside_inds):
        # mark all points as guest type (2)
        labels = labels *2
        return labels
    else:
        # assign probability based on weighted distance
        # if all weighted_dists are 0, then make them all 1 so that they are all still equal but
        # we also get a nonzero value when multiplying by labels
        if weighted_dist.sum() == 0:
            weighted_dist = np.ones_like(weighted_dist)

        candidate_distances = weighted_dist[inside_inds]
        r_weighted_max = candidate_distances.max() # new line
        # now we can assign points as guests based on probability
        # Option A: the npoints highest probs
        if selection == "highest":
            inds = np.argpartition(candidate_distances, 1*n_points)[:n_points]
            clust_points = inside_inds[inds]

        # Option B: assign probability based on weighted distance, sample with that probability
        elif selection == "sampled":
            if prob_function == "Gaussian_decay":
                probs = (1 - candidate_distances / r_weighted_max) * labels[inside_inds]
            elif prob_function == "inverse":
                probs = labels[inside_inds] / (candidate_distances + 1e-10)
            elif prob_function == "growth":
                probs = labels[inside_inds] * candidate_distances
            elif prob_function == "constant":
                probs = np.ones(len(candidate_distances), dtype=np.float64)
            elif prob_function == "exponential":
                # Sharpens the difference between low and high weighted distances
                probs = np.exp(-1*prob_exp * (candidate_distances / r_weighted_max)) * labels[inside_inds]
            else:
                raise ValueError(
                    "Probability function not recognized: must be either Gaussian_decay, inverse, growth, exponential, or constant. Gaussian_decay is standard")
            # if there is only one element in probs and it is 0, then make that equal to 1
            if sum(probs) == 0:
                probs = probs + 1
            probs = probs / sum(probs)
            inds = rng.choice(inside_inds, size=n_points, replace=False, p=probs)
            clust_points = inds

        else:
            raise ValueError("selection must either be sampled or highest")

        # inds are the indices of inside_inds


        #final_labels = np.zeros(len(dist), dtype = np.int8)
        final_labels = labels
        final_labels[clust_points] = 2
        return final_labels


def estimate_cluster_volume(r, weights, exponents, r_max_weighted_ratio, n_samples=50000):
    """
    Estimates the volume of a single cluster using Monte Carlo integration.
    This handles the complex intersection of the sphere and the weighted shape.
    """
    # 1. Determine the bounding box for the simulation
    # The cluster is bounded by the strict r_max (sphere radius)
    # We add a tiny buffer to be safe
    box_lim = r * 1.1
    box_vol = (2 * box_lim) ** 3

    # 2. Generate random points within this box (centered at 0,0,0)
    # Using uniform random points [-box_lim, box_lim]
    pts = np.random.uniform(low=-box_lim, high=box_lim, size=(n_samples, 3))

    # 3. Calculate Euclidean Distance (for r_max check)
    dists = np.sqrt(np.sum(pts ** 2, axis=1))

    # 4. Calculate Weighted Distance (for r_weighted_max check)
    # We use the weights and exponents passed in
    weighted_dists = (
                             (weights['x'] * pts[:, 0]) ** exponents['x'] +
                             (weights['y'] * pts[:, 1]) ** exponents['y'] +
                             (weights['z'] * pts[:, 2]) ** exponents['z']
                     ) ** (1 / 2)  # Assuming root 2 for distance normalization

    # Determine the weighted max threshold
    # In your code: current_r_weighted_max = r * r_max_weighted_max_ratio
    # If r_max_weighted_max_ratio is None, you might be using a fixed value,
    # but usually it scales with r.
    r_w_max = r * (r_max_weighted_ratio if r_max_weighted_ratio else 1.0)

    # 5. Check which points are "Inside"
    # Must satisfy BOTH the sphere radius AND the weighted radius
    inside_mask = (dists <= r) & (weighted_dists <= r_w_max)

    # 6. Calculate Volume fraction
    fraction_inside = np.sum(inside_mask) / n_samples
    estimated_volume = box_vol * fraction_inside

    return estimated_volume

def clustersim(opp, # overlying point pattern
               upp, # underlying point pattern
               pcp,  # overall clustering type concentration
               #intensity, # overall intensity of all points
               rho_c, # guest concentration inside clusters
               rho_b, # guest concentration in background
               cr, # cluster radius mean
               rb, # cluster radius std
               cut = "buffered",
               buffer_factor = 2.5,
               x_weight = 1,
               y_weight = 1,
               z_weight = 1,
               x_exp = 2,
               y_exp = 2,
               z_exp =2,
               r_weighted_max=None,
               r_max_weighted_max_ratio = None,
               prob_function="Gaussian_decay",
               prob_exp = -3,
               selection='sampled'):
    # Prepare dictionaries for the helper function
    weights_dict = {'x': x_weight, 'y': y_weight, 'z': z_weight}
    exps_dict = {'x': x_exp, 'y': y_exp, 'z': z_exp}

    # distinct step: Calculate the volume of a SINGLE mean cluster
    single_cluster_vol = estimate_cluster_volume(
        r=cr,
        weights=weights_dict,
        exponents=exps_dict,
        r_max_weighted_ratio=r_max_weighted_max_ratio
    )

    # Apply the statistical correction for the radius distribution (rb)
    # Volume scales with r^3, so E[r^3] = E[r]^3 * (1 + 3*(sigma/mu)^2)
    sigma = cr * rb #(standard deviation)
    # CV = sigma/mu = rb
    correction_factor = (1 + 3 * rb ** 2)

    mean_clust_volume = single_cluster_vol * correction_factor

    print(f"Estimated Mean Cluster Volume: {mean_clust_volume:.4f}")

    # Recalculate how many clusters we need to hit the target PCP
    v_clusters_needed = upp.volume * (pcp - rho_b) / (rho_c - rho_b)

    # This will now give you a much higher number if your shapes are "skinny"
    n_clusts = int(round(v_clusters_needed / mean_clust_volume))

    print(f"Corrected N_Clusts: {n_clusts}")


    # 2. Adjust Grid Density for Quantization
    # We can't fit 118 points on a grid. We must fit 125 (5^3).
    # If we don't adjust, the spacing is too wide and the 5th row falls off the edge.
    n_side = math.ceil(n_clusts ** (1 / 3))
    n_grid_target = n_side ** 3
    ## cut everything in the scaled opp that isn't within the domain of the original + buffer
    if cut == "buffered":
        buff = buffer_factor * cr
    elif cut == "strict":
        buff = 0
    else:
        raise ValueError("cut must be either 'buffered' or 'strict'")
    # create window to chop opp to
    buffed_window = {
        'x': [upp.domain['x'][0] + buff, upp.domain['x'][1] - buff],
        'y': [upp.domain['y'][0] + buff, upp.domain['y'][1] - buff],
        'z': [upp.domain['z'][0] + buff, upp.domain['z'][1] - buff]
    }

    buffed_volume = (buffed_window['x'][1] - buffed_window['x'][0]) *(buffed_window['y'][1] - buffed_window['y'][0]) * (buffed_window['z'][1] - buffed_window['z'][0])
    needed_opp_concentration = n_grid_target /buffed_volume

    old_opp_concentration = opp.n_points / opp.volume
    # now we know the number of points needed in opp and its domain
    # get number of clusters needed
    # find final volume needed for opp
    v_opp_new = opp.n_points / needed_opp_concentration
    # scale factor for going from old to new opp volume
    opp_scale_factor = (old_opp_concentration**(1/3)) / (needed_opp_concentration ** (1/3))
    print(f"Scaling Factor: {opp_scale_factor}")


    # multiply all opp coordinates by scale factor
    opp_scaled = opp.scale(opp_scale_factor, opp_scale_factor, opp_scale_factor)
    # shift opp domain and coordinates so that they have the same center as upp
    # calculate distances between centers
    shift = {'x':upp.center['x'] - opp.center['x'],
             'y':upp.center['y'] - opp.center['y'],
             'z':upp.center['z'] - opp.center['z']}

    # shift the scaled opp so that it lines up with the upp
    opp_scaled_shifted = opp_scaled.shift(shift['x'], shift['y'], shift['z'])

    # chop opp to the window
    opp_chopped = opp_scaled_shifted.chop(buffed_window, chop_domain=True)

    current_n = opp_chopped.n_points
    if current_n > n_clusts:
        print(f"Oversampling: reducing {current_n} grid points to {n_clusts}")
        keep_inds = rng.choice(np.arange(current_n), size=n_clusts, replace=False)

        opp_chopped.coords['x'] = opp_chopped.coords['x'][keep_inds]
        opp_chopped.coords['y'] = opp_chopped.coords['y'][keep_inds]
        opp_chopped.coords['z'] = opp_chopped.coords['z'][keep_inds]
        opp_chopped.labels = opp_chopped.labels[keep_inds]
        opp_chopped.n_points = n_clusts
    elif current_n < n_clusts:
        print(f"Warning: Still under-sampled ({current_n} < {n_clusts}). Check domain padding.")

    cluster_centers = opp_chopped.coords
    # now generate list of cluster center radii
    cr_all = rng.normal(loc = cr, scale = sigma, size = len(cluster_centers['x']))
    # set all numbers less than 0 to 0
    cr_all[cr_all < 0] = 0

    # now lets generate some clusters!
    updated_upp_labels = np.zeros(upp.n_points, dtype=np.int8)
    for rad, center_x, center_y, center_z in zip(enumerate(cr_all), cluster_centers['x'], cluster_centers['y'], cluster_centers['z']):
        i = rad[0]
        r = rad[1]
        dists = ((((upp.coords['x'] - center_x)**2) +((upp.coords['y'] - center_y)**2) +((upp.coords['z'] - center_z)**2))**(1/2))
        weighted_dists = calc_weighted_dist(upp.coords['x'], upp.coords['y'], upp.coords['z'],
                                            center = np.array([center_x, center_y, center_z]),
                                            x_weight = x_weight,
                                            y_weight = y_weight,
                                            z_weight = z_weight,
                                            x_exp = x_exp,
                                            y_exp = y_exp,
                                            z_exp = z_exp)
        current_r_weighted_max = r_weighted_max  # Start with the user-provided global override
        if r_max_weighted_max_ratio and not r_weighted_max:
            current_r_weighted_max = r * r_max_weighted_max_ratio
        new_labels = assign_clust_points(dist = dists,
                                         weighted_dist=weighted_dists,
                                         rho_c = rho_c,
                                         r_max=r,
                                         r_weighted_max=current_r_weighted_max,
                                         prob_function=prob_function,
                                         prob_exp=prob_exp,
                                         selection=selection
                                         )
        updated_upp_labels = np.maximum(updated_upp_labels, new_labels)
    upp.labels = updated_upp_labels
    background_inds = np.where(updated_upp_labels == 0)[0]
    background_labels = gen_back_guest(updated_upp_labels[background_inds], rho_b, assign_val=3)
    upp.labels[background_inds] = background_labels
    # now we need to assign background points

    return upp, cr_all, cluster_centers

def thin_cluster(coords, probs, labels = None, marks = "all"):
    if marks == "all":
        rolls = rng.uniform(low = 0, high = 1, size = len(coords['x']))
        mask = rolls <= probs
        new_coords = {key:value[mask] for key, value in coords.items()}
        new_labels = labels[mask]
        return new_coords, new_labels
    else:
        # create a vector of length labels that has the right probability for each label
        #point_probs =[probs[item] for item in labels]
        # Faster mapping inside thin_cluster
        v_lookup = np.vectorize(probs.get)
        point_probs = v_lookup(labels)

        rolls = rng.uniform(low = 0, high = 1, size = len(point_probs))
        mask = point_probs >= rolls
        new_coords = {key:value[mask] for key, value in coords.items()}
        new_labels = labels[mask]
        return new_coords, new_labels


