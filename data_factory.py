import clustersim as csim
import numpy as np

rng = np.random.default_rng(42)

# define domain size
dim1 = 60
dim2 = 60
dim3 = 60


domain = {'x': np.array([0.0, dim1]),
             'y': np.array([0.0, dim2]),
             'z': np.array([0.0, dim3])}
#define intensity (points per unit)
intensity = 1

# now opp dim - use smaller domain to save memory
dim1_opp = 30
dim2_opp = 30
dim3_opp = 30

domain_opp = {'x': np.array([0.0, dim1_opp]),
             'y': np.array([0.0, dim2_opp]),
             'z': np.array([0.0, dim3_opp])}

intensity_opp = 1

# number of simulations
n_sims = 1000
# overall percent - keep constant
pcp = 0.1

# uniformly vary rho_c, rho_b, cr, rb
rho_c_vec = rng.uniform(low = pcp*2, high = 1, size = n_sims)
rho_b_vec = rng.uniform(low = 0, high = pcp*0.5, size = n_sims)
cr_vec = rng.uniform(low = 3, high = 15, size = n_sims)
rb_vec = rng.uniform(low = 0, high = 0.5, size = n_sims)

# initialize empty matrix to hold calculated rho_c, rho_b, pcp, and their errors
pattern_stats = np.zeros(shape = [n_sims, 9])
save_prefix = "data/"
# make the data
for rho_c, rho_b, cr, rb, i in zip(rho_c_vec, rho_b_vec, cr_vec, rb_vec, range(n_sims)):
    # make opp
    print(i)
    mat, labels = csim.gen_rand_points(intensity = intensity,
                          dim1 = dim1,
                          dim2 = dim2,
                          dim3 = dim3)
    upp = csim.PointPattern3(mat, domain = domain, labels = labels)
    mat_opp, labels_opp = csim.gen_uniform_points(intensity = intensity_opp,
                                            dim1 = dim1_opp,
                                            dim2 = dim2_opp,
                                            dim3 = dim3_opp)
    opp = csim.PointPattern3(mat_opp, domain = domain_opp, labels = labels_opp)
    clust_pattern, rads, centers = csim.clustersim(opp=opp,
                                                   upp=upp,
                                                   pcp=pcp,
                                                   rho_c=rho_c,
                                                   rho_b=rho_b,
                                                   cr=cr,
                                                   rb=rb,
                                                   cut="buffered",
                                                   buffer_factor=1,
                                                   selection='sampled',
                                                   prob_function='Gaussian_decay')
    # save clusters
    name = save_prefix + "clust_pattern_" + str(i)
    np.savez(name,
             coords=clust_pattern.coords,
             domain = clust_pattern.domain,
             labels=clust_pattern.labels,
             radii=rads, centers=centers)

    pcp_clust = (sum(clust_pattern.labels == 2) + sum(clust_pattern.labels == 3)) / clust_pattern.n_points
    rho_c_clust = sum(clust_pattern.labels == 2) / (sum(clust_pattern.labels == 2) + sum(clust_pattern.labels == 1))
    rho_b_clust = sum(clust_pattern.labels == 3) / ((sum(clust_pattern.labels == 0)) + sum(clust_pattern.labels == 3))
    pcp_perc_error = (pcp - pcp_clust) / pcp
    rho_c_perc_error = (rho_c - rho_c_clust) / rho_c
    rho_b_perc_error = (rho_b - rho_b_clust) / rho_b
    pattern_stats[i, 0] = pcp_clust
    pattern_stats[i, 1] = pcp
    pattern_stats[i, 2] = pcp_perc_error
    pattern_stats[i, 3] = rho_c_clust
    pattern_stats[i, 4] = rho_c
    pattern_stats[i, 5] = rho_c_perc_error
    pattern_stats[i, 6] = rho_b_clust
    pattern_stats[i, 7] = rho_b
    pattern_stats[i, 8] = rho_b_perc_error

    print(f"pcp is {pcp_clust}, expected pcp is {pcp}, pcp percent error is {pcp_perc_error}")
    print(
        f" rho_c is {rho_c_clust}, expected rho_c is {rho_c}, rho_c percent error is {rho_c_perc_error}")
    print(
        f" rho_b is {rho_b_clust}, expected_rho_b is {rho_b}, rho_b percent error is {rho_b_perc_error}")
np.save(save_prefix + "pattern_stats", pattern_stats)
