"""The guest-probability field: estimators, and the truth they are scored against.

    oracle          exact per-atom guest probabilities, by replaying the simulator
    baselines       kernel-smoothing baselines (constant, fixed and adaptive width)
    point_cloud     voxel grids and point selections from labelled point clouds
    density_grid    deprecated analytic grid target (see its module warning)
    physics         the Stage 5 screened diffusion field around precipitates
"""
