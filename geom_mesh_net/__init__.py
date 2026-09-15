"""Simulation, spatial statistics and neural reconstruction for 3D clustered point patterns.

Subpackages:

    simulation    clustersim, the parameters it is run with, and their prior
    statistics    the 14 global spatial-summary features and their presets
    fields        guest-probability fields: the exact oracle, kernel baselines, grids
    neural        coordinate networks and the dataset that trains them
    inference     the conditional normalizing flow and its calibration checks
    viz           plotting helpers (imports pyvista and plotly on use)
    paths         where data, experiments and figures live
"""
