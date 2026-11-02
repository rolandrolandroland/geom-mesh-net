"""Deprecated aliases for the package's modules under their old names.

The modules moved into subpackages. Each old name still imports, warns with a
DeprecationWarning, and resolves to the *same* module object as the new name, so
module state such as clustersim's random generator is shared:

    core_functions.clustersim                  simulation.clustersim
    core_functions.paper_spatial_features      statistics.paper_spatial_features
    core_functions.spatial_stats_01            statistics.barcode
    core_functions.paper_feature_experiments   statistics.feature_cache
    core_functions.field_oracle                fields.oracle
    core_functions.field_baselines             fields.baselines
    core_functions.point_cloud_fields          fields.point_cloud
    core_functions.voxelize_clusters           fields.density_grid
    core_functions.data_loader                 neural (datasets and models)
    core_functions.cluster_visualizer          viz.clusters
    core_functions.compare_benchmarks          viz.benchmarks
"""
