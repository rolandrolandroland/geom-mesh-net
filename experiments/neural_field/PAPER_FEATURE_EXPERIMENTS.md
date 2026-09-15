# Paper Feature Neural-Field Experiments

`train_networks_for_compare_multi_pattern_02.py` implements the complete
staged experiment pipeline for the 14 Bennett et al. spatial features.

## Experiment stages

| Stage | Scope | Inputs |
| --- | --- | --- |
| `coordinate_per_pattern` | One model per pattern | Normalized x, y, z |
| `global_per_pattern` | One model per pattern | Coordinates and 14 global features |
| `coordinate_shared` | One model across patterns | Coordinates |
| `global_shared` | One model across patterns | Coordinates and 14 global features |
| `local_per_pattern` | One model per pattern | Coordinates and 14 local features |
| `global_local_per_pattern` | One model per pattern | Coordinates, global features, and local features |

The same neural-network architecture is used across stages so differences are
attributable to the input features rather than network depth or width.

## Null models

Use `--null-models csr,random_label` to run both comparisons.

- `csr` uses the analytical three-dimensional homogeneous-Poisson values for
  G, F, K, and guest-to-host cross-G.
- `random_label` preserves every measured point and the global guest count. A
  complete global relabeling is generated first, and each local window then
  receives the labels belonging to its points. Guest counts are not fixed
  independently inside local windows.

Local CSR features use the global guest and host intensities by default. This
retains local enrichment and depletion as signal. Set
`--local-csr-intensity-scope local` to condition on each window's local
intensities instead.

## Offline caching

Feature calculation and neural-field training are separate commands:

```bash
PYTHONPATH=geom_mesh_net python \
  geom_mesh_net/example_01/train_networks_for_compare_multi_pattern_02.py \
  cache --patterns 0-4

PYTHONPATH=geom_mesh_net python \
  geom_mesh_net/example_01/train_networks_for_compare_multi_pattern_02.py \
  train --patterns 0-4
```

The `all` command builds every requested cache before starting any model:

```bash
PYTHONPATH=geom_mesh_net python \
  geom_mesh_net/example_01/train_networks_for_compare_multi_pattern_02.py \
  all --patterns 0-4
```

The training phase reads only `.npz` cache files. It never reloads point
patterns, relabels points, calculates summary functions, or voxelizes targets
inside an epoch.

The defaults use 512 sampled voxels and 19 global relabelings for development.
A larger experiment can use, for example:

```bash
PYTHONPATH=geom_mesh_net python \
  geom_mesh_net/example_01/train_networks_for_compare_multi_pattern_02.py \
  all --patterns 0-9 --voxel-samples 4096 --relabelings 99
```

Use `--voxel-samples all` only after benchmarking local random-label feature
calculation. Calculating 14 local features at all 1,728,000 voxels in a pattern
is substantially more expensive than sampled-voxel experiments.

## Data defaults

- Features are calculated from a deterministic 10% thinning of the measured
  point cloud.
- Targets are calculated from the original point cloud.
- Guest marks are 2 and 3.
- The local window is a clipped cube extending 12 physical units from each
  sampled voxel center in every coordinate direction.
- Global radii are G/F 10, K 70, and cross-G 8.
- Local radii are G/F 8, K 8, and cross-G 3.

Guest-free or single-guest local windows remain in the experiment. Their
unavailable guest G and K curves are represented as zero observed curves, so
their features encode depletion relative to the selected null model instead of
silently dropping background voxels.

Set `--observation-source original` to calculate features from the complete
point cloud. Keeping observations thinned while targets use the original cloud
reduces direct target leakage in held-out-voxel evaluation.

## Outputs

The default cache directory is `paper_feature_cache_02`. Each cache contains
sampled coordinates, targets, global features, local features, validity masks,
summary curves, radii, and complete configuration metadata.

The default model directory is `paper_feature_experiments_02`. It contains
feature scalers, run metadata, epoch metrics, best/final checkpoints, and a
top-level `experiment_summary.csv` for comparing stages and null models.

## Interpretation

The per-pattern global stage is intentionally retained as a control. Its 14
features are constant across a pattern and can be absorbed by model biases. The
shared global stage is the meaningful test of whether global features help a
single model distinguish pattern-level context. Local features vary between
voxels and can therefore provide spatial information to per-pattern models.
