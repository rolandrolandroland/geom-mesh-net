# Geom Mesh Net

Spatial statistics and machine learning for 3D clustered point patterns, of the
kind produced by atom probe tomography.

The project has two purposes:

1. Bring spatial-statistics tools from the R packages `rapt` and `rTEM` into
   Python.
2. Use them to infer physical structure from measured point patterns.

## Current state

| Component | Status |
| --- | --- |
| Cluster simulator (`clustersim.py`) | working; 1,000 patterns generated |
| Spatial summary functions and 14 features | working, tested |
| Voxel target fields | working, tested |
| Neural field (per-pattern) | working; see `example_01` |
| Local-feature neural field experiment | run and **halted at its own interpolation gate** |
| Posterior inference of cluster parameters | specified, not yet built |

The active line of work is **amortized Bayesian inference of the physical
cluster parameters**, specified in [`sbi/ROADMAP.md`](sbi/ROADMAP.md). Section 8
of that document records corrections made to the feature library, some of which
affect the earlier results.

## Layout

```
geom_mesh_net/core_functions/   the library
  clustersim.py                 simulate clustered marked point patterns
  paper_spatial_features.py     G, F, K, cross-G and the 14 scalar features
  point_cloud_fields.py         voxel guest-probability fields
  voxelize_clusters.py          density grids from simulation parameters
  spatial_stats_01.py           binned pair-correlation "spatial barcode"
  data_loader.py                torch Dataset and neural field models
  cluster_visualizer.py         3D plotting
  compare_benchmarks.py         benchmark figures
  paper_feature_experiments.py  offline feature caching

example_01/                     neural field experiments and their results
sbi/                            posterior inference of cluster parameters
tests/                          regression tests for the library
walkthroughs/                   explanatory documents
deprecated_code/                superseded scripts, kept for reference
data/                           1,000 simulated patterns (gitignored, ~5 GB)
```

## Environment

```bash
mamba create -n geom_mesh_net python=3.10
mamba activate geom_mesh_net
mamba install numpy scipy matplotlib pandas pyvista -c conda-forge
pip install torch torchvision torchaudio plotly trame trame-vtk trame-vuetify
pip install pytest
pip install -e .
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

62 tests, about 2 seconds. They need no data: the fixtures build synthetic
patterns, and the tests that do want `data/` skip when it is absent.

Where a closed form exists the tests compare against it rather than against a
recorded output, so they can catch a real error rather than merely detecting
change. Ripley's K is checked against the analytic CSR value on Poisson
patterns, the Kaplan-Meier estimator against a hand-worked censored example,
and the CSR baselines against their closed forms.

## Generating data

The dataset is not in version control. To regenerate it:

```bash
PYTHONPATH=. python deprecated_code/tester_scripts/data_factory.py
```

This writes 1,000 patterns totaling roughly 5 GB. Reduce `n_sims` in that file
for a smaller set. See [`walkthroughs/data_factory.md`](walkthroughs/data_factory.md).

The simulator varies four parameters — `rho_c`, `rho_b`, `cr`, `rb` — but
writes only two of them to `data/pattern_stats.npy`. The other two are
recovered by replaying the seeded generator:

```bash
PYTHONPATH=. python sbi/recover_ground_truth.py
```

Run this before modifying `data_factory.py`. See `sbi/ROADMAP.md` section 2.2.

## Walkthroughs

- [`clustersim_introduction.md`](walkthroughs/clustersim_introduction.md) — how
  the cluster simulation works
- [`data_factory.md`](walkthroughs/data_factory.md) — generating the dataset
- [`load_data_train_network.md`](walkthroughs/load_data_train_network.md) —
  loading patterns, voxelizing, and training a field
- [`clustersim_todo.md`](walkthroughs/clustersim_todo.md) — open ideas

## Experiments

- [`example_01/EXPERIMENTAL_METHODOLOGY_01.md`](example_01/EXPERIMENTAL_METHODOLOGY_01.md)
  and its [report](example_01/methodology_01_results/report/REPORT.md) — whether
  spatially varying local features improve a per-pattern neural field. The
  prespecified interpolation gate failed for all three patterns and the work
  correctly stopped. `sbi/ROADMAP.md` section 12 revisits the interpretation.
- [`example_01/PAPER_FEATURE_EXPERIMENTS.md`](example_01/PAPER_FEATURE_EXPERIMENTS.md)
  — a six-stage staged comparison. Implemented; not yet run.

## Note on the K transform

The K transform is now selectable via `PaperFeatureConfig.k_transform` and
defaults to `"sqrt"`, matching the Bennett et al. feature definitions the
project ports. `"cube_root"` is available for comparison — it is the
variance-stabilizing form for a 3D CSR process — but it changes the feature
semantics, so it is opt-in.

The setting that *does* need attention is `k_r_max`. Under `sqrt` the default of
10.0 leaves roughly two thirds of patterns with no interior K extremum, so the
radius-valued features silently return a grid endpoint. Set it from the physical
cluster scale; for `data/` that is 40.0. `PaperFeatureResult.k_extrema_interior`
reports whether each extremum was real. See `sbi/ROADMAP.md` section 8.3.
