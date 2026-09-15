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
| Voxel target fields | working, tested; `generate_density_grid` deprecated as a ground truth |
| Neural field (per-pattern) | working; see `example_01` |
| Local-feature neural field experiment | run and **halted at its own interpolation gate** |
| Ground-truth parameter recovery | done (Stage 0) |
| Global feature extraction for all patterns | done (Stage 1) |
| Amortized posterior over cluster parameters | done (Stage 2) |
| Calibration and coverage | **done, gate passed** (Stage 3) |
| Replay oracle for the guest-probability field | **done, gate passed** (reconstruction Stage 0) |
| Random-centre benchmark dataset | generated; 1,000 patterns |
| Classical baselines and headroom map | **done, gate passed** (reconstruction Stage 1): 48% of test cells have headroom |

The first line of work is **amortized Bayesian inference of the physical
cluster parameters**, specified in [`inference/ROADMAP.md`](inference/ROADMAP.md). Section 8
of that document records corrections made to the feature library, some of which
affect the earlier results.

The second is **implicit neural reconstruction of the solute field**: estimating
where the solute sits from a thinned point cloud, scored against an exact
oracle. It is specified in [`reconstruction/ROADMAP.md`](reconstruction/ROADMAP.md).
Stages 0 and 1 are complete. Each stage has a walkthrough in `docs/experiments/`
(E10 onward).

**For a full account of the package**, see
[`README_detailed.md`](README_detailed.md) — a paper-format description with an
abstract, background, methods, results and discussion, linking to a standalone
walkthrough for each of the nine experiments.

## Layout

```
geom_mesh_net/core_functions/   the library
  clustersim.py                 simulate clustered marked point patterns
  paper_spatial_features.py     G, F, K, cross-G and the 14 scalar features
  point_cloud_fields.py         voxel guest-probability fields
  field_oracle.py               exact guest probabilities by replaying the simulator
  voxelize_clusters.py          density grids from simulation parameters (deprecated as ground truth)
  spatial_stats_01.py           binned pair-correlation "spatial barcode"
  data_loader.py                torch Dataset and neural field models
  cluster_visualizer.py         3D plotting
  compare_benchmarks.py         benchmark figures
  paper_feature_experiments.py  offline feature caching

example_01/                     neural field experiments and their results
inference/                      posterior inference of cluster parameters
reconstruction/                 reconstruction of the solute field
tests/                          regression tests for the library
walkthroughs/                   explanatory documents
deprecated_code/                superseded scripts, kept for reference
data/                           1,000 simulated patterns (gitignored, ~5 GB)
data_random_centres/            the same parameters with random cluster centres (gitignored, ~5 GB)
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
python -m pytest -q
```

186 tests, about 90 seconds. They need no data: the fixtures build synthetic
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
PYTHONPATH=. python inference/recover_ground_truth.py
```

Run this before modifying `data_factory.py`. See `inference/ROADMAP.md`
section 2.2.

## Inference pipeline

```bash
PYTHONPATH=. python inference/recover_ground_truth.py        # Stage 0
PYTHONPATH=. python inference/extract_features.py --workers 7  # Stage 1
```

```bash
PYTHONPATH=. python inference/screen_features.py                # informativeness
PYTHONPATH=. python inference/fit_posterior.py                  # Stage 2
PYTHONPATH=. python inference/validate_posterior.py             # Stage 3
```

Given a measured point pattern, this returns a calibrated posterior over the
four physical cluster parameters. Cross-validated over all 1,000 patterns, the
90% credible intervals cover 88.8% of the time for the in-cluster concentration
and 86.7% for the mean cluster radius, with uniform simulation-based calibration
ranks.

The radius-spread parameter `rb` is barely identifiable from these features, and
the posterior says so: it comes back 96% as wide as the prior, and is calibrated
anyway. Reporting honest ignorance is the property a point estimate cannot have.

See [`inference/ROADMAP.md`](inference/ROADMAP.md) for the staged protocol, the
gates, and what each stage measured.

## Reconstruction pipeline

```bash
PYTHONPATH=. python reconstruction/generate_random_centres.py   # benchmark dataset, ~2 min, 5 GB
PYTHONPATH=. python reconstruction/freeze_benchmark.py          # splits, strata, mask checksums
PYTHONPATH=. python reconstruction/stage0_oracle.py             # Stage 0 oracle and Gate 0, ~2 min
PYTHONPATH=. python reconstruction/stage1_baselines.py          # Stage 1 baselines and Gate 1, ~30 min
PYTHONPATH=. python docs/make_reconstruction_figures.py         # E10+ figures
```

The oracle gives every atom of a simulated pattern its exact probability of
being a solute atom, by replaying the simulator's labelling. Against the realised
labels of 2.9 million in-sphere atoms, its logistic recalibration slope is 1.0007
and no reliability bin misses by more than 0.0008. Every reconstruction is scored
against it.

Standard kernel delocalisation comes within 10% of that truth for large clusters at
realistic detection efficiency. It leaves room in 92% of small-cluster cells, mostly
at cluster rims and cores. Adapting the kernel width to guest density closes 41% of
the gap it leaves.

See [`reconstruction/ROADMAP.md`](reconstruction/ROADMAP.md) for the stages, the
gates, and the pilot measurements that set them.

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
  correctly stopped. `inference/ROADMAP.md` section 12 revisits the interpretation.
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
reports whether each extremum was real. See `inference/ROADMAP.md` section 8.3.
