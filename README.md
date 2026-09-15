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
| Cluster simulator (`simulation/clustersim.py`) | working; 1,000 patterns generated |
| Spatial summary functions and 14 features | working, tested |
| Voxel target fields | working, tested; `generate_density_grid` deprecated as a ground truth |
| Neural field (per-pattern) | working; see `experiments/neural_field` |
| Local-feature neural field experiment | run and **halted at its own interpolation gate** |
| Ground-truth parameter recovery | done (Stage 0) |
| Global feature extraction for all patterns | done (Stage 1) |
| Amortized posterior over cluster parameters | done (Stage 2) |
| Calibration and coverage | **done, gate passed** (Stage 3) |
| Replay oracle for the guest-probability field | **done, gate passed** (reconstruction Stage 0) |
| Random-centre benchmark dataset | generated; 1,000 patterns |
| Classical baselines and headroom map | **done, gate passed** (reconstruction Stage 1): 48% of test cells have headroom |

The first line of work is **amortized Bayesian inference of the physical
cluster parameters**, specified in
[`experiments/inference/ROADMAP.md`](experiments/inference/ROADMAP.md). Section 8
of that document records corrections made to the feature library, some of which
affect the earlier results.

The second is **implicit neural reconstruction of the solute field**: estimating
where the solute sits from a thinned point cloud, scored against an exact
oracle. It is specified in
[`experiments/reconstruction/ROADMAP.md`](experiments/reconstruction/ROADMAP.md).
Stages 0 and 1 are complete. Each stage has a walkthrough in `docs/experiments/`
(E10 onward).

**For a full account of the package**, see
[`README_detailed.md`](README_detailed.md) — a paper-format description with an
abstract, background, methods, results and discussion, linking to a standalone
walkthrough for each of the nine experiments.

## Layout

```
geom_mesh_net/                    the library (installed with pip install -e .)
  simulation/
    clustersim.py                 simulate clustered marked point patterns
    parameters.py                 the four parameters, their prior, the data factory's draws
  statistics/
    paper_spatial_features.py     G, F, K, cross-G and the 14 scalar features
    presets.py                    named feature configurations shared by both tracks
    feature_cache.py              offline feature caching
    barcode.py                    binned pair-correlation "spatial barcode"
  fields/
    oracle.py                     exact guest probabilities by replaying the simulator
    baselines.py                  kernel-smoothing baselines B0, B1, B2
    point_cloud.py                voxel guest-probability fields
    density_grid.py               density grids from simulation parameters (deprecated as ground truth)
  neural/
    datasets.py                   torch Dataset and collate function
    models.py                     neural field models
  inference/
    flow.py                       conditional normalizing flow
    calibration.py                simulation-based calibration and coverage
  viz/                            3D plotting and benchmark figures
  paths.py                        data, experiment and figure locations
  core_functions/                 deprecated aliases for the old module names

experiments/                      run from the repository root as python -m experiments.<track>.<script>
  inference/                      posterior inference of cluster parameters (E1-E7)
  reconstruction/                 reconstruction of the solute field (E10 onward)
  neural_field/                   neural field experiments and their results (E8, E9)
scripts/generate_data.py          the data factory that wrote data/
tests/                            regression tests
docs/
  experiments/                    one walkthrough per experiment
  guides/                         introductory explanatory documents and notebook
  figures/                        figures used by the walkthroughs
deprecated_code/                  superseded scripts, kept for reference
data/                             1,000 simulated patterns (gitignored, ~5 GB)
data_random_centres/              the same parameters with random cluster centres (gitignored, ~5 GB)
```

The old imports, such as `from geom_mesh_net.core_functions import clustersim`,
still work but raise a `DeprecationWarning`. Each old name resolves to the same
module object as the new one. The datasets can live elsewhere: set
`GEOM_MESH_NET_DATA`, `GEOM_MESH_NET_DATA_RANDOM_CENTRES` or
`GEOM_MESH_NET_DATA_SHARED_UPP` (see `geom_mesh_net/paths.py`).

## Environment

```bash
mamba create -n geom_mesh_net python=3.10
mamba activate geom_mesh_net
mamba install numpy scipy matplotlib pandas pyvista -c conda-forge
pip install torch plotly trame trame-vtk trame-vuetify
pip install -e ".[dev,notebooks]"
```

## Tests

```bash
python -m pytest -q
```

198 tests, about 90 seconds. They need no data: the fixtures build synthetic
patterns, and the tests that do want `data/` skip when it is absent.

Where a closed form exists the tests compare against it rather than against a
recorded output, so they can catch a real error rather than merely detecting
change. Ripley's K is checked against the analytic CSR value on Poisson
patterns, the Kaplan-Meier estimator against a hand-worked censored example,
and the CSR baselines against their closed forms.

## Generating data

The dataset is not in version control. To regenerate it:

```bash
PYTHONPATH=. python scripts/generate_data.py
```

This writes 1,000 patterns totaling roughly 5 GB. Reduce `n_sims` in that file
for a smaller set. See [`docs/guides/data_factory.md`](docs/guides/data_factory.md).

The simulator varies four parameters — `rho_c`, `rho_b`, `cr`, `rb` — but
writes only two of them to `data/pattern_stats.npy`. The other two are
recovered by replaying the seeded generator:

```bash
python -m experiments.inference.recover_ground_truth
```

Run this before modifying `scripts/generate_data.py`. See
`experiments/inference/ROADMAP.md` section 2.2.

## Inference pipeline

```bash
python -m experiments.inference.recover_ground_truth             # Stage 0
python -m experiments.inference.extract_features --workers 7      # Stage 1
```

```bash
python -m experiments.inference.screen_features                  # informativeness
python -m experiments.inference.fit_posterior                    # Stage 2
python -m experiments.inference.validate_posterior               # Stage 3
```

Given a measured point pattern, this returns a calibrated posterior over the
four physical cluster parameters. Cross-validated over all 1,000 patterns, the
90% credible intervals cover 88.8% of the time for the in-cluster concentration
and 86.7% for the mean cluster radius, with uniform simulation-based calibration
ranks.

The radius-spread parameter `rb` is barely identifiable from these features, and
the posterior says so: it comes back 96% as wide as the prior, and is calibrated
anyway. Reporting honest ignorance is the property a point estimate cannot have.

See [`experiments/inference/ROADMAP.md`](experiments/inference/ROADMAP.md) for the staged protocol, the
gates, and what each stage measured.

## Reconstruction pipeline

```bash
python -m experiments.reconstruction.generate_random_centres    # benchmark dataset, ~2 min, 5 GB
python -m experiments.reconstruction.freeze_benchmark           # splits, strata, mask checksums
python -m experiments.reconstruction.stage0_oracle              # Stage 0 oracle and Gate 0, ~2 min
python -m experiments.reconstruction.stage1_baselines           # Stage 1 baselines and Gate 1, ~30 min
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

See [`experiments/reconstruction/ROADMAP.md`](experiments/reconstruction/ROADMAP.md) for the stages, the
gates, and the pilot measurements that set them.

## Walkthroughs

- [`clustersim_introduction.md`](docs/guides/clustersim_introduction.md) — how
  the cluster simulation works
- [`data_factory.md`](docs/guides/data_factory.md) — generating the dataset
- [`load_data_train_network.md`](docs/guides/load_data_train_network.md) —
  loading patterns, voxelizing, and training a field
- [`clustersim_todo.md`](docs/guides/clustersim_todo.md) — open ideas

## Experiments

- [`experiments/neural_field/EXPERIMENTAL_METHODOLOGY_01.md`](experiments/neural_field/EXPERIMENTAL_METHODOLOGY_01.md)
  and its [report](experiments/neural_field/methodology_01_results/report/REPORT.md) — whether
  spatially varying local features improve a per-pattern neural field. The
  prespecified interpolation gate failed for all three patterns and the work
  correctly stopped. `experiments/inference/ROADMAP.md` section 12 revisits the
  interpretation.
- [`experiments/neural_field/PAPER_FEATURE_EXPERIMENTS.md`](experiments/neural_field/PAPER_FEATURE_EXPERIMENTS.md)
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
reports whether each extremum was real. See `experiments/inference/ROADMAP.md`
section 8.3.
