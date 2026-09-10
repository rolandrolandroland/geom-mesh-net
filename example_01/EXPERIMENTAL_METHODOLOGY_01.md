# Experimental Methodology 01: Local Spatial Features

## 1. Purpose

This first experiment will test one focused question:

> Does adding spatially varying local Bennett et al. features improve a
> per-pattern neural field relative to the same neural field trained from
> coordinates alone?

The first experiment will not compare loss functions, global features, shared
models, or analytical CSR against random labeling. Those factors will be
introduced only if the local-feature model passes the advancement criteria in
Section 13.

The target wall-clock duration is approximately seven hours on the 2020 M1
MacBook Pro with eight CPU cores and 16 GB of unified memory.

## 2. Experimental hypotheses

The null hypothesis is that adding interpolated local spatial features does
not improve held-out spatial-field reconstruction relative to coordinates
alone.

The alternative hypothesis is that local spatial features improve recovery of
guest-enriched regions without materially degrading whole-field probability
accuracy or calibration.

## 3. Point patterns

The screening experiment will use patterns 0, 1, and 4.

| Pattern | Role | Existing spatial evidence |
| --- | --- | --- |
| 0 | Strong structure | 2 clusters, G maximum difference 0.732, F minimum difference -0.699 |
| 1 | Moderate structure | 10 clusters, G maximum difference 0.444, F minimum difference -0.381 |
| 4 | Weak/fine structure | 170 small clusters, G maximum difference 0.271, F minimum difference -0.200 |

This selection spans a strong low-count cluster pattern, an intermediate
pattern, and a weak pattern containing many small clusters. Pattern selection
is fixed before model fitting.

## 4. Observation and target construction

Each original pattern contains 216,000 labeled points in a 60 by 60 by 60
domain.

1. Use random seed 42 plus the pattern index.
2. Select 10% of the points without replacement as the observation point
   cloud.
3. Use only the observation point cloud to calculate global or local spatial
   features.
4. Use the complementary 90% of points to construct the target field.
5. Treat labels 2 and 3 as guests and labels 0 and 1 as hosts.
6. Construct the target on a 0.5-unit grid using a Gaussian bandwidth of 1.5
   physical units.

The target grid therefore contains 120 cubed, or 1,728,000, voxels. Separating
the observation and target points prevents the feature calculation from using
the exact points that define the evaluation target.

## 5. Random-label null model

The first experiment will use random labeling because the scientific question
is whether guest labels are spatially unusual conditional on the measured
physical point structure.

For each pattern:

1. Preserve every observation-point coordinate.
2. Preserve the total number of observation guest labels.
3. Generate 89 complete whole-pattern random relabelings.
4. Apply each global relabeling to every local window.
5. Do not fix or redraw the number of guests independently inside a local
   window.
6. Use the pointwise median of the 89 relabeled summary curves as the expected
   curve.

The same 89 global relabelings must be reused for every local query location in
a pattern.

## 6. Local feature grid

Exact local features will be calculated on a coarse regular grid before any
model training.

- Grid spacing: 4 physical units
- Grid coordinates per axis: 2, 6, 10, ..., 58
- Grid shape: 15 by 15 by 15
- Exact coarse-grid locations: 3,375 per pattern
- Local window: an axis-aligned cube extending 12 units from the query center
  in each direction
- Boundary behavior: clip the local cube to the 60-unit pattern domain

The local summary-function configuration is fixed as follows:

| Summary | Radius interval | Samples |
| --- | --- | --- |
| Guest G and F | 0 to 8 | 401 |
| Guest K | 0 to 8 | 161 |
| Guest-to-host cross-G | 0 to 3 | 301 |

The F calculation will use an 8 by 8 by 8 regular query grid within each local
window. K will use at most 512 local guest points. Guest-free and single-guest
windows will remain in the dataset; unavailable observed guest G and K curves
will be represented by zero curves so that depletion remains a measurable
signal.

The same 14 feature definitions will be used at every location.

## 7. Interpolation validation

The approximation introduced by the coarse feature grid must be validated
before neural-field results are interpreted.

For each pattern:

1. Select 256 additional voxel centers uniformly without replacement using
   seed 10,042 plus the pattern index.
2. Exclude coarse-grid locations from this set.
3. Calculate the exact 14 local features at all 256 locations using the same
   observation points and global relabelings.
4. Trilinearly interpolate each coarse-grid feature channel to the 256 exact
   validation locations.
5. Compare exact and interpolated values feature by feature.

For each feature, report MAE, RMSE, robust normalized RMSE, Pearson
correlation, and Spearman correlation. Robust normalized RMSE is RMSE divided
by the difference between the coarse-grid 95th and 5th percentiles.

Interpolation passes if:

- Median robust normalized RMSE across the 14 channels is at most 0.15.
- At least 12 of 14 channels have Spearman correlation of at least 0.80.
- No cluster-scale radius channel has robust normalized RMSE above 0.30.

If interpolation fails, neural-field comparison stops. The feature grid must
then be refined or made adaptive before model conclusions are drawn.

## 8. Complete feature field

After interpolation passes, trilinearly interpolate all 14 feature channels to
the complete 120 by 120 by 120 voxel grid. Store the interpolated feature field
as float32 values.

Fit feature-standardization parameters using training-region voxels only.
Apply the same means and standard deviations to training and validation
regions. Coordinate inputs will be scaled independently to the interval -1 to
1.

The complete feature field makes every voxel eligible for training, but the
optimizer will use balanced stochastic batches rather than processing all
1,728,000 voxels during every epoch.

## 9. Spatial validation split

Random individual-voxel validation will not be used.

1. Divide the domain into a 5 by 5 by 5 grid of non-overlapping 12-unit spatial
   blocks.
2. Mark a block as guest-enriched if it contains any voxel in the upper 10% of
   target probabilities.
3. Using seed 42 plus the pattern index, hold out 20% of guest-enriched blocks
   and 20% of the remaining blocks.
4. Use every voxel in the selected blocks as the validation set.
5. Use all remaining blocks as the training population.

The block assignments must be identical for the coordinate-only and
local-feature models.

## 10. Models

Train two models independently for each of the three patterns, for six models
in total.

### Model C: coordinate control

- Inputs: normalized x, y, and z
- Feature count: 0

### Model L: local spatial features

- Inputs: normalized x, y, and z plus the 14 standardized interpolated local
  features
- Feature count: 14

Both models will use the same architecture:

- Three hidden layers
- 128 units per hidden layer
- ReLU hidden activations
- One sigmoid output representing guest probability

No global features will be included in this first experiment.

## 11. Training protocol

The training protocol is fixed rather than treated as another experimental
factor.

- Loss: ordinary binary cross-entropy
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 2,048 voxels
- Maximum epochs: 150
- Training batches per epoch: 32
- Validation frequency: every 5 epochs
- Early-stopping patience: 4 validation checks, equivalent to 20 epochs
- Early-stopping minimum improvement: 0.0001 validation BCE
- Training seed: 42
- Device: CPU unless MPS is available in the normal Terminal environment

Each training batch will contain 1,024 voxels from the upper 10% of training
target probabilities and 1,024 voxels from the remaining training voxels.
Separate shuffled index queues will be used for the two groups and reshuffled
when exhausted. Model C and Model L must receive identical voxel indices in the
same order during every epoch.

Validation will always use the complete held-out spatial blocks without
cluster weighting.

## 12. Evaluation metrics

### Primary metrics

- Brier score over all held-out voxels
- Dice overlap between predicted and target upper-10% regions

### Secondary field metrics

- Binary cross-entropy
- MAE
- RMSE
- Pearson correlation
- Spearman correlation
- Ten-bin expected calibration error
- Precision-recall area under the curve for the upper-10% target region

### Spatial morphology metrics

- Number of connected upper-10% regions
- Median matched cluster-centroid distance
- Relative cluster-volume error
- Fraction of target high-density regions recovered

All metrics will be calculated on the unweighted held-out spatial blocks.
Training loss will not be used as evidence of model superiority.

## 13. Advancement criteria

The local-feature approach advances only if all of the following are true:

1. Interpolation passes the criteria in Section 7.
2. Model L has lower Brier score and higher upper-10% Dice than Model C in at
   least two of the three patterns.
3. The median relative Brier improvement is at least 5%, or the median absolute
   Dice improvement is at least 0.05.
4. Model L does not worsen Brier score by more than 10% in any pattern.

This is a screening experiment with one training seed, not a final statistical
significance test. If the criteria are met, the comparison will be repeated
with three training seeds before introducing additional feature, null-model,
or loss-function factors.

## 14. Seven-hour runtime budget

The observed M1 benchmark required 4.01 seconds of feature-calculation time for
32 query voxels and 5 random relabelings on pattern 0.

Each pattern in this experiment requires:

- 3,375 coarse-grid feature locations
- 256 exact interpolation-validation locations
- 3,631 total exact locations
- 89 whole-pattern relabelings

Across three patterns, this is 969,477 voxel-relabeling evaluations. Linear
scaling from the measured benchmark gives:

```text
4.01 seconds * (3,631 * 89 * 3) / (32 * 5)
= approximately 24,300 seconds
= approximately 6.75 hours
```

Target construction, interpolation, validation, and six small-network training
runs are expected to use the remaining approximately 15 minutes. The practical
target is seven hours, with an expected range of roughly 6.5 to 7.5 hours due
to point-density differences and thermal throttling.

The feature cache must be written after every completed pattern. Rerunning an
interrupted experiment must reuse matching completed caches.

## 15. Reproducibility and outputs

Store the following for every pattern:

- Observation and target point indices
- Coarse-grid and exact-validation query coordinates
- Whole-pattern relabeling seed and count
- Exact coarse and validation features
- Interpolated full-grid features
- Feature-standardization parameters
- Spatial block assignments
- Training batch-index sequences
- Model configuration and parameter count
- Per-epoch training and validation metrics
- Best and final checkpoints
- Final unweighted validation predictions and metrics
- Wall-clock preprocessing, training, and inference times

Use `caffeinate -i` for the long preprocessing command. Internet access is not
required, and completed pattern caches must make the run resumable.

## 16. Deferred experiments

The following experiments are intentionally deferred:

1. Repeating the winning comparison with three training seeds
2. Global features in per-pattern models as a negative control
3. Shared coordinate versus shared global-feature models
4. Global-plus-local feature models
5. Analytical CSR versus random-label expected curves
6. Alternative local-window radii
7. Weighted BCE, composite cluster loss, or focal loss
8. Larger pattern sets and out-of-pattern generalization

Only the strongest configuration from each stage should advance. A full
factorial combination of all choices will not be run.

## 17. Execution status

This protocol has now been implemented and executed by
`execute_experimental_methodology_01.py`. The run includes the deterministic
observation/complement split, exact local-feature grids, interpolation checks,
full-grid interpolation, spatial validation blocks, balanced batches, early
stopping, complete-grid inference, metrics, plots, runtime logs, and resumable
per-pattern caches.

The complete results and interpretation are in
`methodology_01_results/report/REPORT.md`. Because all three patterns failed the
prespecified interpolation gate, the model fits were run only as explicitly
labeled exploratory diagnostics and do not override the stopping rule.
