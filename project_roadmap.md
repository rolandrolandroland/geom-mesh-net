# Geom Mesh Net — project roadmap

*Status as of 22 September 2026.*

Geom Mesh Net asks two questions of a labelled 3D point pattern of the kind atom
probe tomography produces:

1. **Inference.** Which physical parameters produced it, and how sure can we be?
   The answer is a calibrated posterior over four parameters: the solute
   concentration inside precipitates and in the matrix, and the mean and spread of
   precipitate radius.
2. **Reconstruction.** Where is the solute? The answer is a continuous estimate of
   each atom's probability of being a solute atom, scored against an exact oracle.

The two tracks share the cluster simulator, the spatial-statistics library and the
data. They also share one rule: every stage has a pass/fail gate, stated before
the stage runs.

This page is the overview. The detail, including every gate definition and
correction, lives in the two track roadmaps:

- [`experiments/inference/ROADMAP.md`](experiments/inference/ROADMAP.md)
- [`experiments/reconstruction/ROADMAP.md`](experiments/reconstruction/ROADMAP.md)

Where this page and a track roadmap disagree, the track roadmap governs.

| | |
| --- | --- |
| Simulated datasets | 3 × 1,000 patterns: `data/`, `data_random_centres/`, `data_shared_upp/` |
| Pattern size | 216,000 atoms in a 60³ domain |
| Spatial-summary features | 14 |
| Physical parameters | 4 |
| Experiment walkthroughs | 16 (E1–E11 and E15–E19 in `docs/experiments/`) |
| Tests | 252 |

---

## At a glance

| Track | Status | Headline | Next |
| --- | --- | --- | --- |
| Inference | Stages 0–4 **passed**, on both lattice and random centres | 90% credible intervals cover the truth 88–92% of the time for all four parameters | `rho_b`'s rank bias, the one open calibration issue |
| Reconstruction | Stages 0–2 and 5.1 **passed**; 5.2 **failed** | A Fourier-feature field beats tuned smoothing in 48 of 52 headroom test cells, closing 68% of its gap (Gate 2); a diffusion law imposed exactly predicts the matrix, and fitting the precipitates with it recovers the capillary length that detect-then-fit loses (E18) | Write up Stage 2 as E19; decide whether the joint fit becomes Stage 5.3 with a gate; detection is the binding constraint for both |

---

## Shared foundation

**Cluster simulator** (`geom_mesh_net/simulation/clustersim.py`). It works, and a
seeded generator writes all four parameters. One problem was found in it this
week: cluster centres come from a subsampled cubic lattice (see
[Across both tracks](#across-both-tracks)).

**Spatial-statistics library** (`geom_mesh_net/statistics/`). This computes the
summary functions G, F, K and guest-to-host G, and the 14 features of Bennett,
Proudian & Zimmerman (2023).

- Feature extraction reproduces rapt, the paper's R package, to 6.8e-13 on every
  feature. That includes every case where rapt reports a feature as missing
  (`tests/test_rapt_parity.py`).
- Against spatstat, G and guest-to-host G agree to 2e-5. K agrees to 1e-13 once a
  constant n/(n−1) normalisation factor is allowed for.
- **F does not match**. spatstat measures distances with a 26-neighbour chamfer
  transform on a voxel grid. Reproducing that closes most of the gap, but not all.
- `--preset stage1` keeps the earlier port, which the inference results use.
  `--preset paper` reproduces rapt and its walkthrough's grids.

**Replay oracle** (`geom_mesh_net/fields/oracle.py`). This gives every simulated
atom its exact probability of being solute, by replaying the simulator's own
labelling. It replaces `generate_density_grid`, which is wrong inside clusters.

---

## Track 1 — Inference

*Given a pattern, return a calibrated posterior over `rho_c`, `rho_b`, `cr` and
`rb`.* Roadmap: [`experiments/inference/ROADMAP.md`](experiments/inference/ROADMAP.md).

### The result

These figures come from a five-flow ensemble, cross-validated over all 1,000
patterns of `data/`, on the `stage1` features.

| Parameter | Prior | Meaning | R² | Contraction | 90% coverage | SBC ranks |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `rho_c` | U(0.2, 1.0) | solute inside clusters | 0.964 | 0.865 | 0.919 | uniform |
| `rho_b` | U(0, 0.05) | solute in the matrix | 0.925 | 0.901 | 0.902 | **small bias** |
| `cr` | U(3, 15) | mean cluster radius | 0.843 | 0.649 | 0.885 | uniform |
| `rb` | U(0, 0.5) | radius spread | 0.112 | 0.039 | 0.882 | uniform |

`rb` is barely constrained: its posterior comes back 96% as wide as its prior.
It is still calibrated, so the method reports its ignorance with trustworthy error
bars.

### Stages

| Stage | Gate | Outcome | Status |
| --- | --- | --- | --- |
| 0 — Recover and store ground truth | exact equality with stored columns | 0.0 difference on all 1,000 | passed |
| 1 — Extract features | ≥ 95% finite, ≥ 80% interior `Rm` | 100% finite, 96.4% interior, 7.6 min | passed |
| 2 — Fit the flow | validation log-likelihood beats the prior | 6.939 against 1.427 nats | passed |
| 3 — Calibration and coverage | SBC in band; 90% coverage in [0.85, 0.95] | passed with a five-flow ensemble | passed |
| 4 — Feature sufficiency | descriptive, no gate | each parameter is carried by one summary function; K is indispensable for `cr` | done |
| 5 — More simulations | — | demoted: calibration plateaus from n = 400 | open |
| 6 — Real measured data | — | model misspecification is its own research problem | separate project |

Work after the stages (walkthroughs E6 and E7; roadmap §8.7–8.10):

- **Observation augmentation.** It did not help, because it adds near-duplicates.
- **`rho_b` diagnosis.** Width was fixable with an ensemble. A bias remains.
- **Parity with rapt (§8.9).** The K boundary-pinning defect was introduced by the
  Python port; the published method never had it.
- **rapt's missing-value rule and the shared point pattern (§8.10).**

### Why `rb` scores far below the paper

The paper reports R² ≈ 0.71 for radius spread. Moving this pipeline toward the
paper's design one step at a time (ridge regression; §8.10):

| Change | `rb` R² |
| --- | ---: |
| independent points, all rows, 8 always-defined features | 0.07 |
| plus the legacy port's K features | 0.12 |
| only rows where rapt defines every feature | 0.26–0.28 |
| and only the paper's radius range | 0.24 |
| **and one point pattern shared by every simulation, as in the paper** | **0.44** |

Most of the rise comes from removing point-position noise, and it arrives only
through the K features. Real measurements always carry that noise. Training size,
model, guest fraction, position blur and the null model remain untested.

---

## Track 2 — Reconstruction

*Estimate where the solute is, from a thinned point cloud, and score it against an
exact oracle.* Roadmap:
[`experiments/reconstruction/ROADMAP.md`](experiments/reconstruction/ROADMAP.md).

### The result so far

- **The yardstick is exact.** Scored against the realised labels of 2.9 million
  in-sphere atoms, the oracle's recalibration slope is 1.0007 ± 0.0013. No
  reliability bin misses by more than 0.0008.
- **Standard practice leaves room.** Kernel smoothing (B1) was tested on 100 test
  patterns at three detection efficiencies, giving 300 cells. In 48% of them it
  leaves at least 15% of the gap to the truth open. By regime:

  | Regime | Cells with headroom |
  | --- | ---: |
  | small clusters | 92% |
  | large clusters | 16% |
  | 10% detection efficiency | 69% |
  | 80% detection efficiency | 30% |

- **B1's error sits at cluster rims and cores**, where the median excess per atom is
  ten times the matrix's.
- **The bar to beat is B2**, a smoother that adapts its width to local solute
  density. It beats B1 in 94% of cells. Where there is headroom, it closes a median
  41% of the gap B1 leaves, and it still leaves headroom in 19% of cells.
- **A physical law helps the matrix, not yet its constants** (Stage 5). In simulated
  data whose matrix obeys a screened diffusion law, a physics-informed network could not
  recover the law's constants even given the true geometry (E16). Imposed exactly around
  precipitates detected in the data, the law cut the matrix's excess loss to a median 26%
  of a constant's at realistic efficiency, against 79% for an unconstrained network. A rule
  that requires identified constants rejected 61 of 72 matrices that break the law. Gate 5.2
  failed on the capillary length, which errors in the detected radii pull toward zero (E17).
  A bound recomputed without assuming the geometry shows the missing accuracy is the method's,
  not the data's: leaving every radius unknown widens it by only 7% to 29%.

### Stages

| Stage | Question | Gate | Status |
| --- | --- | --- | --- |
| 0 — The yardstick | can the true field be computed exactly? | calibration slope and reliability on development patterns | passed |
| 1 — Baselines and headroom | does classical smoothing leave room? | ≥ 25% of test cells with headroom | passed (48%) |
| 2 — A field fitted to one pattern | does a Fourier-feature network beat B1 and B2? | beat both in ⅔ of headroom cells; close ≥ 20% of B1's gap | **passed** (2026-09-22): 48 of 52 cells, 68% of the gap closed, write-up pending |
| 3 — A field trained across simulations | does a learned prior beat any per-pattern estimator? | on test headroom cells | planned (extension) |
| 4 — Fields to precipitates | do fields find precipitates better than current practice? | beat maximum separation and DBSCAN on F1 and radius error | planned (extension) |
| 5 — Physics-informed field | does a governing equation help? | Gates 5.1 and 5.2 | 5.1 passed (2026-09-16); 5.2 **failed** (2026-09-17) on the capillary length, after the physics-informed network was replaced by the exact law |
| 6 — Position blur in the loss | does modelling the blur recover the sharp field? | Gate 6 | optional |
| 7 — Rendering and write-up | — | — | planned (core) |

**Schedule.** The five-week core is Stages 0, 1, 2, 5 and 7. The extension,
weeks 6–9, is Stages 3, 4 and 6. A failed gate shortens the schedule rather than
extending it.

---

## Across both tracks

**Cluster centres sit on a lattice.** In `data/` and `data_shared_upp/`, centres
come from a subsampled cubic lattice. rapt used Poisson centres.

- In 22 patterns of `data/` the lattice lost its outer shell, leaving far fewer
  clusters than the parameters imply.
- **Measured on 22 September 2026: the inference result does not depend on the
  lattice.** The whole pipeline was re-run on `data_random_centres/`, which holds
  the same 1,000 parameter vectors with random centres. Gate 3 passes there too:
  90% coverage 0.904, 0.894, 0.890, 0.879 against 0.919, 0.902, 0.885, 0.882 on
  the lattice, and every SBC deviation is *smaller*. Results in
  `experiments/inference/posterior_random_centres/`.
- `rho_b`'s rank bias survives the change, at 0.0455 against a 0.0429 band
  (0.0521 on the lattice), and its mean rank is 0.483 against 0.500. So the bias
  is not an artefact of where the simulator puts its clusters.
- `data_random_centres/` was generated for the reconstruction benchmark. It uses
  the same parameter vectors with random centres, so it allows a paired re-run of
  the inference track.
- This is recorded in the reconstruction roadmap (§3.2, §11), but not yet in the
  inference roadmap.

**The comparison against standard practice is built once.** Reconstruction Stage
4 compares fields with the maximum separation method and DBSCAN. It also computes
population estimates of `cr`, `rb`, `rho_c` and `rho_b` from each segmentation and
compares them with the inference posterior. This is the result a referee would ask
for first.

**A learned summary for `rb`.** A CNN over the voxelised pattern can count
clusters and measure individual radii. That is exactly the information a
pair-distance statistic averages away. Its output would feed the existing flow,
not replace it, and it would be judged by calibration and posterior contraction.
It belongs to the inference track and can reuse reconstruction Stage 3's encoder.

**Differences from the paper's design.** Documented in `README_detailed.md` §5.3:

- a shared point pattern vs independent points;
- relabelling vs the analytic CSR null;
- guest fraction 0.051 vs 0.1;
- radius and background ranges;
- position blur;
- 100,000 vs 1,000 patterns;
- missing-feature handling;
- the F estimator.

Lattice centres belong on that list and are not yet in it.

---

## What is not done

These are grouped by when the work should happen, and ordered within each group
by how much it changes what the project can claim.

### Now

1. **Write up Stage 2 and commit it.** Gate 2 passed on all 108 test cells, and
   the result currently exists only as a results file: E19's results sections and
   its test-split figures are unwritten, and the harness, its frozen design
   contract and its tests are uncommitted.
2. **Detection on the Stage 2 field** (both tracks). Precipitates are found by
   segmenting a smoothed field; Stage 2 has now shown that the Fourier-feature
   field beats that smoother. Missed precipitates are what ruins the capillary
   length (E18), and detection is also what Stage 4 compares against practice, so
   one substitution serves both. Pilot it on development patterns.
3. **Stage 5.3: the joint geometry fit, with a gate.** E18's evidence is eight
   development patterns at one efficiency with one degenerate fit among them.
   Fixing thresholds first, keeping the at-a-bound check, and excluding η = 0.1,
   where detection finds a median 44% of precipitates, would turn it into a
   result.

### Then

4. **`rho_b` rank bias.** Mean normalised rank is 0.478 against 0.500, and it
   correlates −0.137 with `rb`. Width, shrinkage, capacity and sample size have
   been ruled out; confounding with `rb` is the remaining hypothesis.
5. **A posterior on rapt-faithful features.** Refit the flow with missing K
   features imputed and flagged, or with censoring encoded explicitly ("peak beyond
   r"). Dropping incomplete rows keeps only 35% of patterns, mostly small clusters.
6. **Paper ranges or documented deviation.** Decide whether to regenerate data at
   the paper's ranges (radius 2–6.5, background 0–0.035, guest fraction 0.051,
   with blur), or keep this dataset and document the difference.
7. **Recheck the `cr` learning-curve drift** with the robust width ratio. The
   reported drift from 1.03 to 1.24 used the estimator later shown to be dominated
   by two patterns. About 10 minutes.
8. **Reconstruction Stage 5.** Done. Stage 5.1, the diffusion-field simulator, passed on
   2026-09-16 (E15). Stage 5.2 failed on 2026-09-17 (E16, E17): the law, imposed exactly,
   predicts the matrix and exposes violations, but detected radii bias the capillary length.
   The work after the gate (E18) found that the information was there all along — an unknown
   geometry costs 7 to 29 per cent of the bound — and that fitting the precipitates with the
   law, centres included, reaches it. What it cannot repair is a precipitate detection never
   found, which puts **Stage 4 on the critical path** rather than beside it.

### Later, or separate

9. **`Rddm`.** Ablation gives +0.149 ± 0.144, which resolves nothing either way.
   The real fix is a per-pattern K radius.
10. **F estimator parity.** Needs spatstat's C source to close the last 0.014–0.028.
11. **Reconstruction extension.** Stage 3 (learned prior), Stage 4 (precipitates
    against standard practice), and optionally Stage 6 (blur).
12. **Ten thousand simulations.** No longer needed for `rho_b`. Still useful for
    rare structures and for the 11 features Stage 4 could not resolve. Features
    survive 90% thinning, so patterns can be stored at a tenth the size (E6).
13. **Free `pcp`.** The overall solute fraction is fixed at 0.1, so it cannot be
    inferred. In a real measurement it is a primary unknown.
14. **Real measured data.** Model-misspecification research, not a next step.

### Housekeeping

- `experiments/neural_field/global_paper_feature_validation/` used
  `k_r_max = 70`, which is now refused. Its K features would change if it were
  regenerated.
- Corrected on 22 September 2026: the single-flow coverage figures quoted in
  `README.md`, the test counts in both READMEs, and §13 of the inference roadmap,
  which listed augmentation as open. `LICENSE` (MIT) was added at the same time.

---

## Claims revised

Conclusions that were stated and later corrected by measurement. Each correction
is recorded in the relevant roadmap rather than overwritten.

| Claimed | Measured | Where |
| --- | --- | --- |
| The translation-corrected K estimator explodes at large radii | Unbiased up to r = L (ratio 1.000); about 9% low only beyond it | inference §8.6 |
| Raising `k_r_max` costs nothing | It trades small-cluster coverage for large; 40 is a compromise at 96.4%, not an optimum | inference Stage 1 |
| `rb` carries no information (registered as R² ≤ 0.10) | 0.112 ± 0.045, positive on all 25 splits; recorded as a failed prediction | E2 |
| Pairing configurations by seed cancels training noise | Error bar shrinks only 1.2×; the power came from more seeds | E5 |
| More simulations would worsen `rho_b`, because augmentation did | Augmentation adds near-duplicates, new draws do not; the conclusion survived only on independent evidence | inference §8.7, E6 |
| The flow is mildly overconfident, worsening with data | Typical patterns are calibrated (robust ratios 0.97–1.13); a few degenerate patterns carry huge errors | inference §8.8 |
| The width ratio is a sound diagnostic | Two patterns in 1,000 moved `rho_c` from 1.38 to 0.97 | inference §8.8 |
| Purging the debug file shrinks the repository to about 1 MB | 46 MB: it was sized by working-tree bytes, not packed size | — |
| The K boundary-pinning defect belongs to the published features | The Python port introduced it; rapt reports those features as missing | inference §8.9 |
| The port's invented K values were useless | They carry a censoring signal: `cr` R² 0.858, against 0.715 when imputed | inference §8.10 |
| `rb` is nearly unidentifiable | True with independent points; reaches 0.44 for small clusters with a shared point pattern | inference §8.10 |
| `generate_density_grid` is a ground-truth field | Wrong inside clusters: where it says 0.87, 77% of atoms are solute | E10 |
| The July global barcode informs the network | Computed on all atoms, it carries no information | reconstruction §3.2 |
| A physics-informed network with the diffusion law as penalties can recover the law's constants | With the true geometry its loss preferred flat constants by 3–144 data margins; replaced by the exact law | E16 |
| (R/r)² matrices can test whether the physics is rejected | They lie within 2.5–4.7 nats of the best diffusion fit over a whole pattern; no rule could reject them | E17 |
| Comparing held-out loss with an unconstrained network tells when the law fails | It rejected 26 of 72 misspecified test cells; requiring identified constants and a win over a constant rejected 60 | E17 |

---

## Reproducing it

Run from the repository root.

```bash
python -m pytest                                                    # 228 tests

# Inference track
PYTHONPATH=. python scripts/generate_data.py                        # data/, ~8 min
python -m experiments.inference.recover_ground_truth                # Stage 0, 6 s
python -m experiments.inference.extract_features --workers 7        # Stage 1, 7.6 min
python -m experiments.inference.fit_posterior                       # Stage 2, 7 s
python -m experiments.inference.validate_posterior --ensemble 5     # Stage 3, 11 min
python -m experiments.inference.ablate_features --seeds 6           # Stage 4, 15 min
python -m experiments.inference.generate_shared_upp                 # §8.10, 5 min
python -m experiments.inference.compare_feature_sets --paired \
    --set independent=experiments/inference/features/paper_features.npz \
    --set shared=experiments/inference/features/shared_upp_paper_features.npz

# Reconstruction track
python -m experiments.reconstruction.generate_random_centres        # benchmark, ~2 min
python -m experiments.reconstruction.freeze_benchmark               # splits and checksums
python -m experiments.reconstruction.stage0_oracle                  # Stage 0, ~2 min
python -m experiments.reconstruction.stage1_baselines               # Stage 1, ~30 min
python -m experiments.reconstruction.generate_diffusion_patterns --split development   # Stage 5 data
python -m experiments.reconstruction.generate_diffusion_patterns --split test
python -m experiments.reconstruction.stage5_simulator               # Gate 5.1, ~1 min
python -m experiments.reconstruction.stage5_physics_fit --split test  # Gate 5.2, ~50 min (see E17 for the controls)
```

The data folders are gitignored and regenerable, about 5 GB each.

Report inference posteriors from an ensemble rather than a single flow. It costs
five training runs of a few seconds each, and it is what makes the coverage claim
correct.
