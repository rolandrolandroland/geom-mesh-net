# Amortized Bayesian Inference of Cluster Parameters

## Outline and roadmap

This document specifies a new line of work: recovering the *physical cluster
parameters* of a labeled 3D point pattern, with calibrated uncertainty, from
its spatial-summary features. It replaces the neural field as the primary
scientific objective. The neural field is retained as a secondary study
(Section 12).

Every number quoted below was measured on this repository's data on
2026-09-10 unless explicitly labeled as an estimate. Measurement commands are
given so each can be rechecked.

---

## 1. The problem

An experimentalist measures a labeled point cloud and wants to know the
physical structure that produced it: how concentrated the solute is inside
precipitates, how much sits in the matrix, how large the precipitates are, and
how variable their size is.

Current practice computes the spatial summary functions G, F, K and the
guest-to-host cross-G, compares them against a null model, and argues
qualitatively that clustering is present. That establishes *whether* structure
exists. It does not deliver the parameters, and it delivers no error bars.

The obstacle is that the forward map from physics to point pattern has no
tractable likelihood. There is no closed form for

> the probability of observing these 216,000 labeled points given
> (rho_c, rho_b, cr, rb)

so the posterior cannot be written down and sampled with MCMC. What *is*
available is a simulator that draws from that likelihood. That is precisely
the setting of simulation-based inference.

### The asset

This project owns the simulator. Therefore it knows the ground truth for
every pattern it has ever generated. Nobody analyzing a real measurement has
that. It converts an unanswerable inverse problem into a supervised one.

---

## 2. Precise problem statement

### 2.1 Parameters and prior

`deprecated_code/tester_scripts/data_factory.py` varies exactly four
parameters, drawn independently and uniformly. The prior is therefore known
exactly rather than assumed — an unusual luxury.

| Parameter | Prior | Physical meaning |
| --- | --- | --- |
| `rho_c` | U(0.2, 1.0) | guest concentration inside clusters |
| `rho_b` | U(0.0, 0.05) | guest concentration in the matrix |
| `cr` | U(3.0, 15.0) | mean cluster radius, in domain units |
| `rb` | U(0.0, 0.5) | relative spread of cluster radii |

So θ ∈ R⁴ and p(θ) = Uniform over that box.

Two parameters that appear parameter-like are **not** inferable:

- `pcp` is held constant at 0.1 across all 1,000 simulations. It carries no
  variation, so nothing can be learned about it. It must not be included in θ.
- `radii` stored in each `.npz` is a *realized output* of variable length
  (2 to 735 clusters per pattern), not a parameter. It is a consequence of
  `cr`, `rb` and the overlying point pattern.

### 2.2 The ground truth is recoverable but not stored

`data/pattern_stats.npy` has shape (1000, 9) and contains only:

| Column | Contents |
| --- | --- |
| 0, 1, 2 | measured `pcp`, true `pcp` (always 0.1), percent error |
| 3, 4, 5 | measured `rho_c`, true `rho_c`, percent error |
| 6, 7, 8 | measured `rho_b`, true `rho_b`, percent error |

**`cr` and `rb` were never written to disk.** They are recoverable because the
factory seeds `np.random.default_rng(42)` and draws in a fixed order. Replaying
that sequence reproduces the saved columns to *zero* error:

```
rho_c (col 4): max abs diff = 0.0
rho_b (col 7): max abs diff = 0.0
```

which certifies that `cr` and `rb` from the same replay are also exact.

This recovery depends on `data_factory.py` remaining byte-identical in its RNG
usage. That is a fragile guarantee, so Stage 0 persists θ to disk and never
relies on the replay again.

### 2.3 Observation and summary statistic

`x` is a labeled point cloud: 216,000 points in a 60 × 60 × 60 domain, labels
0 and 1 for host, 2 and 3 for guest.

`s = f(x) ∈ R¹⁴` is the existing feature vector from
`geom_mesh_net/core_functions/paper_spatial_features.py`, compressing 216,000
points to 14 numbers.

An important simplification: **inference does not need the null model.** The
observed-minus-expected framing exists to test the hypothesis "structure is
present" against a null. A posterior estimator needs only a deterministic
statistic of `x`. Retaining the null subtraction is fine — it centers the
features usefully and the code exists — but the expensive random-relabeling
Monte Carlo is off the critical path. Measured on pattern 1 (216,000 points):

| Null model | Per pattern | All 1,000 patterns |
| --- | ---: | ---: |
| `random_label`, 5 relabelings | 6.23 s | 1.73 h |
| `csr` (closed form) | 1.61 s | **0.45 h** |

and the two agree closely (`G_max_diff` 0.4444 vs 0.4425, `F_min_diff`
−0.3812 vs −0.3808). Use `csr`.

### 2.4 What is wanted

The posterior p(θ | s): a distribution over the four parameters given the
measured features, from which credible intervals follow.

---

## 3. Method: Neural Posterior Estimation

Fit a neural network q_φ(θ | s) that maps 14 features to a *distribution* over
4 parameters. Train by maximum likelihood on the simulated pairs:

> minimize over φ:  − Σᵢ log q_φ(θᵢ | sᵢ)

The load-bearing fact is that because the pairs (θᵢ, xᵢ) are drawn from the
joint p(θ)p(x|θ), the minimizer of that loss **is** the true posterior
p(θ|s). This is not an approximation of some different quantity; it is
Bayesian inference recast as conditional density estimation over samples
already in hand.

### Why a normalizing flow rather than a Gaussian head

A Gaussian head constrains every posterior to an ellipse. The posteriors here
are expected to be curved and correlated — a small number of dense clusters
and a large number of diffuse ones produce similar summary statistics, so
`rho_c` and `cr` should trade off along a ridge. A conditional normalizing
flow learns an invertible map from a simple base density to an arbitrary
shape, with the map's parameters a function of `s`, so it can represent that
ridge.

### Amortization

Training happens once. Inference on a new dataset is a single forward pass.
This matters because the alternative, per-dataset MCMC, needs a likelihood
that does not exist here.

### Reference implementation shape

```python
prior = BoxUniform(low=[0.2, 0.0, 3.0, 0.0], high=[1.0, 0.05, 15.0, 0.5])
theta = torch.as_tensor(np.column_stack([rho_c, rho_b, cr, rb]))  # (N, 4)
s     = torch.as_tensor(features)                                 # (N, 14)

inference = NPE(prior=prior)
posterior = inference.append_simulations(theta, s).train().build_posterior()
samples   = posterior.sample((10_000,), x=s_measured)             # (10000, 4)
```

The `sbi` package supplies this. Roughly 100 lines of PyTorch would too. The
work is not the flow — it is Stage 1 (features for 1,000 patterns) and
Stage 3 (validation).

---

## 4. Validation: the reason this is worth doing

A neural field reports "Brier 0.00697 versus 0.00841." Interpreting that
requires a threshold chosen by the analyst, which is why
`example_01/EXPERIMENTAL_METHODOLOGY_01.md` had to prespecify advancement
criteria by hand.

A posterior is self-checking against a distribution the analyst does not get
to choose.

### 4.1 Simulation-based calibration

For each held-out pattern the flow never trained on, draw L samples from
q(θ|sᵢ) and compute the **rank** of the true θᵢ among them, per dimension. If
q is the true posterior, those ranks are **uniform on {0, ..., L}** by
construction. Departures are diagnostic:

| Rank histogram shape | Diagnosis |
| --- | --- |
| Uniform | calibrated |
| Peaked in the centre | posteriors too wide (underconfident) |
| Peaked at both edges (U-shaped) | posteriors too narrow (**overconfident** — the dangerous failure) |
| Sloped / shifted | biased |

Report the rank ECDF against its uniform envelope for each of the four
parameters.

### 4.2 Coverage

Does the nominal 90% credible interval contain the truth in ~90% of held-out
patterns? Report empirical coverage at the 50%, 80%, 90% and 95% levels. This
is the claim an experimentalist actually cares about.

### 4.3 Sharpness, conditional on calibration

Among calibrated posteriors, narrower is better. Report median posterior
standard deviation per parameter, and the *contraction* relative to the prior:

> contraction = 1 − posterior sd / prior sd

Contraction near 0 means the data are uninformative about that parameter;
near 1 means sharply determined. **Sharpness is only meaningful once
calibration passes** — an overconfident flow is arbitrarily sharp and
worthless.

### 4.4 Why this ordering matters

Calibration is a gate, not a metric to optimize alongside others. A model that
fails SBC has not produced a worse posterior; it has not produced a posterior
at all.

---

## 5. Identifiability: what should and should not be learnable

Measured over all 1,000 patterns by `inference/recover_ground_truth.py`, so
expectations are set before any model is fit. This is what distinguishes a real
result from a fit.

| Relationship | Measured | Implication |
| --- | ---: | --- |
| corr(`cr`, log n_clusters) | **−0.906** | `cr` is strongly determined; expect sharp posteriors |
| corr(`rb`, realized mean radius) | **+0.003** | `rb` affects radius *variance*, not mean |

With a median of 8 clusters per pattern, there is almost no sample from which
to estimate a variance, and the measured +0.003 is indistinguishable from
zero. **`rb` is therefore expected to be weakly identifiable
or unidentifiable, and its posterior should come back close to its prior.**

This is a prediction, not an excuse. If the flow returns a *narrow* posterior
for `rb`, the flow is overconfident and SBC must catch it. If it returns
approximately the prior and SBC passes, the method is working correctly and
honestly reporting ignorance. Either outcome is informative; the point is that
the framework can tell them apart.

`rho_c` and `rho_b` sit between these extremes. `rho_c` governs guest
concentration inside clusters and should be recoverable from the G and cross-G
features. `rho_b` spans U(0, 0.05) against a realized guest fraction of
0.0967 ± 0.0237, so it is a small perturbation on a noisy quantity; moderate
contraction is the expectation.

### Measured outcome (Stage 1 complete)

`inference/screen_features.py` screens the extracted features with ridge
regression over 25 random 800/200 splits. Ridge gives only a point estimate,
with none of the posterior shape that motivates the flow, but it is a valid
lower bound: a parameter ridge cannot predict will not be rescued by a more
flexible model.

| Parameter | best R² | contraction | carried by |
| --- | ---: | ---: | --- |
| `rho_c` | **0.964 ± 0.037** | 0.803 | cross-G: `GXGH_min_diff` −0.87, `GXGH_95diff_r` +0.81 |
| `rho_b` | **0.925 ± 0.061** | 0.707 | F: `F_min_diff` +0.88, `F_min_diff_F` +0.81 |
| `cr` | **0.843 ± 0.025** | 0.608 | K: `Rdm` +0.77, `Tm` +0.76 |
| `rb` | 0.112 ± 0.045 | 0.047 | `Rdm` +0.18, and little else |

Each parameter is carried by its own feature family, and the mapping is
interpretable: guest-to-host cross-G for the in-cluster concentration, empty-space
F for the matrix concentration, K for the cluster length scale. **The five
K-derived features are the only ones carrying `cr` at all**, which is a concrete
argument for keeping them despite Section 8.3.

**The `rb` prediction was violated, narrowly.** It was registered as
"uninformative", operationalized as R² ≤ 0.10. Measured R² is 0.112 ± 0.045, and
positive on all 25 splits (range +0.01 to +0.18) — small, but not zero. The
ceiling has deliberately not been moved.

The substantive expectation survives: contraction is 0.047, so a calibrated
`rb` posterior should be about 95% as wide as its prior. The strict claim that
`rb` carries *no* information was too strong. It carries a little, apparently
through `Rdm`, which is plausible since radius spread should affect the shape of
the K derivative rather than its peak location.

This does not weaken the Stage 3 gate, it sharpens it: `rb`'s posterior should
be *slightly* narrower than the prior and no more. A markedly narrow one is
overconfidence, and SBC remains the arbiter.

### Data hygiene

- Realized guest fraction is 0.0967 ± 0.0237 against a target `pcp` of 0.1, so
  `pcp` is only approximately achieved.
- **2 of the 1,000 patterns contain zero clusters** (0.2%). These are
  legitimate draws from the prior predictive, not a bug, but they must be
  handled explicitly. Dropping them silently biases the training set toward
  large-`cr` patterns that happened to produce clusters. Stage 0 records a
  degeneracy flag per pattern; Stage 2 trains with them included and reports
  the count.

---

## 6. Staged roadmap

Each stage has an explicit gate. A stage that fails its gate stops the line of
work at that point rather than proceeding with a caveat. This mirrors the
discipline of `EXPERIMENTAL_METHODOLOGY_01.md`, which correctly halted on its
interpolation gate.

### Stage 0 — Persist ground truth (hours)

Recover θ by replaying the factory RNG, cross-check against the two saved
columns, and write it to disk with provenance. Also record per-pattern
descriptors needed later: cluster count, realized radius mean and standard
deviation, realized guest fraction, degeneracy flag.

**Gate:** replayed `rho_c` and `rho_b` reproduce `pattern_stats.npy` columns 4
and 7 to within floating-point equality. Non-negotiable — if this fails, the
RNG replay assumption is void and θ must be regenerated by rerunning the
simulator.

Implemented by `inference/recover_ground_truth.py`.

### Stage 1 — Feature extraction for all 1,000 patterns (~1 hour)

Compute the 14 global features for every pattern using the `csr` null, the
`sqrt` transform, and `k_r_max = 40.0` with `k_num_radii = 801` per
Section 8.3. Cache to a single `.npz` with full config
metadata and a configuration signature, following the pattern already
established in `paper_feature_experiments.py`.

Parallelize across cores. The loop is embarrassingly parallel and currently
serial; see Section 11.

**Gate:** all 14 features finite for at least 95% of patterns; the K extrema
interior-diagnostic (Section 8.3) true for at least 80% of patterns. Failure
here means the feature definitions are still degenerate and Stage 2 would
train on noise.

**Result: PASSED.** 1000/1000 patterns finite (100%), interior `Rm` 964/1000
(96.4%), `Rdm` 713/1000 (71.3%), `Rddm` 366/1000 (36.6%). Wall clock 7.6 minutes
across 7 workers, 6.9x speedup over 3.15 s of CPU work per pattern.

`Rddm` reaching an interior extremum in only 37% of patterns confirms it as the
weakest channel and the prime Stage 4 candidate for carrying no information.

One correction to Section 8.3: the 36 patterns that fail the interior-`Rm` check
at `k_r_max = 40` are the *small*-`cr` ones (mean `cr` 5.69 against 9.03 for the
rest), not the large ones. Under `sqrt` the reference grows as r^1.5, so a small
cluster's peak becomes a minor bump on a large rising curve. Raising `k_r_max`
therefore does not strictly dominate — it trades small-`cr` coverage for
large-`cr` coverage, and 40.0 happens to be a good compromise at 96.4%. A
per-pattern adaptive `k_r_max` would fix the remainder and is a Stage 4 item.

### Stage 2 — Fit the posterior (minutes)

Split 800 train / 100 validation / 100 test by pattern index. Fit a
conditional flow. Standardize features using training statistics only.

**Gate:** validation log-likelihood improves over a prior-only baseline —
that is, the flow learns something. A flow that cannot beat the prior means
the features carry no information about θ, which would be a genuine and
publishable negative result about the 14 features.

**Result: PASSED.** Validation log-likelihood 6.939 nats against a uniform
prior of 1.427, a gain of +5.512. Trained in 7.4 seconds over 75 epochs.

Held-out test set, never used for stopping or selection:

| Statistic | Value |
| --- | ---: |
| median log-likelihood | **7.400** nats |
| mean log-likelihood | **−187.576** nats |
| mean excluding one pattern | 6.581 nats |
| patterns below the prior | 6/100 |

Per-parameter contraction: `rho_c` 0.865, `rho_b` 0.901, `cr` 0.649,
`rb` **0.039**.

The `rb` result confirms the registered prediction almost exactly. Section 5
predicted a posterior "about 95% as wide as its prior"; the fitted posterior is
96% as wide. The flow honestly reports that the data do not constrain `rb`,
which is the behaviour a calibrated posterior must have and the thing a point
estimate cannot express at all.

#### The mean and the median disagree, and the reason matters

The entire gap between a median of 7.4 and a mean of −188 is **one pattern**:
index 613, at −19,409 nats.

Pattern 613 has **zero clusters**. It is one of only two such patterns in the
1,000 (Section 5, Data hygiene), and it landed in the test set. Its true
parameters say `cr = 12.27` and `rho_c = 0.988` — large, dense clusters — while
the realized pattern contains none, so its features look like complete spatial
randomness. Its feature vector is not an outlier in the ordinary sense: its most
extreme channel is 3.9 training standard deviations, against a dataset 99th
percentile of 4.2. It is a *structural* outlier, and the flow saw exactly one
example of that structure during training.

This is not a defect in the flow. It is a real property of the joint
distribution, and it is informative: a pattern with no clusters carries no
information about cluster radius, so an honest posterior should widen toward
the prior there. Instead the flow is confidently wrong. That is precisely the
overconfidence failure Stage 3 exists to detect, and it should appear in the
coverage numbers.

The mean was not quietly replaced with the median. A log-density of −19,409
means the model assigned an essentially impossible density to something that
actually happened, and averaging that away would be exactly the kind of
threshold-shopping this document tries to avoid. Both statistics are reported,
with the cause named.

Two consequences for later stages:

1. It is direct evidence for the sample-size limitation in Section 7. Rare
   regions of the joint are unrepresented at n = 1,000, and no amount of
   architecture tuning fixes an unseen structure.
2. Stage 3 should report coverage both over all test patterns and excluding
   degenerate ones, so a single structural outlier neither hides a real
   calibration failure nor manufactures one.

### Stage 3 — Calibration and coverage (hours)

Run SBC and coverage on the held-out test patterns per Section 4.

**Gate — the decisive one:** SBC rank ECDFs lie within their uniform envelope
for `rho_c` and `cr`, and empirical 90% coverage falls in [0.85, 0.95] for
those two parameters. `rb` is expected to be uninformative and is not gated on
sharpness, only on calibration.

If this gate fails, the ordered remedies are: (a) more simulations, (b)
observation augmentation per Section 7, (c) a larger flow, (d) reconsider the
summary statistic. Do not proceed to Stage 4 with an uncalibrated posterior.

**Result: PASSED.**

Assessed by 10-fold cross-validation rather than on the 100-pattern Stage 2 test
set. Only data a flow never trained on is admissible, and 100 ranks give a
Kolmogorov band of ±0.136 — wide enough to accept badly miscalibrated
posteriors. Refitting per fold gives an out-of-fold posterior for all 1,000
patterns and narrows the band to ±0.043, for 66 seconds of compute. Each fold is
a different flow, so this measures the calibration of the *procedure*, which is
the right target for a method being proposed.

| Parameter | max ECDF deviation | band ±0.0429 | 90% coverage | gated |
| --- | ---: | --- | ---: | --- |
| `rho_c` | 0.0305 | uniform | **0.888** | yes — pass |
| `cr` | 0.0305 | uniform | **0.867** | yes — pass |
| `rb` | 0.0295 | uniform | 0.876 | calibration only — pass |
| `rho_b` | 0.0475 | **departs** | 0.847 | no |

**The headline result is `rb`.** It is the parameter the data barely constrain —
R² 0.11, contraction 0.039, a posterior 96% as wide as its prior. Its ranks are
uniform and its 90% interval covers 87.6% of the time. The method reports honest
ignorance and its error bars are trustworthy *while* being uninformative. That
is the property no point estimate can have, and it is the reason this approach
was worth the detour from the neural field.

#### `rho_b` is mildly miscalibrated, and it is not the degenerate patterns

`rho_b` breaches the band at 0.0475 against 0.0429 — 1.11× the band, so a real
departure but a small one. Its 90% coverage is 0.847, just under the 0.85 floor
the other parameters were gated on. Two diagnostics agree on the direction:

- Its rank ECDF difference is a smooth *arch* peaking near rank 0.3, not noise.
  Ranks pile at low values, so the truth sits below the posterior centre more
  often than it should: the posterior is biased slightly high. Mean normalized
  rank is 0.477 against 0.500.
- Its coverage curve lies below the diagonal at every level, which is mild
  overconfidence.

It is tempting to attribute this to the zero-cluster patterns that dominated the
Stage 2 mean. That is wrong: excluding both of them moves the deviation from
0.0475 to 0.0462, still outside the band. Pattern 613 was a separate,
single-pattern problem. `rho_b`'s miscalibration is systematic across the
dataset.

A plausible mechanism is that `rho_b` spans U(0, 0.05) against a realized guest
fraction of 0.0967 ± 0.0237, so it is a small perturbation on a noisy quantity
and the flow slightly over-trusts the F-feature signal that carries it. This is
recorded rather than patched. The honest statement is that three of four
parameters are calibrated, and `rho_b`'s intervals should be read as
approximately 85% rather than 90% until it is fixed.

`rho_b` was not in the pre-registered gate, and it has not been added to or
removed from it after the fact.

### Stage 4 — Feature sufficiency (days)

`walkthroughs/clustersim_todo.md` already asks "which features capture the
most information?" This stage answers it quantitatively rather than by
intuition: refit with feature subsets and compare posterior contraction per
parameter. Features that sharpen the posterior carry information about θ;
features that do not, do not.

Specifically test whether the five K-derived features (`Tm`, `Rm`, `Rdm`,
`Rddm`, `Tdm`) contribute anything once `k_r_max` is set properly. Section 8.3
shows they were largely degenerate before. `Rddm` reaches an interior extremum
in only 8 of 16 patterns even at `k_r_max = 40` and is the prime candidate for
carrying no information.

This is also the natural place to run `k_transform="cube_root"` as a controlled
comparison against the `sqrt` default (Section 8.2), since posterior
contraction gives an objective criterion for which transform carries more
information about θ — something the choice has never been tested against.

**Gate:** none. This is a descriptive study.

### Stage 5 — Scale the simulation budget (days, mostly compute)

If Stage 3 fails on sample size, or Stage 4 suggests the flow is
capacity-limited, rerun the factory with `n_sims` of 10,000 or more. This is
embarrassingly parallel and needs no new science. Section 8.5 fixed the
generator, which previously could not run, and it now writes all four
parameters.

### Stage 6 — Real measured data (open-ended, separate project)

See Section 10. This stage is **not** a continuation; it is a new research
problem gated on Stages 0–3.

---

## 7. Sample size and observation augmentation

1,000 simulations is small for NPE, which is conventionally run with 10⁴–10⁵.
Four parameters and fourteen summaries is a small problem, so 1,000 may
suffice, but this is the most likely reason for Stage 3 to fail.

Two remedies, in order of cost:

**Observation augmentation (free).** Draw several independent random thinnings
of each pattern and compute features for each. This yields multiple `s` per θ,
which is legitimate — they are genuine draws from p(s|θ) — and does double
duty by teaching the flow the *observation* noise in `s`, not merely parameter
variation. It directly widens posteriors toward honesty, which is the failure
mode most likely to break SBC.

A caveat that must be respected: augmented replicates of the same pattern are
**not independent**. Splitting must be **by pattern index**, never by
replicate, or the test set leaks and every calibration statistic is invalid.

**More simulations (compute).** Stage 5.

---

## 8. Corrections to the feature library that this work required

Investigating the K features for this roadmap turned up defects that affect
the existing results as well. They are fixed on this branch.

### 8.1 The K estimator itself is correct

`_translation_corrected_k` was validated against the analytic CSR value
K(r) = (4/3)πr³ on homogeneous Poisson patterns. Mean ratio of estimate to
truth over r ∈ [1, 10] was **1.0009** — unbiased. The estimator is not the
problem. Encoded as a regression test.

### 8.2 The K transform is `sqrt`, by decision

The code computes `sqrt(K_obs) - sqrt(K_exp)`. Under CSR in d dimensions K(r)
is the volume of a radius-r ball, so the transform that linearizes K against r
is the inverse of that volume: `sqrt` in 2D (K_csr = pi r^2), cube root in 3D
(K_csr = (4/3) pi r^3). On that basis alone `cube_root` would be the
variance-stabilizing choice for this data.

**`sqrt` is nonetheless retained as the default**, because these five features
are a port of the Bennett et al. definitions and matching the published feature
semantics takes precedence over the textbook transform. Changing it would make
`Tm`, `Rm`, `Rdm`, `Rddm` and `Tdm` quantities that no longer correspond to the
paper's, and would silently break comparability with
`example_01/methodology_01_results` and
`example_01/global_paper_feature_validation`.

The transform is now selectable via `PaperFeatureConfig.k_transform`, so
`cube_root` is available for a controlled comparison — a reasonable Stage 4
experiment — without being imposed.

The consequence that *does* need acting on is the interaction with `k_r_max`.
Because sqrt(K_csr) grows as r^1.5 rather than r, difference curves under
`sqrt` keep rising further out and their extrema sit at larger radii. A small
`k_r_max` therefore produces no interior extremum at all, which is Section 8.3.

### 8.3 The dominant defect: extrema pinning to the grid boundary

`_extract_k_features` returns `radii[index]` for its three radius-valued
features. When the transformed difference curve has no interior extremum — it
is still rising at `k_r_max` — the peak finder falls back to `argmax`, which
returns the first or last grid point. The function then silently returns 0.0 or
`k_r_max` as though it were a measurement.

With the `sqrt` transform this is governed almost entirely by `k_r_max`.
Fraction of 16 patterns with a genuinely interior extremum:

| `k_r_max` | r/L | Rm | Rdm | Rddm | median Rm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 (old default) | 0.17 | 5/16 | 4/16 | 3/16 | 4.8 |
| 20 | 0.33 | 9/16 | 9/16 | 4/16 | 9.5 |
| 25 | 0.42 | 11/16 | 11/16 | 4/16 | 11.5 |
| 30 | 0.50 | 13/16 | 12/16 | 6/16 | 13.5 |
| 40 | 0.67 | **16/16** | 12/16 | 8/16 | 15.9 |
| 50 | 0.83 | 16/16 | 12/16 | 12/16 | 15.9 |

At the old default of `k_r_max = 10`, roughly two thirds of K features were
boundary artifacts rather than measurements. **This is the most likely
explanation for why `Tm`, `Rm`, `Rdm`, `Rddm` and `Tdm` had the worst
interpolation Spearman correlations (0.53 to 0.72) in methodology 01, while the
G, F and cross-G features scored 0.90 to 0.98.** Interpolating a boundary
artifact cannot succeed, because the quantity is not a smooth function of
position.

#### Raising `k_r_max` costs nothing in feature quality

A naive reading of the correlation between `Rm` and the true cluster radius
suggests large `k_r_max` is harmful: over whichever patterns happen to be
interior, corr(`Rm`, `cr`) falls from 0.954 at `k_r_max = 25` to 0.671 at 40.

That is **selection bias, not degradation.** Restricting to the 13 of 24
patterns that have an interior `Rm` at *every* setting, the correlation is
identical to three decimal places at every setting:

| `k_r_max` | n interior | corr over interior patterns | corr over the common subset |
| ---: | ---: | ---: | ---: |
| 15 | 13/24 | 0.864 | **0.863** |
| 20 | 15/24 | 0.918 | **0.863** |
| 25 | 17/24 | 0.954 | **0.863** |
| 30 | 20/24 | 0.724 | **0.863** |
| 40 | 24/24 | 0.671 | **0.863** |

The common subset spans `cr` from 3.2 to 9.3 only. Small-cluster patterns reach
an interior extremum at small `k_r_max`; large-cluster patterns need a large
one. Raising `k_r_max` admits the harder large-`cr` patterns, which lowers the
pooled correlation while leaving every individual feature exactly as good. The
0.671 figure is the more honest number, measured over a wider and harder set,
not a worse feature.

**Conclusion: use the largest `k_r_max` the estimator supports.** For this
dataset — a 60-unit domain, cluster radii to 15 — Stage 1 uses
**`k_r_max = 40.0` with `k_num_radii = 801`**, giving 16/16 interior `Rm` at a
measured 1.04 s per pattern (about 17 minutes for 1,000 patterns serially, a
few minutes across 8 cores). `Rddm` remains the weakest channel at 8/16 and
should be expected to carry the least information in Stage 4.

Two changes follow:

1. `PaperFeatureResult` and `LocalPaperFeatureResult` now carry
   `k_extrema_interior`, a boolean triple for Rm, Rdm and Rddm. The failure is
   visible rather than silent, and Stage 1 gates on it.
2. `k_r_max` must be set from the physical cluster scale, not left at the
   default of 10.

### 8.4 Known fragility: the extractor takes the *first* local maximum

`_extract_k_features` selects `peaks[0]`, the first local maximum of the
LOESS-smoothed curve, rather than the global maximum. The smoothing span is
then chosen adaptively from an initial peak estimate. For a peak narrow
relative to that span the second smoothing pass rings and introduces a
spurious local maximum at small r, which is selected in preference to the true
peak. On a synthetic Gaussian of width 0.63 over a 10-unit range, the initial
pass locates the true peak at r = 4.0 correctly and `Rm` is nonetheless
reported at r = 0.85.

This was **not changed.** Taking the first peak is plausibly the intended
definition -- the first peak of a K difference is the primary cluster scale --
and altering it would silently change the feature semantics. Real patterns at
the recommended settings produce broad peaks and are unaffected. The behaviour
is pinned by
`tests/test_paper_spatial_features.py::test_extractor_takes_the_first_local_maximum_not_the_global_one`
so that any future change is deliberate rather than accidental.

If Stage 4 finds the K features uninformative even after the Section 8.2 and
8.3 corrections, this is the next thing to examine.

### 8.5 The data generator could not run

`deprecated_code/tester_scripts/data_factory.py` imported
`core_functions.clustersim`, a flat path that stopped resolving when the project
became a package (commit 31357ca, 2026-05-18). **The script that generates the
entire dataset therefore could not be executed at all**, which blocks Stage 5
outright.

Fixed to import `geom_mesh_net.core_functions.clustersim`, and it now also
writes `cr` and `rb` as columns 9 and 10 of `pattern_stats.npy`, appended so
every existing column index stays valid. Future datasets will not need the RNG
replay.

Verified by regenerating three patterns from scratch and comparing against the
replay:

```
rho_c  col  4  max abs diff = 0.000e+00
rho_b  col  7  max abs diff = 0.000e+00
cr     col  9  max abs diff = 0.000e+00
rb     col 10  max abs diff = 0.000e+00
```

This is an independent confirmation of Section 2.2: the recovered `cr` and `rb`
match values produced by actually rerunning the simulator, not merely the two
columns the old script happened to save.

### 8.6 `k_r_max` upper bound

The prior audit note claimed the translation correction "explodes" for large r.
That was wrong, and measurement is clearer. On CSR in a 60-unit cube the
estimator stays unbiased right up to r = L:

| r | r/L | estimate / analytic | relative sd |
| ---: | ---: | ---: | ---: |
| 15 | 0.25 | 0.999 | 0.009 |
| 30 | 0.50 | 0.998 | 0.010 |
| 60 | 1.00 | 1.000 | 0.003 |
| 70 | 1.17 | **0.913** | 0.012 |

The failure is specific to **r > L**, and it biases *downward* by about 9%
rather than exploding. Pairs separated by more than the domain side can only
exist along diagonals, and the `overlap > 0` filter discards them
inconsistently.

`PaperFeatureConfig` now rejects `k_r_max` greater than the shortest domain
side, checked where the domain becomes known. The conventional r ≤ L/4 guidance
concerns interpretability and variance for clustered patterns rather than
unbiasedness under CSR, so it is documented but not enforced — Section 8.3
shows this dataset genuinely needs r_max = 20 > 15.

Note that `example_01/global_paper_feature_validation/validation_config.json`
used `k_r_max = 70.0` in a 60-unit domain, so **the K features in that
validation output are affected** and should be regenerated.

---

## 9. Deliverables

The package is `inference/`, not `sbi/`. A local `sbi/` directory shadows the
`sbi` pip package that Stage 2 depends on — `from sbi.inference import NPE`
fails outright with `PYTHONPATH=.` set — so the name was changed before any
imports were written against it.

| Path | Contents |
| --- | --- |
| `inference/ROADMAP.md` | this document |
| `inference/recover_ground_truth.py` | Stage 0 |
| `inference/ground_truth/` | persisted θ and per-pattern descriptors |
| `inference/extract_features.py` | Stage 1 |
| `inference/features/` | cached features (`.npz` gitignored, `.json` tracked) |
| `inference/screen_features.py` | ridge screen of feature informativeness |
| `inference/flow.py` | conditional autoregressive flow, written here not imported |
| `inference/fit_posterior.py` | Stage 2 |
| `inference/posterior/` | fitted flow, test posterior samples, metadata |
| `inference/validate_posterior.py` | Stage 3 |
| `tests/` | regression tests for the feature library |

Reproducibility follows the standard already set by
`EXPERIMENTAL_METHODOLOGY_01.md` Section 15: record configuration signatures,
seeds, split indices, per-epoch metrics, and wall-clock timings for every
stage.

---

## 10. Risks and honest limitations

**Sample size.** 1,000 simulations is at the low end for NPE. Section 7.

**`rb` is probably unidentifiable.** Section 5. Predicted in advance so it
cannot be rationalized afterward.

**Summary-statistic sufficiency is unproven.** The 14 features were validated
as numerically sound — no NaNs, sane ranges — *not* as sufficient statistics
for θ. If they discard information, posteriors will be honestly wide rather
than wrong: SBC still passes, contraction is just poor. Stage 4 measures this,
and the fallback is to feed the raw G/F/K curves or learn an embedding.

**The simulator defines what is being inferred.** A flow trained on
`clustersim` output infers `clustersim`'s parameters. This is the binding
limitation on Stage 6.

**Real data is a separate research problem, not a final step.** The earlier
framing — "run it on a real dataset and report cluster radius with a 90%
interval" — understated this. It presumes two things not established here:
that real measured data is available, and that the simulator is faithful
enough to transfer. A real measurement carries detector efficiency, trajectory
aberration, and non-spherical, non-Gaussian precipitate morphology that
`clustersim` does not model. A flow trained on simulation and applied to real
data will produce confident, calibrated-looking, wrong answers, and SBC on
simulations cannot detect that, because SBC only ever validates
self-consistency *within* the simulator.

Addressing it needs model misspecification work in its own right: robust or
noise-aware summary statistics, prior predictive checks against real data to
confirm the simulator can even produce patterns resembling it, and ideally a
measurement with independently known ground truth.

**The sim-to-sim result stands on its own.** Calibrated posteriors over
simulated patterns are a real, self-contained, publishable contribution, and
also the honest prerequisite for anything on real data. Stages 0–4 are
worthwhile even if Stage 6 never happens.

---

## 11. Known performance work

`calculate_local_paper_features` loops serially over query points
(`for query_index, query_point in enumerate(queries)`), each iteration
independent. Methodology 01 spent 34 minutes here on 8 cores. Parallelizing
would cut it to roughly 6 minutes.

Not on the critical path for this roadmap, which needs only *global* features,
but it is the single largest remaining inefficiency and it blocks the local
feature experiments in Section 12.

---

## 12. Relationship to the neural field work

The neural field is not abandoned. It answers a different question — dense
field reconstruction rather than parameter inference — and it retains one
unresolved thread worth pursuing independently.

Methodology 01 concluded that local features "do not advance." Re-examining
its own numbers against a constant-predictor baseline on the held-out voxels:

| Pattern | Constant Brier | Coordinate | Local | Coordinate vs. constant |
| --- | ---: | ---: | ---: | --- |
| 0 (2 large clusters) | 0.02947 | 0.000145 | 0.000469 | 200× better |
| 1 (10 clusters) | 0.04121 | 0.003666 | 0.005062 | 11× better |
| 4 (170 small clusters) | 0.00954 | 0.008408 | 0.006968 | **1.13× better** |

On patterns 0 and 1 the coordinate control had essentially solved the task, so
no feature set could improve on it and "features do not help" is uninformative
there. On pattern 4 the coordinate model barely beat a constant, and that is
exactly where local features won clearly: Dice 0.215 → 0.402, PR-AUC
0.201 → 0.358.

The conclusion about *interpolation* was sound and correctly halted the work.
The conclusion about *features* was an artifact of pattern selection: the
median was taken over one unsolved problem and two already-solved ones.

Two confounds must be removed before that comparison is rerun:

1. **No positional encoding exists anywhere in the codebase.** The field is a
   plain ReLU MLP on raw normalized coordinates. Pattern 4's clusters have
   radius ≈ 4.4 in a 60-unit domain, precisely the regime where ReLU spectral
   bias bites. The one pattern where features appeared to help may simply be
   the pattern where the control was crippled. Adding Fourier features to the
   baseline is a prerequisite for any clean feature comparison.
   `papers/How to Train Neural Field Representations` covers this directly.
2. **The K features were largely degenerate.** Section 8.3. Any rerun should
   use the corrected settings.

Also note that pattern 0's reported `pearson = 0.9976` sits alongside
`spearman = 0.391`. Pearson is being carried by a handful of high-value cluster
voxels and should not be a headline metric.

---

## 13. Immediate next actions

**Stages 0 through 3 are complete and every gate passed.** The core claim of
this roadmap is established: physical cluster parameters can be recovered from
classical spatial-summary features with calibrated uncertainty.

| Stage | Outcome |
| --- | --- |
| 0 — ground truth | θ recovered and persisted, exact on all 1,000 patterns |
| 1 — features | 100% finite, 96.4% interior `Rm`, 7.6 min on 7 workers |
| 2 — fit | validation 6.939 nats against a 1.427 prior |
| 3 — calibration | `rho_c` and `cr` uniform and covering; `rb` calibrated while uninformative |

Supporting work: feature-library corrections (Section 8), a repaired data
generator (8.5), and 117 regression tests including a flow checked against a
closed-form posterior and calibration diagnostics checked against deliberately
miscalibrated inputs.

Open, in rough priority order:

1. **`rho_b` miscalibration** (Section 6, Stage 3). Small, systematic, and not
   caused by the degenerate patterns. The cheapest probe is whether it survives
   more simulations, since a mild bias can be a small-sample artifact.
2. **Stage 4 — feature sufficiency.** Now well posed: posterior contraction
   gives an objective ranking. Two specific questions are ready to settle —
   whether `Rddm` contributes anything (interior in only 37% of patterns), and
   whether `cube_root` beats `sqrt` (Section 8.2), which the project has argued
   about but never measured.
3. **Stage 5 — more simulations.** Motivated concretely rather than
   speculatively: pattern 613 showed that rare structures are unlearnable at
   n = 1,000, and item 1 may be a sample-size artifact too.
4. **Observation augmentation** (Section 7). Free, and it would tell the flow
   about observation noise rather than only parameter variation. Replicates must
   stay on one side of the split.
5. **Stage 6 — real data.** Still a separate research problem, not a next step.
   See Section 10.

Also open, unrelated to inference:

- Whether to regenerate `example_01/global_paper_feature_validation/`, whose
  `k_r_max = 70.0` in a 60-unit domain is now refused outright (Section 8.6).
  Its K features are affected; its G, F and cross-G features are not.
- The neural field thread in Section 12, whose feature comparison needs Fourier
  features in the baseline before it can be rerun cleanly.
