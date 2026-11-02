# E21 Constrained geometry and physical inference

30 October 2026.

**Execution status:** constrained fitting, boundary proposals, training diagnostics
and the E21 runner are implemented. **The scientific workflow stopped at its
numerical prerequisite**. Both four-case B
preflights (versions 1 and 2) obtained zero converged fits. Version 3 corrected the
round-budget handling and tested development case 0 through all three prescribed
starts and rounds; it also obtained zero converged fits, in 348.813 seconds. The
four-arm comparison has not run. Capillary profiles and calibrated uncertainty
remain unimplemented, and confirmation and reserved-test data remain unopened.

The implemented study compares Gaussian and Fourier boundary initializations on one
fixed list of detected precipitates. Its immediate prerequisite is a fitter that
respects its physical domain and actually converges on the development cases. Passing
software tests does not satisfy that experimental prerequisite.

Implementation plan: [reliable physics inference](../guides/physics_inference_implementation_plan.md) ·
Design: [e21_design.json](../../experiments/reconstruction/e21_design.json) ·
Runner: [e21_geometry.py](../../experiments/reconstruction/e21_geometry.py) ·
Protocol: [e21_protocol.py](../../experiments/reconstruction/e21_protocol.py)

## Version 1 numerical preflight

The [saved preflight summary](../../experiments/reconstruction/results/e21_geometry_v1/preflight_summary.json)
records the complete attempted scope. Every case used three prescribed physical
starts. None produced a feasible, converged, assignment-stable solution eligible
for selection.

| Development case | Detected candidates | Fitting atoms | Fitting seconds | Recorded failure |
| --- | ---: | ---: | ---: | --- |
| 0 | 95 | 64,064 | 23.285 | All three starts proposed invalid raw probabilities; each was rejected before likelihood evaluation |
| 1 | 87 | 63,682 | 25.546 | All three starts proposed invalid raw probabilities; each was rejected before likelihood evaluation |
| 2 | 91 | 64,134 | 3.600 | All three starts hit the optimizer's initial feasibility check at the constraint boundary |
| 3 | 57 | 63,973 | 1.244 | Two starts had no feasible sampled chemistry; one hit the initial feasibility check |

The boundary initialization failures are consistent with the diagnosed discrepancy
between a tiny tolerated numerical constraint residual and the optimizer's strict
`keep_feasible` check. They remain failures in the archived record. Domain rejection
protected the likelihood calculation but did not make the optimization successful.

This is a failed **numerical prerequisite**, not evidence that Fourier refinement
does or does not improve physical inference. No neural model was trained for this
comparison. The total above measures attempted physical fitting; it does not estimate
the cost of successful convergence or a full neural study.

## Version 2 numerical preflight

The [version 2 summary](../../experiments/reconstruction/results/e21_geometry_v2/preflight_summary.json)
retains all four cases and all twelve prescribed starts. It uses the same development
realizations as version 1, so these are numerical diagnostics rather than independent
confirmation evidence.

| Case | Fitting seconds | Start outcomes |
| --- | ---: | --- |
| 0 | 478.821 | Three starts reached the first round's 150-iteration cap |
| 1 | 343.547 | Two starts reached that cap; one exhausted 600 evaluations |
| 2 | 338.601 | Two starts reached that cap; one exhausted 600 evaluations |
| 3 | 87.211 | One start reached that cap; two had no feasible sampled initial chemistry |

The guard rejected 1,724 unsafe trial points and scored **zero invalid likelihoods**.
This supports the guard's operation on these trajectories, not optimizer reliability.
The completed first-round stationarity residuals were about 0.00012–0.00044 against
a required 0.00001. Final nearest-particle assignments were not stable.

Review also found that a safe per-round cap ended a start instead of allowing the
remaining declared rounds. The version 3 correction continues such a round only
while the original three-round and 600-evaluation budgets permit. A cap never counts
as convergence; domain failures, stalled searches and exhausted budgets remain
failures. A separate derivative optimization recomputes probability-extremum
Jacobians on their active points, preserving tied extrema and moving surface probes.
Its equivalence tests do not establish convergence on the full cases.

## Version 3 bounded check and stopping decision

The [version 3 summary](../../experiments/reconstruction/results/e21_geometry_v3/preflight_summary.json)
records **one requested development case**, not a completed four-case preflight.
Case 0 contains 95 detected candidates and 64,064 fitting atoms. All three starts
completed three rounds of 150 iterations. The total per-start evaluation ceiling
remained 600, and the stationarity threshold remained 0.00001.

| Start | Evaluations | Final stationarity residual | Assignment-stability check | Outcome |
| --- | ---: | ---: | --- | --- |
| Default | 465 | 0.00009807 | Passed | Round budget exhausted; not converged |
| `(ell, xi) = (1, 3)` | 480 | 0.00018602 | Failed | Round budget exhausted; not converged |
| `(ell, xi) = (6, 12)` | 472 | 0.00013889 | Passed | Round budget exhausted; not converged |

The final states satisfied the checked geometry and probability conditions on the
fitting/probe domain. Across 1,417 evaluations, 18 unsafe trial points were rejected
and zero invalid likelihoods were scored. Passing assignment stability allows small
assignment changes when their objective and gradient effects meet the declared
criterion; it does not mean every atom's assignment is identical.

**Decision:** stop expansion. No start qualifies for fit selection. The remaining
version 3 preflight cases, neural comparison, confirmation and reserved test were
not run. Version 3's `preflight_ready` remains false, and even a successful one-case
subset could not satisfy the required four-case accounting. The improvement in
loss is not a convergence or identifiability certificate.

All three versions retain their reports and exact source snapshots; every one of
the 98 source hashes in each archive was verified. The same development realization
was reused to diagnose the implementation, so version 3 is not independent
confirmation evidence. No performance claim is inferred by comparing runtimes
across the different completed budgets.

### Immediate numerical work

The next development version should isolate which fitted parameters cause the
large-case convergence problem before further neural training:

1. Hold the **observed** candidate centres and prepared radii fixed, fit chemistry
   and the interior profile, and require the same fresh-objective convergence
   certificate. Separately use a model-matched simulated field with known parameters
   to verify recovery; keep any truth-assisted diagnostic out of promotion.
2. If that converges, release radii with centres fixed, then release centres. Record
   the stationarity contribution of each parameter block and the effects of updated
   nearest-particle assignments at each stage.
3. Use those diagnostics to choose a solver or parameter-scaling revision. Preserve
   failures and version any changed settings. Recheck all four baseline cases before
   resuming the four-arm comparison.

This is a proposed diagnostic sequence, not a fitted method already validated by
E21. The current evidence does not establish whether the difficulty primarily comes
from parameter scaling, constrained directions, assignment changes, or the mismatch
between the generating and fitted observation models.

## Research question

Does a neural concentration field improve the boundaries and physical parameters
estimated from the **same observed particles**, after correcting the numerical
failures of the earlier physics fitter?

[E20](E20_field_to_particles.md) improved concentration prediction without improving
particle detection or physical inference. E21 therefore holds particle detection
fixed and tests boundary initialization separately. A successful neural result would
establish an advantage for this procedure and budget on this simulated distribution.
If both initializations converge to the same optimum, the data may contain the same
usable physical information despite different initial boundary estimates.

## Implemented components

| Component | Responsibility |
| --- | --- |
| [e21_simulation.py](../../experiments/reconstruction/e21_simulation.py) | Fresh geometry, positions, dependent interior labels, thinning, observation splits, and immutable dataset identity |
| [constrained_joint_fit.py](../../geom_mesh_net/fields/constrained_joint_fit.py) | Geometry preparation, versioned field model, guarded likelihood, constrained optimization, and distinct numerical diagnostics |
| [boundary_refinement.py](../../geom_mesh_net/fields/boundary_refinement.py) | Stable candidate identities, common radial samples, boundary proposals, and deterministic fallback |
| [field_diagnostics.py](../../geom_mesh_net/neural/field_diagnostics.py) | Neural histories, selected and terminal checkpoints, refit artifacts, and development-only training diagnostics |
| [e21_protocol.py](../../experiments/reconstruction/e21_protocol.py) | Source and runtime identity, atomic reports, attempt accounting, numerical readiness, and promotion decisions |
| [e21_geometry.py](../../experiments/reconstruction/e21_geometry.py) | Phase sequencing, shared catalogue, fitting, scoring, interrupted-attempt recovery, and reserved-test access |

These interfaces implement the numerical and comparison stages. Capillary profiles,
calibrated uncertainty, and the measurement-limit study remain later work.

## Fresh simulation and observation design

E21 uses a new fully contained sphere generator, version
`e21-full-sphere-fixed-count-v1`. Each case contains 216,000 uniformly located atoms
in a box of side 60. The parameter ranges are fixed in the design: mean radius
3–4, relative radius spread 0.2–0.4, target volume fraction 0.08–0.12, interior guest
fraction 0.4–0.9, matrix mean 0.05–0.15, capillary length 2–5, and critical-radius
fraction 0.6–0.8. Lengths use the simulator's units.

The sphere count is determined from the target volume fraction and the analytic
third moment of the untruncated radius distribution. Radii below 2 are redrawn under
a bounded truncation procedure. Spheres are placed from largest to smallest, fully
inside the box, with at least a one-unit surface gap. The realized volume fraction
is recorded separately. Placement failure is an outcome, not permission to draw a
more favorable replacement case.

The matrix uses the existing surface-average screened field, scaled to the requested
mean at the simulated matrix positions. Raw probabilities must lie in [0, 1]; invalid
generating fields are rejected without clipping or replacement. Each particle has a
fixed guest count, sampled without replacement using distance-dependent weights.
This preserves the earlier simulator family's dependence between interior marks.
It also preserves a sharp generating interface. The corrected fitter's smooth
interface is therefore deliberately not an exact model of these observations.

The geometry distribution differs from E20's historical generator because complete
spheres are now required. E21 outcomes must be described as results on this new
distribution, rather than numerical replications of the earlier patterns.

Independent named streams control parameters, geometry, atom positions, labels,
thinning, the outer split, neural randomness and resampling. Case identity includes
the phase and stream identities, so development case 0 and confirmation case 0 are
different realizations. No historical dataset is read or modified. E20 patterns
74–81 remain governed by E20's failed promotion decision.

Thinning retains each atom with probability 0.37. Before fitting reads any label,
20% of retained atoms are assigned to the held-back set. About 29.6% of the original
atoms therefore enter detection and physical fitting. An inner split of only that
fitting set selects the neural field; the selected procedure refits on all outer
fitting atoms. Held-back and removed labels do not choose parameters, checkpoints,
proposals, optimizer starts or fallback decisions.

## Four arms with one particle catalogue

The Gaussian significance detector supplies the initial catalogue from the outer
fitting observations. Its candidate IDs, count, centres and observation fingerprint
are shared by every arm. Initial precision and recall are consequently invariants,
rather than promotion metrics. Particle birth, deletion and merging are disabled.

| Arm | Initial radii | Physical fit |
| --- | --- | --- |
| A | Original Gaussian detector radii | Historical unconstrained implementation |
| B | Original Gaussian detector radii | Corrected constrained implementation |
| C | Gaussian boundary proposals, otherwise original radii | Same corrected implementation |
| D | Fourier boundary proposals, otherwise the corresponding C radii | Same corrected implementation |

A and B receive the same identity-preserving initial geometry preparation. The
preparation holds centres fixed and minimizes changes to radii subject to bounds,
containment and pairwise separation. An impossible catalogue fails instead of losing
a difficult particle. The corrected fit subsequently frees both centres and radii,
retaining their original identities.

The two fields are evaluated on the same deterministic radial samples: 41 radii
between 0.6 and 1.4 times the original radius, along 64 directions. A direction is
masked throughout its radial window if any part leaves the box or belongs to a
closer catalogue centre. Holding angular membership fixed prevents changing sample
coverage from creating a false radial transition.

The proposal estimator finds the strongest outward drop in the smoothed radial
profile. It rejects inadequate angular coverage, insufficient contrast, competing
transitions, search-bound contact and invalid probabilities. Rejected Gaussian
proposals retain the original radius; rejected Fourier proposals retain C. If every
Fourier proposal is rejected, D's initial geometry exactly reproduces C. Proposal
quality criteria are empirical filters, not confidence intervals for a material
interface.

Neural diagnostics preserve the history and selected checkpoint for every frequency
candidate, the chosen state before refitting, and the primary refit. Development
smoke cases additionally retain periodic checkpoints and a refit with matched data
passes. That extra refit is diagnostic; it cannot replace the primary procedure
after seeing held-back scores. Nonfinite primary training runs remain failed runs.

## Corrected physical model and numerical contract

The model retains spherical precipitates, a surface-average Gibbs–Thomson condition,
a screened exterior field, six interior profile knots and fixed interface width
0.3. Its version is `screened_spheres_constant_interior_v1`.

The new model continues each source constantly inside its own sphere, using the
larger of the actual source distance and its radius in that source's kernel. This
removes the interior singularity from the blended observation probability while
leaving the exterior matrix field unchanged. Consequently **B versus A changes both
the observation model and the optimizer**. C versus B measures generic boundary
refinement; D versus C isolates the neural initialization under a common corrected
model. A separate matched-model control would be needed to isolate the optimizer
alone.

Before every coupling solve, the fitter checks radius bounds, box containment and
pairwise surface separation with a numerical margin of 0.00001. This margin is
distinct from the generator's one-unit gap. It checks raw blended probabilities
before evaluating logarithms, using fitting positions, fixed domain probes and
surface probes. Invalid probabilities cannot acquire a finite clipped likelihood.
Final diagnostics also inspect held-back positions without their labels. These
checks establish sampled-domain validity, not validity at every possible point.

Version 1 uses SciPy `trust-constr`, supplied geometry Jacobians, automatic derivatives
of the likelihood and probability constraints, and feasible optimizer bounds.
Backend trial points still pass through the explicit domain guard. A prohibited
trial terminates that attempt with a structured failure. The current preflight
shows why the guard must be tested independently of the optimizer's stated
feasibility settings. Versions 2 and 3 therefore use `feasible_lbfgs`, which projects proposed
steps onto linearized constraints and checks the actual domain during backtracking.
Version 3 also uses the remaining declared rounds after safe iteration caps. Every
version preserves all earlier attempts and their exact source snapshots.

Corrected fits use CPU float64, three Torch threads, three fixed starts, at most 150
iterations per optimization round, three rounds, and 600 objective evaluations per
start. The starts use the default radius-derived values, then `(ell, xi)` values
`(1, 3)` and `(6, 12)`. C and D use identical budgets. The current neural device is
CPU float32 and is frozen before cases; the design does not silently change device
after a run begins.

Feasibility, sampled probability validity, numerical convergence, stable atom
assignments, search-bound contact and ambiguity have separate fields. Starts compete
only when feasible, converged and assignment-stable, using fitting likelihood alone.
An iteration limit is not convergence. Historical A retains its own optimizer and
cannot acquire a convergence certificate merely by producing finite parameters.

## Measurements and promotion rules

Candidate-to-truth associations are fixed from the original catalogue before fitting
and used only for scoring. Primary radius error remains attached to those candidates
even if fitted centres move; rematching cannot make a difficult particle disappear
from the denominator. Matching counts, initial precision and recall, original
geometry, proposed radii, fallback reasons and preparation adjustments remain in
the artifacts.

The primary paired scores are mean absolute relative radius error and absolute
relative capillary-length error. Screening-length error, full held-back log loss,
full removed-atom log loss, bound contact and ambiguity are recorded separately.
The current `matrix_heldback_log_loss` is the **full blended prediction restricted
to true-matrix held-back atoms**. It is not E20's clipped matrix-extension diagnostic,
which can evaluate matrix extension inside an estimated sphere. These definitions
must not be treated as interchangeable.

The prescribed study contains 12 development cases, 24 confirmation cases and 24
conditionally released test cases. The first four development cases are a numerical
smoke group; the other eight remain development. Full execution entails 240 base
arm fits before multistart, diagnostic refits or future profile calculations.

Confirmation requires every one of its 96 case-and-arm records, including errors or
explicitly blocked outcomes. Duplicate, unexpected or malformed identities make the
report incomplete. Only all of the following can promote D:

1. Every C and D fit is geometrically feasible, has valid sampled probabilities,
   converges numerically, and satisfies the objective-evaluation contract.
2. D lowers mean absolute relative radius error in at least 18 of 24 cases, with a
   median paired reduction of at least 0.01.
3. D lowers absolute relative capillary error in at least 18 of 24 cases, with a
   median paired reduction of at least 0.02.
4. D has no higher frequency of search-bound or ambiguity flags than C.
5. The median paired change in full held-back log loss, D minus C, is nonpositive.
   Screening-length errors and matrix loss must also be present as secondary scores.

These are prespecified engineering utility criteria. They are neither significance
tests nor confidence statements about physical constants. A geometry-only success
does not release the reserved test. Failed confirmation retains B; C-versus-B results
remain descriptive. Development never promotes a method.

## Execution order and stopping rules

Run commands from the repository root with the project environment active. Default
output and artifact directories for the current version 3 design are
`experiments/reconstruction/results/e21_geometry_v3` and
`experiments/reconstruction/results/e21_artifacts/v3`. The failed version 1 and 2
directories remain archived; these commands do not resume them under changed code.

For a bounded diagnostic, run one designated development case:

```bash
python -m experiments.reconstruction.e21_geometry --preflight --development-cases 0
```

This command cannot authorize the neural comparison. Only if the bounded diagnostic
is numerically ready should the four-case prerequisite be completed:


```bash
python -m experiments.reconstruction.e21_geometry --preflight
```

All four must satisfy the numerical readiness conditions before the runner permits
neural comparison. The B records are then reused exactly in the four-arm smoke run:

```bash
python -m experiments.reconstruction.e21_geometry
```

The smoke gate requires all 16 arm records, all 12 corrected-arm fits to be valid and
converged, and positive measured runtimes for every arm. Only then may development
expand:

```bash
python -m experiments.reconstruction.e21_geometry --all-development
```

Confirmation requires complete results for all 12 development cases and numerical
readiness for every corrected fit. Freezing changes the design's `status` from
`development` to `frozen`; settings, executable sources, runtime and other inputs
must match the completed development record. A method change needs a new development
version rather than silently reusing the old readiness decision.

```bash
python -m experiments.reconstruction.e21_geometry --phase confirmation
```

Only the complete, recomputed physical-inference gate permits:

```bash
python -m experiments.reconstruction.e21_geometry --phase test
```

Test authorization checks underlying records and provenance rather than trusting a
saved `passed` flag. Numerical failures stop scaling. Unrun prescribed identities
receive explicit prerequisite-failure records, so failure cannot improve a result
by reducing its denominator. Test data are not generated or opened before successful
confirmation. Versions 1 and 2 failed the first decision point; version 3 also
failed its bounded check. A future numerical revision needs a new development
version and complete readiness evidence before the comparison advances.

## Evidence and resume behavior

Reports record the full design, exact environment versions, requested and actual
thread counts, complete relevant source hashes, input metadata and every case's
seed streams. Data identity covers arrays, physical parameters, field constants,
generator settings and observation settings. Source snapshots are archived by
provenance fingerprint. Source, design, runtime or input changes refuse a resume.

Each case-and-arm attempt has a unique directory. Attempt 1 is always canonical,
including a failed or interrupted attempt. Later attempts cannot replace it or rescue
promotion. Atomic report updates may append attempts but cannot alter recorded rows.
If a process stops after an attempt directory is reserved, resume adopts its completed
result or records an explicit interruption; it does not fit that canonical attempt
again. Each case directory is also bound to its exact execution provenance before
any cached data or neural artifact is read. Fit rows carry the same execution
fingerprint; missing or changed provenance prevents orphan adoption. Reusing an old
artifact path with a new report cannot relabel an old result as new-code evidence.
Confirmation has separate case identities, so freezing its design does not rewrite
development bindings. Neural checkpoints are additionally checked against hashes,
the exact fitting data, and the selection/refit design.

Regression tests cover protocol completeness and tampering, raw probability guards,
geometry preservation, all-fallback equality, fixed truth associations, independent
simulation streams, cache identity, held-back-label isolation and reserved-test
access. Their numerical and scientific scope remains distinct from the failed
versions 1 and 2 readiness results and the failed bounded version 3 check.

## Final software verification

The complete suite passed **558 tests in 96.31 seconds** after the version 3 changes.
The [verification record](../../experiments/reconstruction/results/e21_geometry_v3/verification.json)
also records source-archive and documentation-link checks. The live executable
source matches the version 3 archive. The saved data directories contain only four,
four and one development cases for versions 1, 2 and 3 respectively, with no
confirmation or test realizations generated.

These software checks coexist with the failed large-case numerical prerequisite.
They do not establish that every fitting problem converges or that inferred physical
parameters are identifiable or calibrated.

## Work after numerical readiness

After a usable fitter and controlled comparison, the outline calls for a separate
capillary-profile implementation. Exact zero capillarity is representable in the
new fitter, but a supported fixed-zero parameter alone is not a likelihood-ratio
test or an uncertainty interval. Profiles must refit nuisance geometry and chemistry,
handle failed points, expose open or disconnected support, and verify nesting of
null and alternative fits.

The subsequent E22 study must calibrate intervals and null decisions on independent
simulations, preserving the generator's interior dependence. It must distinguish
conditional catalogue calculations from the complete detection-and-inference
procedure, and separate genuine zero-capillarity populations from wrong-field
controls. Measurement degradation, omitted-particle oracle controls, separate
matrix-extension diagnostics, and real APT validation remain later stages.

Neither numerical convergence nor a favorable E21 comparison establishes physical
identifiability, sampling coverage, a universal precision limit or interfacial free
energy in a material. Those claims require the later uncertainty study, thermodynamic
calibration and evidence about the measurement process.
