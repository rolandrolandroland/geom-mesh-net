# Implementation plan for reliable physics inference

29 October 2026. Proposed implementation and experiment sequence following the
read-only project audit. The phases below retain the implementation design; the
checkpoint note records the narrow repairs completed before the larger study.

**Checkpoint update, 29 October 2026:** commit `0c08355` preserves the original
E20 source, design and reports. Merge preparation adds the Phase 1 case-accounting
and growth-penalty repairs, their regression tests, and corrections to E18's
scientific interpretation. Historical results and their source hashes remain
unchanged. The constrained fitter, new boundary comparison and uncertainty study
remain future work. See [E20's reproduction note](../experiments/E20_field_to_particles.md#reproduction-and-saved-evidence)
before attempting to resume the archived experiment from the changed source.

## Objective and first deliverable

Build a procedure that estimates precipitate geometry and curvature-dependent
chemistry, reports uncertainty, and recognizes when the observations cannot resolve
the physical parameters. The neural field supplies candidate boundary information;
the diffusion model supplies a constrained physical interpretation.

The first deliverable is a controlled comparison of Gaussian and Fourier boundary
estimates on the **same detected particles**, followed by the same physically
feasible fitter. The second is an uncertainty-validation study. A useful outcome
may retain the Gaussian method if neural refinement adds no demonstrated benefit.

The research question is:

> How do missed particles, geometry error, and spatial resolution limit recovery of
> curvature-dependent chemistry, and can those limits be recognized from the data?

The existing [E20 experiment](../experiments/E20_field_to_particles.md) supplies the
starting evidence: Fourier improved concentration prediction but missed 35 of 300
particles, versus 8 for the Gaussian detector, and failed promotion. Both physics
initializations produced an invalid fit on the final confirmation pattern. These
results remain part of the record.

The October audit adds two implementation defects to repair: duplicate or unexpected
cases can distort the older Stage 5 gate, and the particle-growth penalty counts one
parameter per particle even when its three centre coordinates are fitted. It also
found that the 35 missed particles contain only about 0.29% of simulated atoms,
explaining how their loss can be obscured by an average prediction score. This is
evidence for a mismatch between the scoring objective and particle detection, not
proof of a particular training failure.

## Sequence and decision points

| Phase | Deliverable | Requirement before proceeding |
| --- | --- | --- |
| 0 | Preserved E20 evidence and a new experiment namespace | Existing results remain reproducible and cannot be overwritten |
| 1 | Correct experiment accounting and parameter counting | Targeted regression tests and the existing suite pass |
| 2 | Fitter that maintains its geometry assumptions | Instrumented objective evaluations stay feasible; failures are explicit |
| 3 | Saved training diagnostics and candidate boundary proposals | Candidate identities survive every transformation; fallback is deterministic |
| 4 | E21 controlled geometry comparison | Complete frozen confirmation determines whether neural refinement advances |
| 5 | Capillary profiles and an explicit zero-capillarity model | Known ambiguities and null cases behave correctly in numerical tests |
| 6 | E22 uncertainty and measurement-limit validation | Independent repetitions establish the scope of coverage and false-positive claims |
| 7 | Reproducible research report and decision map | Every claimed regime has complete evidence and stated limitations |

Phases 1 and 3 can proceed independently after preservation. Phase 4 needs phases
1–3. Phase 5 can start once the corrected fitter is stable. Phase 6 uses whichever
geometry method the controlled comparison supports; it does not require a neural
victory. A failed comparison is documented and closes that version of the experiment.

## Phase 0 Preserve the current evidence

### Record an immutable baseline

1. Preserve the current source, including uncommitted and untracked E20 files, before
   editing shared modules. A commit identifier alone does not identify this working
   tree. Store a complete source manifest and an archive or version-control snapshot.
2. Preserve the E20 design, calibration and confirmation reports, failed-device-run
   archive, checkpoint hashes, dataset/oracle identities, and environment versions.
3. Do not alter E20's stored fingerprints to make its reports appear compatible with
   changed code. Historical reproduction uses the preserved source and environment.
4. Create separate proposed namespaces for E21 boundary refinement and E22 uncertainty.
   Use new report and checkpoint directories. Never resume one design into another.
5. Keep diffusion patterns 74–81 reserved under E20's existing failed promotion rule.
   The new studies will obtain fresh evaluation data from new simulation streams.

### Extend provenance

Hash the simulator, oracle loader, thinning and splitting code, detector, field
trainer, boundary estimator, fitter, scoring code, and complete design. Record
Python, NumPy, SciPy, Torch, device, precision, thread counts and seed streams.
Include the exact parameter metadata in dataset identity; coordinate and label
hashes alone do not identify the physical truth used for scoring.

Each case needs its phase, simulation identity, observation identity, method,
attempt number and status. Retries retain all attempts. An explicit canonical-attempt
rule must be independent of truth or held-back scores. Reports are written atomically.

**Exit criterion:** a small reproduction audit verifies E20's stored decisions, and
the new runner refuses an altered source, design, dataset or split identity.

## Phase 1 Repair scientific accounting

### Require exactly the expected cases

Update the older gate in
[stage5_physics_fit.py](../../experiments/reconstruction/stage5_physics_fit.py) using
the stricter pattern in
[stage4_particles.py](../../experiments/reconstruction/stage4_particles.py).

Compute the expected case keys before reading scores. Reject missing, duplicate,
unexpected and malformed keys. Require every prescribed field control and method.
Do not remove error records before completeness checks or hide failed cases by
reducing a denominator. Distinguish an unfinished report from a completed experiment
whose method failed.

Regression cases must include the previously reproduced duplicate-case pass, a
favourable row given an unexpected identity, an omitted alternative-field control,
an error replacing an expected case, and a manually edited saved `passed` flag.
Evaluation authorization must recompute the decision from the underlying records.

### Count active added parameters

Correct `grow_geometry` in
[joint_fit.py](../../geom_mesh_net/fields/joint_fit.py) to compare the improvement
in summed negative log likelihood with:

`0.5 * log(number_of_fitting_atoms) * number_of_added_parameters`.

One added sphere contributes one radius when its centre is fixed, or four parameters
when its three coordinates are free. Shared chemistry, width and interior-profile
parameters cancel if their dimensions are unchanged. Count parameters that influence
the active model, including future per-particle additions; unused stored tensors do
not count. Test one and several births, fixed and free centres, a custom penalty,
and equality at the strict acceptance threshold.

Keep growth disabled in E21. Its data-driven candidate search and parameter
boundaries make this information-criterion rule a selection heuristic, not a
calibrated test for capillarity.

### Narrow earlier accuracy claims

When updating E18 documentation, describe its error-to-Fisher-bound ratios as
development diagnostics. Use a reference with the same nuisance parameters as the
fitted model, including centres when they are free. Explain that one absolute error
per pattern does not establish sampling variance, efficiency or interval coverage.
Keep the generating-model and interface-width assumptions beside every bound.
Preserve the original numerical results and record corrections explicitly.

**Exit criterion:** targeted regressions pass, the original complete Stage 5 report
retains its genuine failed verdict, and the full suite passes. A changed test count
alone is not evidence of physical validity.

## Phase 2 Maintain valid geometry during fitting

### Introduce a separately versioned fitter

Add `geom_mesh_net/fields/constrained_joint_fit.py` rather than silently changing the
meaning of archived E18 and E20 fits. Reuse the field equations where appropriate,
but expose a parameter vector and separate geometry validation, raw prediction,
likelihood, derivatives and optimizer reporting.

Keep the first model deliberately close to E20: spherical precipitates, the same
surface-average Gibbs–Thomson condition, six interior-profile segments, fixed
interface width 0.3 in the simulator's length units, CPU float64, and an explicitly
bounded optimization budget. The interface width remains a modelling assumption;
freeing it belongs to a later sensitivity study.

Use scaled physical coordinates and explicit optimizer bounds for geometry, smooth
log coordinates for strictly positive physical quantities, and a direct nonnegative
capillary parameter that includes zero. Do not inherit `exp(clamp(log_parameter))`:
it can produce zero gradients outside its bounds and cannot represent the exact
capillarity null. Fixing capillary length to zero must remove it from the active
optimization vector.

### Enforce the geometric constraints

For every particle pair, require:

`distance(centre_i, centre_j) >= radius_i + radius_j + numerical_margin`.

Use a small documented numerical margin, separate from E20's 0.5 initialization
gap. Do not impose a large physical gap without testing its bias. Retain explicit
radius bounds. For the initial synthetic study, require complete spheres within the
known box; boundary-truncated particles require a different observation model.

Start from a feasible geometry. With fixed candidate centres, preparing radii can be
a constrained projection minimizing their change subject to pairwise separation,
radius and box bounds. Preserve candidate IDs and report all changes. If no feasible
geometry exists without deleting a candidate or crossing the radius floor, record a
case failure. Do not call an overlap resolver that can silently merge particles.

Prototype SciPy `trust-constr` with feasible inequality constraints, supplied
Jacobians and a quasi-Newton Hessian approximation. The installed baseline is SciPy
1.15.3; its [constraint interface](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.NonlinearConstraint.html)
supports `keep_feasible`, and its
[solver report](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-trustconstr.html)
exposes optimality and constraint violation. Treat this as a backend to verify, not
a guarantee supplied by its name.

Instrument every objective evaluation, including trial points. Validate separation
before the coupling solve. Supply derivatives so numerical differencing does not
probe invalid geometry. An invalid evaluation must never become an accepted
likelihood value. If the backend cannot meet this contract, stop the prototype and
use an explicitly feasibility-preserving step/line-search implementation. A finite
overlap penalty alone is not sufficient.

### Separate raw physical checks from numerical clipping

Provide raw matrix and full-prediction interfaces. Check finite amplitudes, solve
residuals and conditioning, concentrations outside fitted spheres, and the blended
probabilities actually entering the likelihood. Use fixed fitting-domain probes and
surface quadrature selected without held-back labels; final diagnostics also inspect
held-back positions. Finite sampling establishes validity only on the checked domain.

Check raw likelihood probabilities before logarithms or clipping. Material domain
violations require an enforced constraint or a structured failed attempt; they
cannot receive a finite clipped likelihood. Small numerical safeguards for logarithms
must not turn materially negative or greater-than-one predictions into valid
observations. The matrix extension inside
a precipitate is not itself a physical concentration; if its use in the current
blend causes invalid total predictions, make the extension/blend change explicit as
a new model version and validate it before comparison. Keep that change common to
every corrected-fitter arm.

### Record distinct outcomes

Return separate fields for geometric feasibility, sampled probability validity,
numerical convergence, search-bound contact, and identifiability. A small loss or a
finite parameter vector is insufficient for convergence. Record iterations,
evaluations, constraint violation, stationarity measure, objective history,
assignment changes and termination reason.

Use the same deterministic set of physical starting values and the same evaluation
budget for every geometry arm. Select among feasible, converged solutions using
fitting likelihood only; retain every attempt. Record materially different solutions
with similar likelihood as ambiguity. A boundary estimate may be feasible yet have
truncated parameter support; do not confuse that with an identified estimate.

The nearest-particle assignment currently changes between optimization rounds.
Recompute assignments, evaluate the full objective again, and require a stable final
assignment/objective criterion. Reaching a round or iteration cap is a separate
outcome, not automatic convergence.

### Test the fitter before scaling

Test isolated and nearly touching spheres, attempted overlaps, impossible starting
geometries, box boundaries, extreme concentrations, ill-conditioned coupling, and
search-bound contact. Verify derivatives against finite differences at safe interior
points. Record every trial's minimum gap in an adversarial optimization test.
Check agreement with the old field on well-separated, valid parameter settings.
Exercise the previously invalid E20 geometry as a regression fixture, without
changing its archived result.

**Exit criterion:** no accepted fit or evaluated physical likelihood violates the
declared domain; numerical failures are distinguishable from unresolved parameters;
small simulations with known solutions behave as expected. Profile and coverage
claims remain deferred.

## Phase 3 Diagnose training and propose boundaries

### Preserve evidence before changing the network

Extend training artifact collection to save every frequency candidate's selected
checkpoint, its history, the selected checkpoint before refitting, and the final
refit. Save periodic diagnostic checkpoints on a small development subset. Keep
fitting observations, held-back observations and truth-scoring code separate.

Compare the existing same-update refit with a matched-pass refit on development
data. The current restart sees more fitting atoms for the same update count, so
roughly 20% fewer passes is a plausible contributor, not an established defect.
Use a fixed extractor when comparing checkpoints.

For scoring only, record signal and prediction error inside true particles by size,
contrast and observed count, as well as matrix and whole-pattern scores. Evaluate
the model function directly when checking whether a grid lost a feature. Determine
whether small-particle signal was never learned, appeared after the selected
checkpoint, or disappeared during refitting.

An oracle-field extraction control can test the extractor, and a noiseless-target
training control can probe representation/optimization limits. Both are explicitly
diagnostic. Neither enters the deployable comparison. Do not change architecture,
loss weighting and detector rules simultaneously. Any later weighted objective needs
fresh probability-calibration checks.

### Build one candidate catalogue

Run the existing Gaussian significance detector once on fitting atoms. Assign stable
candidate IDs, centres and original radii. Freeze this catalogue before branching
into comparison arms. Initial candidate centres stay fixed during boundary proposal;
centres may move later under the common constrained physics fitter.

Add `geom_mesh_net/fields/boundary_refinement.py` with these proposed interfaces:

| Interface | Responsibility |
| --- | --- |
| `CandidateCatalog` | Stable IDs, centres, baseline radii and observation provenance |
| `sample_candidate_profiles` | Shared radial/angular sampling and observation-derived masking |
| `propose_radii` | One boundary estimator applied to either concentration field |
| `apply_proposals_with_fallback` | Deterministic acceptance/fallback without deleting candidates |
| `prepare_feasible_geometry` | Identity-preserving radius projection with an adjustment record |

Evaluate both fields on identical physical sample locations. Restrict searches to a
development-calibrated neighbourhood of the baseline radius. Mask samples outside
the box or assigned to neighbouring candidates by the same fixed rule. Select a
simple radial-transition estimator during development, such as a fitted contrast
transition, and freeze its settings before confirmation. Its output is a boundary
initialization proposal, not an already validated material interface.

Record angular coverage, concentration contrast, competing transitions, search-bound
contact and numerical validity. These are empirical quality criteria; Gaussian count
variance formulas do not supply uncertainty for neural predictions.

If the Gaussian profile is insufficient, retain the original detector radius. If
the Fourier profile is insufficient, retain the Gaussian-refined radius. Report
fallback reasons and frequency. Fallback everywhere must reproduce the Gaussian
refinement arm exactly. Report all-candidate results first; an eligible-only subset
cannot support the primary improvement claim.

**Exit criterion:** identical input fields give identical proposals, missing signal
never deletes candidates, all-fallback equality holds, and held-back labels or truth
cannot affect proposals, eligibility, initialization or fit selection.

## Phase 4 Run the controlled geometry comparison

### Define the four arms

Use new E21 runner and design files, such as
`experiments/reconstruction/e21_geometry.py` and `e21_design.json`.

| Arm | Boundary initialization | Physics fit | Contrast answered |
| --- | --- | --- | --- |
| A | Original Gaussian radii | Existing fitter | Historical implementation control |
| B | Original Gaussian radii | Corrected constrained fitter | B versus A isolates the fitter correction |
| C | Gaussian-field refinement with original-radius fallback | Corrected constrained fitter | C versus B isolates generic boundary refinement |
| D | Fourier-field proposals with C fallback | Corrected constrained fitter | D versus C isolates neural initialization |

Every arm starts from the same candidate identities and observation split. Apply the
same identity-preserving feasibility preparation and record its adjustments. For
A versus B, use exactly the same prepared initial geometry. C and D use the same
proposal estimator and feasibility procedure, with only the supplying field changed.
Disable particle birth and deletion in all four arms.

Apply the same final diagnostics to A and retain its invalid fits as control
outcomes. The evaluation-time feasibility contract applies to the new fitter;
the historical control is not assumed to satisfy it. If correcting the raw blend
also changes the observation model, label B versus A a combined model-and-fitter
change. A separate matched-model optimizer control would be needed to isolate the
optimizer contribution in that situation.

Use identical physical likelihoods, starting-constant rules, diagnostic points and
budgets within the corrected-fitter comparison. The network supplies an
initialization, not an additional independent likelihood derived from the same
atoms. If both initializations converge to the same optimum, report that result.
An advantage under a finite budget is an optimization advantage; it does not by
itself establish additional physical information in the observations.

### Allocate fresh data prospectively

Proposed initial scope is one efficiency, 0.37, with 20% of observed atoms held back
before label-dependent fitting, matching E20's effective fitting fraction of about
29.6% of original atoms. Use 12 new development geometries, 24 new confirmation
geometries and 24 newly reserved test geometries. The 12 development cases may be
staged as four smoke cases followed by eight additional cases; all twelve remain
development data.

Use a new dataset directory and independently seeded streams for parameters,
geometry/positions, labels, thinning, observation splits, network initialization and
resampling. Write their exact identifiers into a manifest before scoring. New labels
on an old geometry are conditional repetitions, not new geometry tests. Do not
overwrite the old generator's global entropy values or relabel existing patterns as
fresh. Freeze method settings after development and before confirmation outcomes.

The initial four-arm scope entails 240 base physical fits and up to 60 neural field
preparations, before multistart or profile work. Release confirmation and test jobs
only at their respective decision points. Reuse exact compatible field artifacts;
never reuse scores from a different fitting split.

### Measure the right quantities

Initial centre-based precision and recall must be identical across arms because the
candidate centres are identical. They are invariants and diagnostics, not an
improvement gate. Do not reuse E20's F1-win requirement.

Primary geometry measurements are paired radius error, centre displacement after
fitting, overlap/feasibility outcomes and fallback frequency. Fix candidate-to-truth
associations from the original catalogue for the primary paired geometry analysis;
report later rematching separately so a difficult particle cannot vanish from the
error denominator. Keep unmatched candidates and missed true particles visible.

Physical measurements are absolute and signed errors in capillary length and
screening length, parameter-bound contact, convergence and ambiguity, plus full-field
held-back and removed-atom log loss. Keep matrix-extension loss as a secondary
diagnostic, with its definition explicit. Report per-case values, pattern-level
paired summaries and uncertainty across patterns. Atoms within one pattern are not
independent experimental repetitions.

### Predeclare advancement rules

The following are proposed engineering utility thresholds to encode before
confirmation, not significance tests or universal material tolerances:

- Require exactly every planned case and arm, with failure records retained.
- Require all 24 C and D fits to be feasible and numerically converged for numerical
  improvement claims. If a case fails, finish the prescribed report but do not
  silently make a 23-case paired analysis the primary result.
- Call neural boundary refinement useful only if D lowers mean absolute relative
  radius error in at least 18 of 24 cases and its median paired reduction is at least
  0.01, or one percentage point, on the fixed matched candidates.
- Call it useful for physical inference only if D also lowers absolute relative
  capillary error in at least 18 of 24 cases, with a median paired reduction of at
  least 0.02. Require no higher rate of search-bound/ambiguity flags and no positive
  median paired change in full-field held-back loss. Report screening-length errors
  and matrix loss as mandatory secondary outcomes.
- Keep failure, fallback, geometry and parameter results separate. A geometry-only
  improvement is not a physical-inference success.

These thresholds may be revised during development for a documented practical
reason, before confirmation is inspected. A change afterwards creates a new design
and needs new confirmation evidence. Only passage of every physical-inference
criterion above releases the 24 new test cases for one frozen evaluation; the final
report includes all cases and the same criteria. Geometry-only success does not
release that test. Failed confirmation leaves it unopened and retains B as the
working baseline. The C-versus-B results remain descriptive unless a separate,
prespecified selection rule is evaluated on fresh evidence. Uncertainty work can
continue with B regardless of whether neural refinement advances.

### Isolate missing-particle effects separately

On development simulations, compare fixed true geometry containing all particles
with exactly the same geometry after removing specified particles. Include both
the actual Gaussian-missed subset and controlled omissions by particle size. Keep
the remaining centres/radii fixed while estimating constants. This isolates omission
effects from errors in retained geometry. Label the comparison oracle-assisted and
keep it outside deployment and promotion scores.

## Phase 5 Build parameter profiles and a capillarity null

### Profile the capillary length

Add `geom_mesh_net/inference/capillarity_profile.py`. For each proposed capillary
length, hold that value fixed and refit the nuisance quantities: equilibrium and
background concentration, screening length, radii, centres and interior-profile
parameters. Initially keep interface width fixed at the study's declared value;
an unknown width requires additional profiling and validation.

Use a coarse parameter grid, refine around likelihood crossings, and expand the
search domain by a predeclared rule. Run warm starts from both directions and a
fixed set of independent starts. Record every constrained fit and do not interpolate
across failed points. A profile point that beats the alleged global optimum triggers
a baseline refit and a consistency check. Retain disconnected confidence sets,
open ends and bound truncation rather than forcing a narrow symmetric interval.

Use summed negative log likelihood for likelihood-ratio statistics. Keep any
optimizer barrier or feasibility penalty out of the reported statistical likelihood.
The profile itself measures compatibility within a model; calling its set a
confidence interval requires the validation in phase 6.

### Represent zero capillarity exactly

Implement an explicit model with capillary length fixed to zero. Setting a positive
log-parameter to its lower bound is not the same null. Keep the same geometry and
nuisance treatment in the null and alternative. A deliberate null constraint is not
an accidental search-bound failure.

The alternative must contain that exact null as a feasible special case. Its best
negative log likelihood cannot exceed the null's beyond numerical tolerance. A
violation triggers the prescribed refit/consistency procedure and, if unresolved,
an optimization failure. Do not replace a negative likelihood-ratio statistic with
zero and continue as if optimization had succeeded.

The existing diffusion generator ties supersaturation to capillary length. Simply
setting its capillary length to zero also forces equilibrium and far-field
concentrations together, producing a flat matrix. Add a generator path where
equilibrium concentration, background concentration and screening length can be
specified independently: **no curvature dependence may still coexist with a
nonuniform diffusion field**. Reject physically invalid parameter combinations and
record generator failures; do not clip them into an apparently valid benchmark.

Use simulation-calibrated likelihood-ratio thresholds. The zero boundary, weakly
identified nuisance parameters, fitted geometry and candidate selection prevent
assuming an ordinary chi-squared threshold is valid. Alternative-field rejection
is a separate model-adequacy question; rejection of zero capillarity does not prove
the screened-diffusion law is correct.

Develop a separate adequacy check using full-field held-back prediction against
the flexible field and constant baselines, plus residuals stratified by estimated
surface distance and particle size. Freeze the statistic and calibrate thresholds
on independent simulations from the valid model. Evaluate smoothed-noise and
shuffled-surface controls separately. Where a generating process has no meaningful
capillary parameter, measure erroneous endorsement of the capillary model rather
than parameter coverage. These checks establish sensitivity only to tested failures.

### Test known ambiguities

Test equal-radius populations where changing equilibrium concentration can exactly
offset a change in capillary length, narrow versus broad radius spreads, weak
contrast, low observations, and intentionally truncated search ranges. Equal-radius
cases must not return a sharply identified capillary estimate solely because an
optimizer terminated. Include tests with finite but invalid profile branches and
different starting points finding competing solutions.

**Exit criterion:** numerical profile consistency, explicit null handling and known
non-identifiability tests pass. No coverage claim follows from these checks alone.

## Phase 6 Validate uncertainty and measurement limits

### Separate three validation questions

| Validation layer | Purpose | Scope of a successful result |
| --- | --- | --- |
| Model-matched simulation | Verify numerical inference under the fitted data model | Calibration within that model |
| Existing simulator family with independent new draws | Test transfer across sharp interfaces and its label-generation dependence | Empirical performance for that simulator |
| Measurement and physical-model perturbations | Test blur, omissions and wrong physical laws | Robustness only over the declared perturbations |

For the existing simulator family, preserve its fixed-count/interior label
dependence. An independent-Bernoulli parametric bootstrap is a useful model-matched
control but cannot alone validate uncertainty for that generator or real APT.

Distinguish intervals conditional on an already selected candidate catalogue from
end-to-end intervals. End-to-end repetitions rerun observation, detection,
refinement, fit selection, and interval construction; merely perturbing labels with
the candidate list frozen does not propagate detection uncertainty. If candidate
selection remains unmodelled, describe the construction as fixed-catalogue and
model-conditional, and quantify its coverage in the full pipeline. Holding one
catalogue fixed does not automatically produce a statistically valid selective
confidence interval after that catalogue was chosen from the same labels.
End-to-end empirical calibration can support a procedure-level claim within its
tested simulation distribution without establishing validity for every individual
selected catalogue.

### Use a bounded progression of simulations

1. Smoke-test profile construction on a few known development geometries.
2. Run 25 complete repetitions per proposed condition to expose failures and measure
   runtime. These repetitions are development data, not evidence of calibrated coverage.
3. Freeze interval and null-test calibration using separate simulation streams.
4. Evaluate at least 200 independent complete datasets per condition for which a
   coverage claim is sought, and at least 200 per declared null family. First target
   one favourable regime and one difficult regime at efficiency 0.37.
5. Extend to additional efficiencies or blur conditions only after a separate budget
   and complete case manifest are frozen. A single validated regime is a legitimate
   first deliverable; pooling unlike regimes cannot conceal a failing subgroup.

Two hundred repetitions give a binomial standard error near 1.5 percentage points
for 95% coverage and about 1.5 points for a 5% false-positive rate. Report binomial
intervals, not only percentages. This precision is a design target, not a guarantee
that the method will meet it.

### Define coverage and abstention explicitly

Use nominal 95% confidence sets and a nominal 5% capillarity test as initial targets.
Prespecify acceptable interval width for a scientifically useful estimate. Report
bias, RMSE, set width, coverage, rejection, inconclusive outcomes and technical
failures separately for every condition.

For the initial simulator study, a proposed utility cutoff is a confidence-set
span of at most 1.0 in its length units, without an artificial search-bound
truncation. This is a practical target for the study's parameter range, not a
material-independent tolerance. Freeze it before evaluation and report the full
width distribution regardless of the classification.

Every planned dataset contributes to the denominator. An absent or invalid interval
is a failure in the unconditional coverage calculation. Also report coverage among
completed intervals, the completion rate, and the fraction judged scientifically
resolved. A broad valid set is informative uncertainty, not a technical error.
An inconclusive fit cannot be counted as a correct rejection of a wrong model.

Proposed initial advancement criteria are: at least 95% technically valid interval
outputs; a one-sided 95% lower confidence bound of at least 90% on unconditional
coverage; and a one-sided 95% upper confidence bound no greater than 10% on the
conservative false-positive frequency, counting technical failures as false
positives. Also report the observed false-claim frequency separately. These are explicit tolerances around nominal
95%/5% targets, not evidence of exact calibration. Report statistical power at
prespecified nonzero capillary lengths and informative-interval frequency so that
always returning an unbounded set cannot appear to solve the inference problem.

Simulation-calibration and evaluation sets remain separate. If uncertainty is valid
only for well-detected, well-resolved cases, the eligibility rule must be computable
from observations and evaluated on all cases before filtering. A rule that requires
knowing which true particles were missed is a diagnostic, not a deployable decision.

### Add perturbations one at a time

Start with radius spread and particle omission because the audit already identifies
their relevance. Next vary detection efficiency and positional blur. Describe blur
through a forward observation model: generate latent atoms and labels, then distort
positions and detect atoms. A logistic interface width is not automatically a
validated model of that process. Keep isotropic blur, anisotropic blur and local
magnification distinct. Add alternate physical fields as separate model-adequacy
controls, distinct from genuine zero-capillarity null populations.

Claims that the procedure recognizes failure of the screened-diffusion model need
an additional frozen evaluation with both smoothed-noise and shuffled-surface
controls, each with at least 200 independent complete datasets. Prespecify the
adequacy decision and report false endorsement separately for each control family;
do not pool them with the capillarity null tests. For such a claim, require a
one-sided 95% upper bound of at most 10% on false endorsement, conservatively
counting technical failures as endorsements. Also report acceptance on valid-model
datasets so that rejecting every model cannot appear useful. These are separately
budgeted conditions, adding at least 400 full procedures; they are not prerequisites
for the narrower initial claim of interval coverage under a correctly specified model.

Use full controlled repeats at fixed conditions to estimate conditional coverage,
and independent geometry/parameter draws to assess population performance. Separate
geometry variation from repeated labels on one geometry. Do not use a single
resampling scheme to claim both results.

**Exit criterion:** produce a map of the regimes with reliable, unresolved or
invalid inference, supported by complete repeated-run results. Failure to calibrate
is itself a result and blocks application claims for that regime.

## Implementation layout and review batches

The following paths are proposed additions, not existing APIs:

| Batch | Main additions or changes | Review focus |
| --- | --- | --- |
| 1 | Gate corrections in `stage5_physics_fit.py`; active parameter counting in `joint_fit.py`; dedicated regression tests | Exactly-once scope and correct complexity accounting |
| 2 | `fields/constrained_joint_fit.py`; structured fit diagnostics; `tests/test_constrained_joint_fit.py` | Feasible evaluations, raw probability validity and convergence |
| 3 | Training checkpoint/history export; `fields/boundary_refinement.py`; boundary/refit tests | Candidate identity, fallback and fitting-data isolation |
| 4 | `experiments/reconstruction/e21_geometry.py`, `e21_design.json`, generator adapter and report/plot helpers | Four arms, fresh splits, frozen selection and complete decisions |
| 5 | `inference/capillarity_profile.py`; explicit null model; profile tests | Nuisance reoptimization, ambiguous sets and exact zero |
| 6 | `experiments/reconstruction/e22_uncertainty.py`, `e22_design.json`, full-pipeline simulation runner | Calibration/evaluation separation, coverage and false-positive accounting |
| 7 | E21/E22 walkthroughs and roadmap corrections | Claims match completed evidence |

Add typed result structures where they prevent confusion between numerical failure,
physical-domain failure, weak identification and a successful estimate. Avoid
reusing one `reliable` boolean for all four.

Keep large checkpoints, sampled profiles and repeated simulation arrays in ignored
artifact directories with hashes. Retain compact manifests, per-case summaries,
selection decisions, failure records, gate calculations and figure-generating code.
Small summaries should link to the exact artifacts needed to reproduce a finding.

Run focused tests for each batch, then the full suite at integration boundaries.
Do not expand statistical experiments merely to compensate for an implementation
test failure. Review numerical models separately from experiment accounting.

## Compute controls and stopping rules

Measure the constrained fitter and profile cost on development cases before
launching confirmation. E20 recorded about 35.5 minutes for eight field preparations
and 7.5 minutes for eight bounded physics fits; those are reference measurements for
the old methods, not timing promises for constrained optimization.

Estimate cost as the sum of field preparations, fit starts, profile points and full
pipeline repetitions. Profile work can dominate by orders of magnitude. Record an
evaluation budget per fit, maximum profile points, fixed restart count, artifact
storage estimate and total planned cases before a batch begins. Start with coarse
profiles and a declared adaptive refinement rule. Do not launch a broad Cartesian
grid of all measurement conditions first.

If the cost exceeds the intended study budget, reduce the number of claimed regimes
before evaluation begins or report a pilot. Do not reduce repetitions after seeing
favourable results and retain the same calibration claim. Stop numerical scaling if
feasibility or convergence regressions appear. Stop the neural branch if it fails
the frozen comparison; continue uncertainty validation with the supported baseline.

## Final research artifacts

The completed work should provide:

1. A reusable constrained fitter and explicit diagnostics.
2. A paired geometry comparison that identifies whether neural refinement adds value.
3. Reproducible capillary profiles and an exact zero-capillarity control.
4. Validated intervals or a precise account of where calibration fails.
5. A measurement-condition map showing useful estimates, unresolved parameters and
   model failures, with sample counts and uncertainty on every rate.
6. A report distinguishing numerical correctness, model-conditional inference,
   simulator robustness and eventual real-data evidence.

The scope remains spherical precipitates and a phenomenological screened field with
surface-average interfacial conditions. Conversion to a material's interfacial free
energy needs its thermodynamic model and temperature; a fitted capillary length alone
does not supply that mapping. Real APT application also needs an independently
checked measurement model and external validation.

Capillarity measurements have precedent in
[Sudbrack and colleagues' Ni–Al–Cr study](https://ntrs.nasa.gov/api/citations/20070006544/downloads/20070006544.pdf),
and small-particle size/composition limitations are documented by
[De Geuser and Gault](https://arxiv.org/abs/1910.10532). The intended contribution is
the propagation and recognition of geometry and measurement uncertainty in physical
inference. Establishing novelty requires a focused literature review of that
specific question.

## First implementation milestone

Complete phases 0–2 and the candidate-identity/fallback tests from phase 3, then run
the four-arm comparison on the first four new development cases. Deliver a numerical
validity report and compute estimate before scaling to the remaining development
cases. This milestone proves that the experiment can be conducted fairly; it does
not claim a neural improvement or calibrated physical inference.
