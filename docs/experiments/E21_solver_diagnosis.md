# E21 staged solver diagnosis

2 November 2026.

**Execution status: complete; the full physical fitter is not numerically ready.**
All 22 prescribed attempts finished under frozen source and settings. The four
small-control fits and three observed fixed-geometry chemistry/profile fits
converged. **All 15 observed fits that released geometry failed their numerical
requirements**, including the sequential warm starts. The
[complete diagnostic report](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/diagnosis.json)
retains every success and failure. Both inventory-completion flags are true;
`promotion` and `inference_validated` remain false.

The E21 numerical work and its failed preflights were committed and
pushed at `b54591b` before this diagnostic work began. This document records the
procedure, measured outcomes and remaining numerical work.
The archived E21 versions 1–3 retain their original reports, data, source snapshots
and failed readiness decisions.

The immediate question is why the constrained physical fit fails its numerical
convergence requirement on a large observed particle catalogue. The diagnostic
holds that catalogue fixed and progressively releases parameter blocks. It does
not train a neural field, compare the four E21 arms, or estimate measurement limits.
It cannot grant permission to open confirmation or reserved-test data.

Related evidence: [E21 constrained geometry](E21_constrained_geometry.md) ·
Implementation plan: [reliable physics inference](../guides/physics_inference_implementation_plan.md) ·
Frozen design: [e21_solver_diagnosis_design.json](../../experiments/reconstruction/e21_solver_diagnosis_design.json) ·
Runner: [e21_solver_diagnosis.py](../../experiments/reconstruction/e21_solver_diagnosis.py).

## Measured results and stopping decision

The complete accounting contains **22 of 22 prescribed attempts**. All failures
remain in their original denominators. The diagnostic successfully distinguishes
an easy conditional chemistry problem from the unresolved geometry optimization
problem; it does not provide a validated physical inference procedure.

| Input and stage | Completed / prescribed | Converged / completed | Recorded outcome |
| --- | ---: | ---: | --- |
| Small control, four independent blocks | 4 / 4 | 4 / 4 | All active residuals below `1e-5` |
| Observed chemistry and profile | 3 / 3 | 3 / 3 | Active residuals `4.49e-6`–`9.36e-6`; assignments stable |
| Observed radii, centres fixed | 3 / 3 | 0 / 3 | Three-round budgets exhausted; active residuals `3.60e-5`–`5.47e-5` |
| Observed centres, radii fixed | 3 / 3 | 0 / 3 | Each exhausted 600 evaluations; no accepted convergence certificate |
| Observed full joint fit | 3 / 3 | 0 / 3 | Reproduces archived version 3 endpoints; active residuals `9.81e-5`–`1.86e-4` |
| Observed sequential radius stage | 3 / 3 | 0 / 3 | Each exhausted 600 evaluations despite a converged chemistry parent |
| Observed sequential full joint stage | 3 / 3 | 0 / 3 | Three-round budgets exhausted; active residuals `1.26e-4`–`1.62e-4` |

Across 8,333 fitting evaluations, 909 unsafe trials were rejected before likelihood
evaluation and **zero invalid likelihoods were scored**. The rejection counts
describe guard operation, not convergence. Additional endpoint probes are recorded
separately and do not change these fitting budgets or outcomes.

**Decision:** retain the numerical stop. No observed geometry fit qualifies for
selection. Neural training, the four-arm comparison, confirmation, reserved-test
evaluation and uncertainty claims remain blocked by the existing prerequisite.
Completing this diagnostic inventory does not satisfy E21's four-case preflight.

The small control's prediction RMSE is approximately `0.00017`–`0.00055`, and its
expected mean-NLL excess above the generating entropy is `1.52e-7`–`1.66e-6`.
This is success on a small, favorable exact-model numerical control. It does not
establish noisy-data parameter recovery or performance on a large catalogue.

Holding observed geometry fixed makes chemistry/profile fitting numerically
manageable under the original budget. It does not settle parameter resolution:
the three converged fits have screening lengths approximately 13.30–16.57 while
their summed NLLs differ by only 0.139. These are diagnostic estimates conditional
on one imperfect catalogue, not identified physical measurements or uncertainty
intervals.

The radius-only failures are not merely failures of assignment stabilization:
all three pass that check, and their remaining active residual is in the radius
block. The centre-only failures have different termination contexts. Starts 0 and
1 exhaust their budgets before a fresh actual-assignment endpoint evaluation; their
saved loss is explicitly for the last round's frozen assignments. Those losses
must not be compared as if they were fresh full-objective endpoints. Start 2 has
a fresh active endpoint residual of `2.90e-5`, still above tolerance. Earlier-round
residuals do not substitute for missing endpoint certificates.

All three independent joint starts exactly reproduce their corresponding archived
version 3 centres, radii, physical constants, interior profile, final loss and
terminal status. This is a reproducibility check of the unchanged full-space
trajectory, not three new independent experimental cases. The new active-block
interfaces and state serialization did not silently rescue or alter those fits.

All three sequential radius stages exhaust the evaluation budget. Their terminal
losses retain frozen-round assignment context and lack a fresh active endpoint
certificate. Earlier assignment-stability flags cannot fill that gap. Their valid
accepted states are carried forward as explicitly nonconverged parents for the
prescribed sequential joint diagnostic. All three subsequent joint stages also
exhaust their round budgets. Each complete chain uses 1,125–1,144 evaluations
including its chemistry and radius parents, yet none meets full-fit convergence.
This staged initialization therefore does not resolve the problem under the
declared procedure and cumulative budget.

### A separate feasibility-tolerance mechanism

The [final saved trace analysis](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/finaltrace_summary.json)
and its [analysis script](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/analyze_traces.py)
identify a concrete boundary-handling inconsistency. The physical geometry guard
allows a numerical residual of `1e-10` in physical length units. The direction
projection accepts `1e-9` in row-normalized constraint coordinates. A centre-only
containment row has norm 5, so that projection tolerance can permit a physical
residual up to `5e-9`, larger than the geometry guard accepts. In addition, clipping
an admissible negative current slack to zero removes the projection's request to
restore that small deficit inward.

Centre-only start 1 illustrates the effect. Of 600 evaluations, 537 were rejected
for geometry and two for raw probabilities; only 61 reached the likelihood. Its
saved minimum box clearance was about `-9.99969e-11`, just inside the guard's
numerical allowance. The rejected box clearances had median `-1.01063e-10`, just
outside it, with repeated runs of rejected trials. The guard correctly prevented
invalid likelihood evaluation, but the search consumed its budget near a boundary.
The tolerance-derived bound is an allowed maximum, not a claim that every refused
trial attained it.

This mechanism does not explain every failed fit. Independent joint start 0 had
**zero** geometry or probability refusals, accepted 443 of its 450 steps without
backtracking, and never activated the direction-size cap. Its failure therefore
cannot be attributed to repeated boundary rejection or an excessively small cap.
That trajectory supplies the separate nonsmooth endpoint evidence below.

Across all three independent joint fits, only 14 of 1,417 trials were geometry
refusals. The sequential joint fits had none in 1,461 trials. Each of those six
fits accepted 450 steps, usually full steps, without activating the direction-size
cap. Boundary rejection alone cannot explain the full-joint failures. Likewise,
the three sequential radius fits had no unsafe refusals but exhausted their
evaluation budgets after 319–386 accepted steps.

The trace analysis is descriptive work on saved evaluations, outside the fitting
budget. For exceptions inside unfinished rounds, iteration-level histories are
incomplete even though the accepted checkpoint and evaluation trace survive.
Complete backtracking, step-cap or secant-history claims must not be inferred for
those unfinished rounds.

### Endpoint evidence from the archived full fit

The [archived endpoint probes](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/archived_v3_knots.json)
reproduce the version 3 default-start mean NLL exactly. Each of the four selected
radius coordinates has a nearby verified one-coordinate cusp minimum. Three
correspond to interior-profile knots; the fourth is at a particle-surface crossing,
where the interior clamp and constant source continuation can both contribute.
Fresh and frozen assignments agree through these probes, with zero assignment
changes. This demonstrates a source of derivative discontinuity separate from
assignment changes.

[View the archived radius-cusp figure](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/archived_radius_cusps.png).
It plots the saved one-coordinate probes; its
[provenance](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/archived_radius_cusps_provenance.json)
and [plotting script](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/plot_archived_cusps.py)
are preserved alongside it. The figure does not add fitted points or a joint
optimality certificate.

The evidence does **not** show that the endpoint is a joint optimum, that every
remaining residual comes from a cusp, or that radius-only failures have the same
cause. In particular, the full-space ranking limitation below applies to endpoint
probes from reduced fits. A failed geometry probe is retained as an invalid
diagnostic point; it is not scored as a valid likelihood or counted as an invalid
likelihood evaluated during fitting.

### Separate posthoc active-space checks

After the prescribed run, a bounded
[endpoint-analysis script](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/analyze_active_endpoints.py)
ranked radius coordinates using the actual radius-only active space. This corrects
the ranking limitation of the original full-space endpoint helper without changing
the fitted model or any saved attempt. The three largest active radius components
were inspected in each independent radius-only start.

All nine selected one-coordinate sections have feasible, finite-difference-verified
sign-changing cusps, with fresh and frozen assignments agreeing and no probe
assignment switches. Six are at surface clamp/source-continuation crossings and
three at interior knots, between `3.54e-7` and `3.30e-5` physical length units from
the saved endpoint. For five of the nine, both one-sided derivative magnitudes
exceed `1e-5` in the scaled radius coordinate. These probes needed no coupled
geometry movement. The three saved records are
[start 0](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/posthoc_radii_start_00.json),
[start 1](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/posthoc_radii_start_01.json)
and [start 2](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/posthoc_radii_start_02.json).

Five endpoints whose recorded budgets prevented a fresh final block evaluation
were also checked after the run. Their fresh active residuals all remain above
`1e-5`:

| Saved endpoint | Assignment changes on refresh | Fresh active residual | Change in mean NLL from saved objective |
| --- | ---: | ---: | ---: |
| Centre-only start 0 | 18 | `5.00e-5` | `-1.03e-9` |
| Centre-only start 1 | 2,239 | `4.58e-4` | `-1.34e-6` |
| Sequential radius start 0 | 34 | `7.16e-5` | `-1.37e-8` |
| Sequential radius start 1 | 4 | `4.31e-5` | `+1.58e-13` |
| Sequential radius start 2 | 9 | `6.42e-5` | `+4.27e-14` |

The 39 radius-probe evaluations and ten block/value evaluations all passed their
physical guards. They are **49 additional diagnostic evaluations outside the fit
budgets**, not continuation of optimization. Original failure statuses remain
unchanged. The local cusp evidence does not establish joint nonsmooth stationarity,
and a surface crossing cannot be attributed uniquely to the interior clamp versus
the source continuation. Probe timings overlap the full test run and are not
performance benchmarks.

### Next development step

The next version should first reconcile projection feasibility with the physical
guard: preserve the need to restore small negative slack, certify constraints in
consistent physical units, and keep a checked numerical margin inside the domain.
This is a targeted feasibility repair, not permission to relax the physical guard
or declare budget-limited fits converged. It needs boundary regression cases and a
fresh versioned run; the present source and outcomes stay frozen.

Then separate that repair from the nonsmooth optimization question. One route
retains the existing model and uses an appropriate search and stationarity check.
Another changes the observation model to differentiable monotone interpolation
and consistent surface joins. The latter needs its own frozen comparison;
smoothing only the interior knots leaves other nonsmooth terms. Full-space probes
do not settle reduced-space behavior, and one-coordinate cusp tests do not certify
joint convergence. The final sequential outcomes confirm that warm initialization
alone has not supplied a numerically reliable full fit. The separate active-space
probes strengthen the evidence for nonsmooth geometry directions without changing
any prescribed fit status or promoting a method.

These numerical questions take priority over further neural training. A useful
staged initialization would still need a complete full-fit certificate and a
comparison that accounts for its extra computation.

## Starting evidence and scope

E21 version 3 tested only development case 0: 95 detected candidates and 64,064
fitting atoms. Each of three starts exhausted the declared three rounds of 150
iterations. Final stationarity residuals were approximately 0.000098, 0.000186 and
0.000139, above the required 0.00001. Two starts passed the refreshed-assignment
stability check. None was eligible for selection. These facts motivate numerical
diagnosis; improving the likelihood alone did not satisfy convergence.

The new diagnostic has a separate, versioned report and artifact namespace. It uses
only two inputs:

1. **Observed development case 0:** the existing fitting observations and the
   original observed catalogue with its saved feasible-radius preparation. True
   particle locations or radii do not initialize, select or modify this fit. The
   same development realization has already informed several implementation
   revisions, so it provides no independent confirmation evidence.
2. **Small model-matched control:** unequal-radius spheres, fixed positions and
   fixed physical parameters generate probabilities using exactly the fitted
   model, including its smooth interface blend, piecewise linear interior profile
   and fixed interface width. The response is the
   generating probability, rather than a new random binary mark. This is an
   expected-likelihood numerical control with known geometry, not an APT observation
   model or an independent physical recovery study.

Unequal radii avoid the exact equal-radius capillary-length/equilibrium-concentration
confounding already identified in the model. They do not guarantee practical
identifiability. The control must verify its raw probabilities without clipping.
Its geometry, parameters, observation positions, seeds and starting values are
frozen before fitting. Truth is legitimate for constructing this explicit control;
that privilege does not extend to the observed case.

The control has 6,000 uniformly sampled positions in a box of side 32, with position
seed `2026110201`. Its sphere radii are 2, 3 and 4, centred at `(7, 8, 8)`,
`(17, 10, 12)` and `(12, 21, 20)`. The generating constants are
`c_eq = 0.03`, `ell = 2`, `xi = 5`, `c_inf = 0.1`; the single starting set is
`c_eq = 0.032`, `ell = 1.5`, `xi = 6`, `c_inf = 0.09`. The generating and initial
profile levels are separately specified in the design. All lengths use the model's
units. This is a deliberately favorable local numerical control.

## Primary comparison: independent parameter-block ablations

Each stage starts afresh from the same prepared geometry, prescribed chemistry
start and initial interior profile. The stages are independent ablations, not a
chain in which the previous stage has already optimized part of the problem.

| Stage | Free parameters | Fixed parameters |
| --- | --- | --- |
| Chemistry and profile | Equilibrium concentration `c_eq`, capillary length `ell`, screening length `xi`, far-field concentration `c_inf`, and all interior-profile logits | Every candidate centre and radius; interface width |
| Add radii | Chemistry, interior profile and every candidate radius | Every candidate centre; interface width |
| Add centres independently | Chemistry, interior profile and every candidate centre | Every candidate radius; interface width |
| Full joint fit | Chemistry, interior profile, radii and centres | Interface width |

The radius-only and centre-only geometry releases are sibling ablations. Comparing
them with the full fit helps distinguish either geometry block from their joint
interaction. All four stages begin with the same base state; the full stage does
not inherit the optimized geometry of either sibling.

The shared monotone interior profile is a nuisance component of the likelihood and
must be fitted even in the first stage. Fixing an incorrect profile could otherwise
make a geometry-fixed chemistry fit appear to fail for an unrelated reason. There
are six profile knots with seven level parameters in the existing implementation.
The fixed interface width remains 0.3.

The observed catalogue has one fixed candidate count, order and identity mapping.
No particle is born, deleted, merged or replaced. Geometry preparation is recorded
once; a supposedly fixed radius must not be silently changed by later preparation.
The initial-state identity and the free-parameter mask are recorded for every run.
Frozen vector coordinates must remain unchanged through optimization.

All three prescribed starts are retained: the existing default, `(ell, xi) = (1, 3)`
and `(6, 12)`. The observed input reuses the archived feasible initial chemistry for
each start, including E21's data-derived default, and the original default interior
profile.
The radius, centre and full stages do not receive a successful chemistry-stage solution
in this primary comparison. A failed stage remains a result; it is not replaced by
a favorable new initialization. The observed case therefore has twelve primary
attempts. The small model-matched control uses the same four independent stages
with one prospectively fixed prescribed start.

## Secondary comparison: sequential warm starts

A separately identified sequence can ask whether staged optimization is a useful
initialization strategy. For each prescribed start, its chemistry-stage terminal
state initializes the radius stage, and its radius-stage terminal state initializes
the full joint stage. Transfer includes the complete parameter vector: concentrations,
capillary and screening lengths, interior profile, centres and radii.

Each transfer records its parent run, parent start, state identity and validity.
Only a saved accepted state satisfying the physical domain can initialize the next
stage. A rejected trial is not a warm-start state. If no valid accepted state exists,
the dependent stage records that failure explicitly. The diagnostic can distinguish
a valid budget-limited parent from a converged parent; carrying the former forward
does not retroactively make it converged.

This sequence has additional computation and inherited optimization history.
Therefore, its final loss cannot be presented as evidence that parameter staging
alone improves the optimizer under an equal budget. Report per-stage and cumulative
iterations, evaluations and elapsed time. A later efficiency claim would require a
separately frozen comparison with matched total computation.

The chemistry-stage parent is shared with the primary comparison rather than run
again. The two additional stages for each of three observed starts give six
secondary attempts, for eighteen observed attempts in total. The small control
has four primary attempts and no secondary chain.

## Numerical contract and stopping rules

The diagnostic retains the E21 budget and validity requirements:

- 150 iterations per round, at most three rounds per start, and 600 objective
  evaluations per start across all rounds;
- constrained stationarity tolerance `gtol = 1e-5`, using the same declared
  parameter coordinates and scaling;
- the same physical bounds, contained separated spheres, raw-probability guards,
  fixed probes and moving surface probes;
- float64 physics on CPU, with runtime and thread settings recorded;
- refreshed nearest-particle assignments and a final certificate for the actual
  objective, rather than just the previous round's fixed-assignment surrogate.

For a reduced stage, the stationarity calculation and constraint projection use
only the free coordinates. Gradients in frozen coordinates are reported separately
for diagnosis and cannot prevent convergence of the reduced problem. Conversely,
convergence with geometry fixed does not certify convergence of the full problem.
Constraint derivatives must use the corresponding free-coordinate columns.

These derivative checks describe the current smooth branch of a piecewise-defined
objective. The profile has knot crossings and the particle assignment can change;
the reported branch derivatives alone are not a nonsmooth optimality certificate.
Separate radius-knot probes inspect one-coordinate sections near selected endpoint
residuals, with fresh and fixed assignments. They retain invalid probes explicitly
and never alter the fitted state or declare joint convergence. The frozen settings
inspect four radius coordinates and at most two crossings per coordinate, with a
side offset of `1e-6` in the scaled radius coordinate, reduced when needed to avoid
neighboring crossings. Probe time is recorded
separately from the optimization budget.

The endpoint helper ranks radius components of the **full-space** projected
direction, even when the fitted stage fixed some geometry coordinates. For a
radius-only stage, that ranking can select a radius whose feasible full-space
movement requires moving its centre. A one-radius probe can then correctly fail
the containment or separation guard. The absence of a detected cusp among those
four selected coordinates does not establish the absence of relevant cusps in the
reduced problem. Use the fitter's active-space certificate to assess reduced-stage
convergence. Invalid endpoint probes are separate diagnostic evaluations, not
invalid likelihoods scored during fitting.

An iteration cap, evaluation cap, stalled search, invalid trial or assignment
instability does not become convergence. A valid state at a per-round cap may use
the remaining declared rounds and evaluation budget. Unsafe trial probabilities
are rejected before likelihood evaluation. The numerical prerequisite for a stage
requires physical feasibility, valid sampled probabilities, a satisfactory active
stationarity certificate and the required refreshed-assignment stability.

The independent stages diagnose different optimization problems; all prescribed
results are retained even if an earlier stage fails. The secondary chain requires
a valid saved parent state. Neither pathway is a promotion procedure. No outcome
changes the four-case E21 preflight accounting or its later confirmation gate.

## Failure records and reproducibility

Every prescribed start has a terminal record, including initialization failures,
exceptions and exhausted budgets. Records distinguish:

- the requested initial state and any documented feasibility adjustment;
- the last accepted feasible state, with the complete vector and decoded physical
  quantities, geometry and interior profile;
- the last evaluated or rejected trial when available, labeled separately;
- fitting objective, active and full-gradient diagnostics, constraint slack,
  geometry and sampled-probability checks, bound contacts and assignment changes;
- iteration and evaluation counts, rejected-trial counts, status, termination
  reason, and parent lineage for a sequential run.

When a quantity is unavailable, the record states why instead of fabricating a
successful certificate. An exception must not discard a previously accepted state.
No start is silently omitted from summary denominators.

The diagnostic verifies the archived case's data and catalogue identities before
using them. New execution provenance identifies the frozen diagnostic design,
complete relevant source manifest, runtime, input-file hashes, seed settings and
starting-state identities. Reuse is permitted only when all required identities
agree. A new report cannot relabel an old cached result as a current execution.
Existing artifacts are not silently overwritten, and the archived E21 reports are
never rewritten by this procedure.

Only the development case 0 input is allowed. Confirmation and reserved-test inputs
must not be opened or generated. Held-back observations do not choose starts,
parameter blocks, tolerances, stopping rules or the state passed between stages.

## Verification

The [recorded software validation](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/validation.json)
reports **631 tests passed in 129.73 seconds**. These checks
include active-coordinate invariance, complete-state preservation and transfer,
domain guards, malformed protocol inputs, archive provenance, interrupted attempts
and post-fit diagnostic behavior. Passing tests is distinct from the measured
numerical failures above.

The [saved archive verification](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/archive_verification.json)
checks all 22 fit artifacts and 101 relevant source files against both the current
tree and the archived source snapshot. It verifies exact initial states, frozen
coordinates, saved terminal-state/fit agreement and sequential lineage. All three
independent joint endpoints reproduce the original version 3 fits, and no failed
fit is silently selected. The
[verification script](../../experiments/reconstruction/results/e21_solver_diagnosis_v1/verify_archive.py)
is preserved with the evidence. The eight posthoc output files have separately
verified script and report identities. These are integrity and accounting checks,
not physical-readiness or uncertainty-validation claims.

## Interfaces and commands

The runner accepts only the small control, observed development case 0, or both.
From the repository root, the complete prescribed diagnostic is:

```sh
python -m experiments.reconstruction.e21_solver_diagnosis --case all
```

The control can be run first and the observed case added under the same unchanged
execution provenance:

```sh
python -m experiments.reconstruction.e21_solver_diagnosis --case control
python -m experiments.reconstruction.e21_solver_diagnosis --case observed
```

`--design`, `--output` and `--artifacts` select the design file and new diagnostic
destinations. Defaults are `e21_solver_diagnosis_design.json`,
`results/e21_solver_diagnosis_v1` and `results/e21_artifacts/diagnosis_v1`, relative
to the reconstruction experiment directory. A changed design or source requires a
new version and output directory. These commands are instructions for the frozen
diagnostic. The saved report above records its completed attempt inventory.

| Interface | Responsibility |
| --- | --- |
| `load_design` | Validate diagnostic scope, stage masks, lineage and runtime |
| `load_observed_case` | Verify pinned archive identities and extract fitting rows only |
| `make_control` | Generate deterministic exact-model probability targets |
| `initial_states` | Export the full common starting vectors |
| `resolve_start` | Select the prescribed cold state or the same start's accepted parent state |
| `run_stage` | Preserve one immutable attempt, its fit artifact and terminal result |
| `run_diagnosis` | Execute the requested fixed inventory with source/runtime provenance |

The report distinguishes completed accounting from numerical convergence. A
nonconverged fit is a completed diagnostic outcome. An interrupted or malformed
execution is an infrastructure failure. Neither a zero process exit code nor
`all_diagnostic_stages_complete` is a physics-readiness decision; promotion and
inference-validation flags remain false.

## Interpreting the outcomes

| Outcome | Supported interpretation | What it does not establish |
| --- | --- | --- |
| The small exact-model control fails with geometry fixed | A numerical or implementation problem persists even in that simplified setting | The location of the bug, or failure of the physical idea |
| Control passes; observed chemistry/profile stage fails | The observed case introduces a harder likelihood, constraints, scale or model mismatch | That the detector, physical model or optimizer is uniquely responsible |
| Chemistry/profile passes; adding radii fails | The added radius degrees of freedom or their interaction with constraints prevent the required convergence under this budget | That radius information is absent from the observations |
| Chemistry/profile passes; centre-only release fails | Centre optimization and its interactions, including changing assignments, require further diagnosis | That assignment changes alone caused the failure |
| Radius-only and centre-only releases pass; full fit fails | Joint geometry optimization adds a difficulty absent from either restricted problem | A unique diagnosis of the relevant coupling or parameter scaling |
| A sequential chain reaches full convergence | That frozen staged initialization reaches a certified full fit for this case and budget sequence | Equal-compute superiority, identifiability, calibrated uncertainty or generalization |
| All independent stages pass | The chosen numerical setup satisfies its certificates on these diagnostic inputs | Readiness on all four E21 smoke cases or a neural advantage |

The small control changes both scale and model match. Success on it cannot exclude
poor conditioning on a 95-particle catalogue. Likewise, a physical parameter estimate
close to its generating value is not a confidence interval or a uniqueness result.

The next solver or scaling revision should follow the saved diagnostics and receive
its own versioned settings. Before the neural comparison resumes, the complete
four-case baseline preflight must still pass under one frozen implementation.
Capillary profiling, the exact-null comparison, and calibrated uncertainty remain
future work; this diagnostic does not implement or validate them.
