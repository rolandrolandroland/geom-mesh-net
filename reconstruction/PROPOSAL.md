# Implicit Neural Reconstruction of Atomic Clusters

*Project proposal · geom-mesh-net · September 2026*

## Summary

An atom probe measurement is a three-dimensional map of individual atoms and
their chemical identities, taken from a needle of material a few tens of
nanometres across. Metallurgists use it to see precipitates: nanometre-scale
clusters of solute atoms that control how strong and how stable an alloy is. But
the instrument records only a fraction of the atoms, blurs their positions, and
does not say which atoms belong to which precipitate.

Where the precipitates are, how large they are and what they contain must
therefore be reconstructed from the recorded atoms. Today that is done with
smoothing and cluster-finding heuristics whose accuracy nobody can measure,
because a real specimen does not come with an answer key.

This project builds the answer key, then uses it to find out how well
reconstruction can be done. It works with simulated atom probe data, where the
simulator's own random procedure can be replayed to give the exact probability
that each atom is a solute atom. Against that reference it measures:

- how close the standard method comes to the truth, and where it falls short;
- whether a neural implicit field does better when fitted to a single
  measurement. Such a field is a small network that represents solute
  concentration as a continuous function of position;
- whether a network trained on thousands of simulations does better still, and
  when its learned expectations mislead it;
- whether better fields give better measurements of individual precipitates than
  the cluster-finding methods in use;
- whether building physical laws into the network recovers both the field and the
  physical constants behind it, and whether the method can tell when those laws
  do not hold. The laws in question are diffusion around precipitates and the
  blur of the instrument.

Every stage has a pass-or-fail criterion written down before it runs. Every score
is reported beside the simplest baseline and beside the exact reference. The first
two stages are complete. The reference has been built and checked against 2.9
million simulated atoms. Standard smoothing has been measured against it, and
leaves room for improvement in 48% of test cases, mostly for small precipitates.

| | Question | Stage | Status |
| --- | --- | --- | --- |
| Q0 | Can the true solute probability of every simulated atom be computed exactly? | 0 | answered: yes |
| Q1 | How close does standard practice come to the truth, and where does it fall short? | 1 | answered: close for large precipitates, short for small ones |
| Q2 | Does a neural field fitted to one measurement beat tuned smoothing? | 2 | next |
| Q3 | Does learning from simulations help, and when does it mislead? | 3 | planned |
| Q4 | Do better fields give better measurements of precipitates? | 4 | planned |
| Q5 | Can a physical law be built into the network, and can the network tell when the law fails? | 5 | planned |
| Q6 | Can the instrument's blur be undone inside the training objective? | 6 | optional |

---

## 1. Background

### 1.1 What an atom probe records

In atom probe tomography, a sharp needle of material is held at high voltage, and
its surface atoms are pulled off one at a time as ions. Each ion flies to a
position-sensitive detector. Its flight time identifies its chemistry, and its
impact point, traced back through a reconstruction, gives its original location.
The result is a point cloud of millions of atoms, each with a position and a
species.

Two imperfections of the instrument shape everything in this project.

- **Detection efficiency.** Only a fraction of the atoms reach the record: roughly
  37% to 80%, depending on the instrument. The missing atoms are lost at random,
  regardless of their chemistry.
- **Spatial blur.** Reconstructed positions scatter around the true ones,
  typically more across the specimen than along its depth.

### 1.2 Why the solute field matters, and how it is estimated today

In many alloys, strength and resistance to high temperatures come from
precipitates. Analysts therefore want:

- where each precipitate is;
- how large it is and what it contains;
- the composition of the matrix around it;
- how concentration changes across its boundary.

There are two standard ways to get from atoms to those quantities.

- **Delocalisation and isoconcentration surfaces.** The volume is divided into
  voxels and each atom's contribution is smoothed over its neighbourhood with a
  Gaussian kernel, giving a local concentration. Surfaces drawn at a chosen
  concentration threshold outline the precipitates. Concentration profiles
  measured outward from those surfaces, called proximity histograms, show the
  interfaces.
- **Cluster finding.** Algorithms such as the maximum separation method link
  solute atoms that lie within a chosen distance of one another, and keep the
  groups above a minimum size. DBSCAN is a close relative.

Both depend on parameters the analyst chooses: kernel width, concentration
threshold, linking distance and minimum size. Different reasonable choices give
different numbers of precipitates with different sizes. On a real specimen there
is no way to know which choice was right.

### 1.3 What this package already provides

This repository grew out of the statistical methods of Bennett, Proudian and
Zimmerman (2023) and the R packages rapt and rTEM. It already contains four
things.

- **A simulator** of clustered atom probe data. Four parameters describe each
  simulated specimen: the solute concentration inside precipitates, the solute
  concentration in the matrix, the mean precipitate radius, and the spread of
  radii. One thousand specimens of 216,000 atoms each have been simulated.
- **Spatial statistics**: a Python port of the summary functions used in atom
  probe analysis, and of the fourteen features built from them. The functions are
  nearest-neighbour distance, empty-space distance, Ripley's K and a
  guest-to-host cross function. The port reproduces the R implementation to about
  twelve decimal places.
- **A calibrated inference method**, from a separate line of work, that estimates
  the four population parameters of a specimen with honest error bars: its 90%
  intervals contain the truth 88–92% of the time. That method answers *what kind*
  of precipitate population a specimen has. It does not say *where* the
  precipitates are, which is this project's subject.
- **Earlier neural-field experiments**, which fitted small networks to individual
  specimens. They stopped at their own validation gate. A later review showed they
  could not have been conclusive: they had no simple baseline, no positional
  encoding, and inputs that were partly artefacts.

### 1.4 Why simulation makes the question answerable

A reconstruction can only be scored if the truth is known, and for a simulated
specimen it is. The simulator decides which atoms become solute atoms by a known
random procedure, applied to stored inputs: the atom positions and the
precipitate centres and radii. Replaying that procedure gives each atom's exact
probability of being a solute atom. This project calls that reference **the
oracle**.

Comparing reconstructions against the oracle also needs a fair scoring rule. The
one used here is the log loss on the atoms the detector did not record: for each
missing atom, how surprised the reconstruction is by the atom's actual identity.

On average, that score is a fixed amount no method can avoid, plus the divergence
between the reconstruction and the truth. So when two methods are compared on the
missing atoms, the difference between their scores is exactly the difference in
how far each one is from the truth, and computing it does not require the truth.
The same score can be computed on a real specimen by setting some recorded atoms
aside.

### 1.5 The starting point

Four facts shape the design before any modelling begins.

1. **Missing atoms do not change what is being estimated.** The detector loses
   atoms regardless of their chemistry, so the solute probability at a location is
   the same whether or not an atom there was recorded. Reconstruction here means
   estimating from fewer samples, not inverting a distortion. Nothing needs to be
   invented.
2. **In this simulator, positions carry no information; identities carry all of
   it.** Atoms are spread uniformly, so the right training objective scores
   predicted probabilities against recorded identities (binary cross-entropy).
   Conservation of the total amount of solute then follows automatically and needs
   no separate term.
3. **The true field has sharp edges.** Inside a simulated precipitate, the solute
   probability falls towards zero at the boundary and then jumps to the matrix
   value. Assuming the field is smooth is wrong exactly where reconstruction is
   hardest.
4. **Spatial statistics survive missing atoms.** Summary functions such as Ripley's
   K have the same expected value however many atoms are lost at random. A
   reconstruction that reproduces them has shown nothing about its accuracy.

A pilot on twelve development specimens, and the completed first stage,
established the following.

- **Standard smoothing is good for large precipitates and weak for small ones.**
  Tuned on recorded atoms, it closes 81–96% of the distance from a constant guess
  to the truth for large precipitates. For small ones it closes only 45–82%, and
  least when few atoms are recorded. That is where a better method has room to
  show itself.
- **The oracle is exact.** Against the simulated identities of 2.9 million atoms,
  its probabilities are calibrated to within 0.0008.
- **Several defects surfaced, and were fixed or recorded.**
  - The grid previously used as ground truth was wrong inside precipitates, by
    about 0.2 in probability.
  - The specimen-wide statistics fed to an earlier network were computed on all
    atoms rather than solute atoms, and carried no information.
  - Precipitate centres in the original dataset sit on a regular lattice, an
    artefact a learning method could exploit. A benchmark dataset with randomly
    placed centres was generated instead.
  - In 22 of the original specimens, the lattice silently lost its outer layer.
    Those specimens have far fewer precipitates than their parameters specify.

---

## 2. The questions

Each question below is answered by one stage, described in Section 3. They build
on one another: later questions reuse the oracle, the benchmark and the baselines
of earlier ones.

### Q0. Can the true solute probability of every simulated atom be computed exactly?

*Answered: yes, in Stage 0.* Everything else depends on this. Without an exact
reference, a method that reproduced the reference's own mistakes would score as
perfect.

### Q1. How close does standard practice come to the truth, and where does it fall short?

Analysts use delocalisation because it works reasonably well, but "reasonably"
has never been measured. This question maps the gap between tuned smoothing and
the truth across precipitate size, solute concentration and detection efficiency.

The map decides whether the rest of the project is worth doing. If smoothing is
already close to the truth everywhere, no reconstruction method can improve on it
by much.

### Q2. Does a neural field fitted to a single measurement estimate the solute field better than tuned smoothing?

Neural implicit fields represent a quantity as a continuous function learned by a
small network. They have transformed 3D shape reconstruction. Plain networks,
however, struggle to represent sharp detail, and positional encodings such as
Fourier features were designed to fix that.

This question asks whether such a field, fitted only to one specimen's recorded
atoms, recovers small precipitates and sharp edges better than smoothing, or
whether it amounts to a more expensive smoother.

### Q3. Can a network trained on many simulations learn what precipitates look like, and reconstruct unseen measurements better, and when does that learned knowledge mislead it?

A method fitted to one specimen knows nothing about precipitates in general. A
network trained on thousands of simulated specimens can learn that precipitates
are compact, how their composition varies from centre to edge, and how they relate
to the recorded atoms around them. That knowledge should help most where data are
sparse.

It is also a risk. A network that has only ever seen spheres may draw spheres
where the truth is not spherical, or may exploit regularities of the simulator
that no real material shares. Both halves of the question are measured.

### Q4. Do better fields give better measurements of individual precipitates than the cluster-finding methods in use?

Practitioners report precipitates, not probability fields: how many there are, how
large they are, and what they contain. This question asks whether turning
reconstructed fields into precipitates beats the maximum separation method and
DBSCAN on detection, size and composition.

It also delivers something the inference work needs: the first comparison between
its calibrated parameter estimates and the numbers standard practice would have
produced.

### Q5. Can a governing physical law be built into the network so that it recovers both the solute field and the physical constants behind it, and can the method recognise when the law does not apply?

A physics-informed neural network is trained to satisfy a governing equation as
well as to fit data. The equation is checked at many points by automatic
differentiation.

Around growing or coarsening precipitates, solute diffuses. Large precipitates
draw solute from a depleted zone around them, while small ones shed solute into an
enriched zone. The resulting concentration profile obeys a known equation, with a
boundary condition set by the curvature of each precipitate. The simulator does
not yet contain this physics, so it will be added.

The question is whether enforcing that equation lets a network recover faint
concentration gradients, and two physical constants, from sparse and noisy data.
The constants are the capillary length and the screening length. Just as
important is whether the method can tell from its own data when the equation is
wrong, instead of imposing it with confidence.

### Q6. Can the instrument's blur be undone by building it into the training objective?

Blur is the second imperfection of the instrument. If the blur is known, a network
can represent the sharp field while being trained to explain the blurred
observations. Unlike missing atoms, this is a true inverse problem, and
deconvolution is known to amplify noise.

The question is how much sharpness can be recovered, and whether a network trained
on simulations does better than one fitted to a single specimen. This stage is
optional.

### A cross-cutting question: what can classical spatial statistics contribute inside a neural reconstruction?

Three uses have been proposed:

- as extra inputs describing the whole specimen;
- as features combined with a convolutional network to estimate parameters;
- as a loss that forces reconstructions to reproduce the specimen's statistics.

Each is examined in the stage where it belongs (Section 3.9). None of them is, on
its own, a physics-informed method, because spatial statistics summarise data;
they are not physical laws.

---

## 3. Methods

### 3.1 Shared foundations

**The simulated specimens.** Each specimen is a 60 × 60 × 60 box holding 216,000
atoms at random positions, with spherical precipitates placed inside it.

- Inside each precipitate, a set fraction of the atoms become solute atoms, chosen
  with a preference for the centre.
- Outside every precipitate, atoms become solute atoms at the matrix
  concentration.
- The detector is simulated by keeping each atom independently with probability
  η, the detection efficiency.

**Two datasets.** The original 1,000 specimens place precipitate centres on a
regular lattice. They remain the basis of the inference work, and are used here
only to measure how much a learning method exploits that lattice. This project's
benchmark is a second set of 1,000 specimens with identical parameters but
randomly placed centres, as in the published design.

**The oracle.** For every atom of every benchmark specimen, the oracle gives the
probability of being a solute atom, computed by replaying the simulator's
labelling. Its correctness rests on checks that do not share its code:

- exact calculation for small cases;
- thousands of replays of the simulator's own labelling functions;
- calibration against the simulated identities of millions of atoms.

**The benchmark.** Specimens were divided before any modelling began.

| Group | Specimens | Used for |
| --- | --- | --- |
| Development | 100 | debugging methods and choosing thresholds |
| Training | 700 | methods that learn across specimens |
| Validation | 100 | choosing among those models |
| Test | 100 | scoring, once per stage |

Each specimen is observed at three efficiencies: 10% as a stress test, then 37%
and 80%, the ends of the range real instruments reach. The kept atoms are fixed by
stored seeds, so every method sees exactly the same observations.

Specimens are grouped by precipitate radius (3–6, 6–10 and 10–15) and by
in-precipitate concentration (below or above 0.5), so that results can be reported
separately where they differ.

**Scoring.**

- The primary score is **excess log loss**: the log loss on the atoms the detector
  missed, minus the oracle's. It measures the average divergence from the truth,
  in nats per atom.
- **Gap closed** rescales that score for one specimen: 0 means no better than a
  constant guess, and 1 means as good as the oracle.
- Every score is also reported **by region** (the matrix, the rims of precipitates
  and their cores), because 86% of atoms are in the matrix and an overall average
  would hide what happens at precipitates.
- **Calibration** asks whether predicted probabilities mean what they say.
- **Precipitate-level scores**, meaning detection, size and composition errors,
  are added in Stage 4.
- A **predictive check** relabels the specimen according to the reconstruction and
  compares its spatial statistics with the truth. Because statistics survive
  missing atoms, this counts as a necessary condition only.

**Where improvement is possible.** A specimen at a given efficiency has
*headroom* when tuned smoothing leaves at least 15% of the gap to the oracle open,
and at least 0.01 nats per atom. Neural methods are judged on these cells, and
must do no harm elsewhere.

**How decisions are made.**

- Each stage's pass-or-fail criterion is written into the roadmap before the stage
  runs, and is not changed afterwards.
- Thresholds are set on development specimens only.
- Model choices, such as kernel widths, network sizes and stopping points, use
  recorded atoms only, never the atoms being scored. Every method could therefore
  be applied unchanged to a real specimen.
- Comparisons are paired specimen by specimen, and reported as medians with
  intervals.
- When a result overturns an earlier conclusion, both are kept on the record.

### 3.2 Q1: how far standard practice is from the truth (Stage 1)

Three baselines are fitted to each test specimen's recorded atoms, at each
efficiency:

- **a constant**: the recorded solute fraction, everywhere;
- **Gaussian smoothing**, the delocalisation estimator of standard practice, with
  its kernel width chosen by five-fold cross-validation on recorded atoms;
- **adaptive smoothing**, whose kernel widens where atoms are sparse, with its
  neighbourhood size chosen the same way.

The result is a headroom map: excess log loss and gap closed for each baseline, by
precipitate size, concentration, efficiency and region.

**Success criterion.** At least a quarter of test cells have headroom. The pilot
suggests about 40%.

**If it fails,** smoothing is within 15% of the truth almost everywhere. That is
itself the answer to Q1. Stages 2 and 3 are skipped, and smoothing goes forward to
precipitate measurement.

### 3.3 Q2: a neural field fitted to one measurement (Stage 2)

The field is a small multilayer network that maps a position to a solute
probability. Its input first passes through random Fourier features, sines and
cosines at many frequencies, so that it can represent sharp edges. A
sine-activated network (SIREN) is the alternative.

The network is trained on one specimen's recorded atoms, with binary
cross-entropy and nothing else. Its frequency scale, width and stopping point are
chosen on a fifth of the recorded atoms, held aside.

Two comparisons isolate what matters.

- **The same network without positional encoding.** This removes the confound that
  stopped the earlier experiments.
- **A smoothness penalty on the field's gradient.** It is added as a test with a
  stated prediction: validation will reject it, and forcing it will make
  precipitate rims worse.

The stage runs on 36 test specimens, six per group, at all three efficiencies.

**Success criterion.** On cells with headroom, the neural field must:

- beat the better smoothing baseline in at least two-thirds of cells;
- close at least a fifth of smoothing's remaining gap in the typical cell;
- be no less calibrated.

Elsewhere it must do no meaningful harm.

**If it fails,** the conclusion is that a field fitted to one specimen is a
reparameterised smoother at this data size. The project continues, because Stage
3 works by a different mechanism and Stage 5 reuses this code.

### 3.4 Q3: learning from simulations (Stage 3)

The network follows the design used to reconstruct 3D shapes, including human
bodies, from sparse point clouds: convolutional occupancy networks and implicit
feature networks.

- Recorded atoms are binned into a coarse 3D grid of atom counts and solute counts.
- A 3D U-Net turns that grid into a field of learned features.
- A small decoder reads the features at any position and outputs a solute
  probability.

The network is trained on the 700 training specimens. Each training step:

- picks a random efficiency;
- crops a random region;
- rotates or reflects it, since the simulated physics has no preferred direction;
- scores the network against the identities of the atoms that were not recorded.

Under this objective, the best possible network outputs the probability of a
solute atom given the observation. The same principle makes the inference work's
posterior estimates correct.

Four variants test which parts of the design matter:

- no continuous decoder;
- no positional detail in the decoder;
- specimen-wide spatial statistics added as a conditioning signal;
- training against the oracle's probabilities instead of the recorded identities.

Two further experiments measure when learned knowledge misleads.

- **The lattice test** retrains the network on the original dataset, whose centres
  sit on a lattice. The improvement over the random-centre result measures how
  much the network exploits a simulator artefact.
- **The out-of-distribution test** scores the trained network on 50 specimens
  unlike its training data. They have ellipsoidal precipitates, mean radii beyond
  the training range, and uniform composition inside precipitates. The stated
  prediction is that the network degrades more than smoothing does, most of all on
  ellipsoids.

**Success criterion.** On cells with headroom, the network must:

- beat the best of smoothing, adaptive smoothing and the Stage 2 field in at least
  two-thirds of cells;
- close at least 30% of smoothing's remaining gap in the typical cell;
- stay calibrated;
- be worse than smoothing in no group of specimens.

The out-of-distribution result decides whether the method may be called robust,
not whether the project continues.

### 3.5 Q4: from fields to precipitates, against standard practice (Stage 4)

Precipitates are extracted from each reconstructed field by drawing an
isoconcentration surface and splitting it into connected pieces. The threshold is
chosen on validation specimens.

The same is done for the smoothing baseline, which is exactly the isosurface route
of standard practice. The maximum separation method and DBSCAN run on recorded
solute atoms, with their parameters chosen on development specimens by the usual
rules.

Each method's precipitates are matched to the true ones and scored on:

- detection precision, recall and F1;
- the radius and composition errors of matched precipitates;
- the error in matrix composition;
- how many precipitates are wrongly split or merged.

Population estimates from each segmentation, such as mean radius, spread and
concentrations, are compared with the calibrated parameter estimates of the
inference work. A final test asks whether forcing a field to reproduce the
recorded atoms' Ripley's K improves precipitate radii, and what that costs in
accuracy (Section 3.9).

**Success criterion.** At 37% efficiency, on test specimens with at least two
precipitates, the best field-based method must beat the better of the maximum
separation method and DBSCAN on F1 in at least two-thirds of specimens, and on
typical radius error.

### 3.6 Q5: a physics-informed network (Stage 5)

**First, the simulator gains physics a network can enforce.** Around each
precipitate, the matrix concentration becomes a quasi-stationary diffusion profile.

- Far from precipitates, it tends to a background value.
- At each precipitate's surface, it takes the value set by the Gibbs–Thomson
  relation. Through a material constant called the capillary length, this makes
  small precipitates richer in solute at their surfaces than large ones.
- Between precipitates, the profiles overlap and decay over a screening length,
  which is set by how densely precipitates fill the volume.

The combined profile satisfies a screened diffusion equation exactly, everywhere
in the matrix, so the new simulator comes with an exact analytic reference. It is
checked before use:

- the equation's residual must vanish to numerical precision;
- almost no probabilities may need clipping;
- the reference must pass the same calibration checks as Stage 0.

**Then the network.** It is a field with smooth activations; a standard network
cannot be used, because its second derivatives are zero almost everywhere. It is
trained on the recorded matrix atoms with three terms:

- agreement with the recorded identities;
- the residual of the diffusion equation at points throughout the matrix, computed
  by automatic differentiation;
- the Gibbs–Thomson condition at precipitate surfaces.

The capillary length, the screening length and the background concentrations are
learnable, so the network recovers the physical constants along with the field.
Precipitate locations and radii come from smoothing, so nothing is used that a
real specimen would lack.

Each comparison has a stated prediction:

| Comparison | Prediction |
| --- | --- |
| The same network without the physics terms | worse in the matrix, most of all at low efficiency |
| Smoothing | worse still; faint depletion zones are below its noise |
| The exact physical model, fitted with the true precipitate geometry | an upper bound the network approaches |
| Specimens whose matrix deliberately violates the equation | the method rejects its own physics |

The last comparison is essential. Physical constants are reported only when the
physics-informed network predicts held-out recorded atoms at least as well as the
unconstrained network does.

**Success criterion.** Thresholds are fixed on 50 development specimens before the
test run. The physics-informed network must:

- beat the unconstrained network in the matrix, on at least two-thirds of
  correctly specified specimens;
- recover the capillary length and screening length within the committed accuracy;
- accept its physics on at least two-thirds of correctly specified specimens, and
  reject it on at least two-thirds of violating ones.

If the last condition fails, no physics-informed claim is made, whatever the other
two show.

### 3.7 Q6: undoing the instrument's blur (Stage 6, optional)

Recorded positions are displaced by Gaussian noise, wider across the specimen than
along its depth. Atoms are uniform in the simulator, so the identities at blurred
positions follow the true field blurred by the same kernel.

The network represents the sharp field. It is trained to explain the observations
after its own prediction has been blurred, averaged over random displacements, and
it is scored against the sharp oracle at the true positions. Three comparisons:

- a network that ignores blur;
- smoothing;
- the Stage 3 network, retrained on blurred data.

The stated prediction is that deblurring a single specimen helps only while the
blur is small compared with precipitates, and that the network trained across
simulations does best, because deconvolution needs prior knowledge.

If Stages 5 and 6 both succeed, a final experiment combines them: a diffusion-field
specimen observed through blur and missing atoms, with both kinds of physics in a
single training objective.

**Success criterion.** Thresholds are fixed on development specimens first. The
blur-aware network must beat the blur-unaware one against the sharp oracle in at
least two-thirds of test specimens, at the two smallest blur levels.

### 3.8 Presentation (Stage 7)

- Three-dimensional renderings show the recorded solute atoms with the true and
  reconstructed precipitate surfaces, the reconstructed surfaces coloured by error.
- Radial profiles through single precipitates compare the oracle with each method.
- Each stage is written up in the repository's paper-style format.
- The roadmap records every result and every correction.

### 3.9 Spatial statistics inside the network

**As specimen-wide inputs.** Adding the same few statistics at every position gives
a network nothing to locate precipitates with.

- For one specimen, they are a constant.
- Across specimens, they describe what kind of clustering is present, but never
  where the precipitates are.

They could only help as extra context for a network that already sees the local
atoms, and Stage 3 tests that. An earlier version was measured to carry no
information at all, because it was computed on all atoms rather than on solute
atoms.

**Combined with a convolutional network, to estimate parameters.** The idea is
sound, but it belongs to the inference work, not to reconstruction. The in-
precipitate and matrix concentrations are already estimated with calibrated
uncertainty. The useful version would add a learned summary to the existing
inference method, to recover the spread of precipitate radii, which the current
statistics barely constrain.

**As a loss term.** Requiring a reconstruction to reproduce the recorded atoms'
Ripley's K is technically possible, and the recorded atoms do give an unbiased
target. It still fails as a training objective, for three reasons.

- It is not physics.
- Many different fields share the same K.
- It pulls against accuracy. The most accurate field averages over uncertainty
  about where edges lie, and an averaged field should show less short-range
  clustering than the truth.

The term is therefore used where that trade might be worth making, in Stage 4's
precipitate-size test, and as the predictive check.

---

## 4. Plan and timeline

The project runs in two phases that share their first week.

| Week | Work | Answers |
| --- | --- | --- |
| 1 | Stage 0, the oracle and benchmark (done); Stage 1, baselines and headroom map (done) | Q0, Q1 |
| 2 | Stage 2, a neural field fitted to one measurement | Q2 |
| 3–4 | Stage 5, diffusion physics and a physics-informed network | Q5 |
| 5 | Stage 7, renderings and write-up | — |
| 6–7 | Stage 3, a network trained on simulations | Q3 |
| 8 | Stage 4, precipitates against standard practice | Q4 |
| 9 | Stage 6, blur (optional) | Q6 |

The first five weeks form a complete project on their own. They deliver a measured
picture of standard practice, a test of neural fields on single measurements, and
a physics-informed result that checks its own assumptions. The following four weeks
hold the larger payoff for atom probe practice. A failed criterion shortens the
plan rather than extending it.

All of it runs on a laptop with a 16 GB Apple M1. Fitting a field to one specimen
takes about a minute, a physics-informed fit about a quarter of an hour, and
training the Stage 3 network a few hours.

---

## 5. What the project will be able to claim

| After | Claim |
| --- | --- |
| Stages 0–1 | An exact reference for the solute field of simulated atom probe data, and a measured account of how close standard delocalisation comes to it, by precipitate size, concentration and detection efficiency. |
| Stage 2 passes | A neural implicit field fitted to one sparse measurement recovers small precipitates better than tuned delocalisation. |
| Stage 3 passes | A network trained on simulations reconstructs unseen measurements closer to the truth than any single-measurement method, under a proper scoring rule, with its failure modes measured out of distribution. |
| Stage 4 passes | Reconstructed fields identify precipitates, and measure their sizes, better than the maximum separation method. |
| Stage 5 passes | A physics-informed network recovers solute depletion profiles and physical constants from sparse, noisy data, and rejects its physics when the data violate it. |

A negative result is also a result: each stage's failure branch records what was
learned.

The project will not claim:

- that a network invents the missing atoms;
- that it respects thermodynamics, before Stage 5 passes;
- that it is equivalent to neural radiance fields;
- that reproducing spatial statistics validates a reconstruction.

---

## 6. Risks and limitations

- **There may be little room to improve.** For large precipitates at realistic
  efficiencies, smoothing is already close to the truth. The project is designed to
  measure that rather than hide it: criteria are judged where improvement is
  possible.
- **Learned expectations can mislead.** A network trained on simulations may impose
  simulated shapes on data that do not have them. The lattice and
  out-of-distribution tests measure how much.
- **A simulation is not a specimen.** The simulator has uniform atom density,
  spherical precipitates, no trajectory aberrations and no mass-spectrum overlaps,
  and Stage 5's diffusion physics is a modelling choice. Every result goes from
  simulation to simulation. Carrying the methods to real data would take separate
  work on model misspecification, which the physics acceptance test only begins to
  address.
- **The oracle is conditional on geometry.** It knows the true precipitate positions
  but not the recorded identities, so for the very smallest precipitates a method
  could in principle do marginally better than it. At the sizes simulated, the
  effect is negligible.
- **Averages hide precipitates.** Most atoms are matrix atoms, so every result is
  reported by region as well as overall.

---

## 7. Relationship to the parameter-inference work

The two lines of work share a simulator, datasets, statistics code and a way of
making decisions. Three things pass between them.

- **Stage 4's comparison with standard practice** is the comparison against
  current practice that the inference work lists as its most valuable unfinished
  item.
- **The learned summary of Section 3.9** is proposed as inference work.
- **Two findings of this project concern the inference work's dataset:** its
  precipitate centres sit on a lattice, and 22 of its specimens lost precipitates
  to a lattice defect. Neither is yet recorded in the inference roadmap.

---

## 8. Deliverables

- An exact oracle for simulated atom probe data, tested against independent
  references (done).
- A frozen benchmark: the random-centre dataset, splits, observations and
  checksums (done).
- Modules in the package library for baselines, neural fields, the conditional
  network, precipitate extraction and the physics-informed network, each tested
  against closed forms where one exists.
- A script and a results record for every stage.
- Renderings, figures and walkthroughs in the repository's paper-style format.
- The roadmap, kept as the authoritative record of every criterion, result and
  correction.

---

## References

- Bennett, R. A., Proudian, A. P., & Zimmerman, J. D. (2023). Cluster
  characterization in atom probe tomography: Machine learning using multiple
  summary functions. *Ultramicroscopy* 247, 113687.
- Blau, Y., & Michaeli, T. (2018). The perception–distortion tradeoff. *CVPR*.
- Chibane, J., Alldieck, T., & Pons-Moll, G. (2020). Implicit functions in feature
  space for 3D shape reconstruction and completion. *CVPR*.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm
  for discovering clusters in large spatial databases with noise. *KDD*.
- Hellman, O. C., Vandenbroucke, J. A., Rüsing, J., Isheim, D., & Seidman, D. N.
  (2000). Analysis of three-dimensional atom-probe data by the proximity histogram.
  *Microscopy and Microanalysis* 6, 437–444.
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang,
  L. (2021). Physics-informed machine learning. *Nature Reviews Physics* 3, 422–440.
- Peng, S., Niemeyer, M., Mescheder, L., Pollefeys, M., & Geiger, A. (2020).
  Convolutional occupancy networks. *ECCV*.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural
  networks. *Journal of Computational Physics* 378, 686–707.
- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G.
  (2020). Implicit neural representations with periodic activation functions.
  *NeurIPS*.
- Tancik, M., et al. (2020). Fourier features let networks learn high frequency
  functions in low dimensional domains. *NeurIPS*.
- Yeong, C. L. Y., & Torquato, S. (1998). Reconstructing random media. *Physical
  Review E* 57, 495.

*The authoritative, continuously updated record of criteria, results and
corrections is `reconstruction/ROADMAP.md`. Where this proposal and the roadmap
differ, the roadmap governs.*
