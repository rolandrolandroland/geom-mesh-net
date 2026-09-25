# E19 — A Fourier-feature field fitted to one pattern: Gate 2

*Reconstruction Stage 2. Fitted only to the atoms a detector records in one pattern, does a
coordinate network with Fourier features estimate the guest probability better than tuned
kernel smoothing?*

[← back to README_detailed](../../README_detailed.md#7-experiment-walkthroughs) ·
Protocol: [`experiments/reconstruction/ROADMAP.md`](../../experiments/reconstruction/ROADMAP.md), Stage 2, its design and pilot records ·
Contract: [`stage2_design.json`](../../experiments/reconstruction/stage2_design.json) (frozen, hash `f9a0085f62490e83`) ·
Implemented by [`neural/implicit.py`](../../geom_mesh_net/neural/implicit.py),
[`stage2_field.py`](../../experiments/reconstruction/stage2_field.py) and
[`pilot/check_fourier_field.py`](../../experiments/reconstruction/pilot/check_fourier_field.py)

<!-- RESULTS: abstract, runtime line, sections 4 onward, figures -->

---

## 1. Introduction

### 1.1 The question

Stage 1 measured how close standard smoothing gets to the truth: B1, one Gaussian bandwidth,
and B2, a bandwidth that narrows where guests are dense, both chosen by cross-validation. In 48%
of test cells smoothing left at least 15% of the gap to the oracle open, most of it at cluster
rims and cores. Stage 2 asks whether a network fitted to the same observed atoms, and nothing
else, closes that gap.

The network is a coordinate MLP: position in, guest probability out. A plain MLP on raw
coordinates learns low frequencies first and fine structure slowly — spectral bias — so the
coordinates are first mapped to random Fourier features (Tancik et al., 2020). Two controls
separate what that encoding does from what the network does, so a result can say more than
"it won" or "it lost".

### 1.2 What was fixed before the test

Everything the experiment could tune was settled on development patterns and frozen in
`stage2_design.json` before a single test cell was fitted; the harness refuses test cells from
a design whose status is not `frozen`, and hashes the design into every result. Within a cell,
only two things are chosen, both from that cell's own validation atoms: the frequency scale σ
and the stopping checkpoint. Removed atoms are read only after every choice is made.

---

## 2. Methods

### 2.1 Data and cells

The 36 test patterns of the frozen benchmark (six per `cr` band × `rho_c` band) at detection
efficiencies 0.1, 0.37 and 0.8: 108 cells. Each pattern holds 216,000 atoms in a 60-unit box.
The observed atoms of a cell are those its frozen thinning mask keeps; the rest are removed and
scored. Design decisions were made on 36 development cells (the Stage 1 pilot's twelve
patterns); a second, held-back set of twelve development patterns ran once, to validate the
frozen harness.

### 2.2 The data boundary

Each cell's dataset file and thinning mask are checked against their frozen checksums. The
observed atoms are then split 80 / 20 into fitting and validation atoms by a dedicated random
stream, before anything reads a label, and the split is hashed. `prepare_cell` returns the
observed atoms and, as a separate object, the removed labels, the oracle and the regions
derived from the truth; no fitting routine accepts the second. Separate recorded streams draw
the split, the Fourier frequencies, the initial weights, the minibatch order and the
predictive-check relabelling.

### 2.3 The model

For a position x, with u = x / 60 scaled by the known box length:

    γ(u) = [sin 2πBu, cos 2πBu],   B = σ·B0,   (B0)ij ~ N(0, 1),   256 frequencies

feeding an MLP of four hidden layers of 256 ReLU units and one output logit; the probability is
its sigmoid. B0 is drawn once per cell and seed, so every σ candidate scales the same
frequencies and starts from the same weights and batch order, and differs only in σ. The model
starts exactly at the constant field of its fitting labels: the output weights are zeroed and
the output bias is their log odds. Update 0 is therefore a candidate checkpoint, and a model
that never beats the constant returns it.

### 2.4 Two controls

| Model | Parameters | What it tests |
| --- | ---: | --- |
| Fourier MLP (the method) | 328,961 | — |
| MLP of the same width and depth on 2x/60 − 1 | 198,657 | whether the Fourier encoding helps |
| linear logit on the same Fourier features | 513 | whether the network helps beyond the representation |

The raw-coordinate control matches width and depth, not capacity. Each control was given its
own learning rate on development data, so an optimisation failure could not pass as a
limitation of the model.

### 2.5 Training and selection

Unweighted binary cross-entropy on the fitting labels, Adam at 1e-4 (linear control 1e-2,
raw-coordinate control 1e-3), batches of 32,768 drawn from successive random permutations of
the fitting atoms, or the whole set when it is smaller. Validation cross-entropy is evaluated
every 25 updates; the best checkpoint is kept, training stops after 200 updates without
improvement or at 2,000, and the best checkpoint is restored. σ is chosen from {0.375, 0.75,
1.5, 3, 6} by the same validation cross-entropy, ties to the smaller σ.

### 2.6 Refit and seed

The selected fit is then restarted on all observed atoms — fitting and validation together —
with the same frequencies, initial weights and batch stream, for exactly the selected number of
updates, starting from the constant of all observed atoms. One seed, seed 0. The same policy
applies to both controls.

### 2.7 The comparators

B0, the observed guest fraction; B1 and B2 by Stage 1's procedure — five-fold cross-validation
on all observed atoms and prediction from all of them — B2 on the grid widened by O12. The
neural models choose on one validation split instead of five folds; that difference is part of
the comparison.

### 2.8 Scores and Gate 2

Stage 1's metrics, on removed atoms: excess log loss over the oracle (the primary score, equal
in expectation to the mean KL divergence from the truth), gap closed, excess by region (rim,
core, matrix, interior), Brier excess, expected calibration error, and the predictive check,
which is reported and not gated. A cell has *headroom* when B1 leaves at least 15% of the
constant-to-oracle gap open and at least 0.01 nats per atom.

Gate 2, as written before Stage 2 began, with its implementation fixed before the test:

- on headroom cells, the Fourier MLP beats the better of B1 and B2 in at least two thirds;
- the median share of B1's remaining gap it closes on headroom cells is at least 20%;
- the median over headroom cells of ECE(field) − ECE(B1) is at most 0.005;
- the median over the other cells of its excess minus B1's is at most 0.002 nats per atom.

Every one of the 108 cells must be present without error, or the verdict is `incomplete`.

---

## 3. The development pilot

Before anything was frozen, `pilot/check_fourier_field.py` ran the method and both controls on
the 36 development cells, with σ and the checkpoint chosen inside each cell exactly as the gate
chooses them. The development cells' removed atoms decided only the global settings. Every
number below is regenerated by `python -m experiments.reconstruction.pilot.check_fourier_field
--report`.

### 3.1 The planned σ grid was wrong

The plan's grid was {3, 6, 12, 24}, placed by matching each σ to the width of the Gaussian
kernel its random features imply, L / (2πσ), against B1's bandwidths. On four contrasting cells
every selection landed at σ = 0.75 or 1.5, below the whole grid; σ ≥ 12 never improved on the
constant. The kernel of the *features* is not the resolution of the *trained network*, which
composes them through four nonlinear layers. The grid became {0.375, 0.75, 1.5, 3, 6}. On large
clusters σ = 0.375 was chosen at the lower edge in 7 of 36 cells, so σ = 0.1875 was tried there:
validation would have selected it once in twelve cells, where it would have been worse.

### 3.2 The network memorises within a few hundred updates

<!-- figure: training curves -->

With 330,000 parameters and as few as 17,000 fitting labels, the network drives its fitting
cross-entropy towards zero while its validation cross-entropy turns upward within tens to
hundreds of updates. Early stopping is not a refinement here but the regulariser. A lower
learning rate did not change how good the best checkpoint was, only when it came, and 1e-4 made
it late enough for 25-update checkpoints to resolve.

### 3.3 Refit, seeds, width and device

- Restarting the selected fit on all observed atoms improved 50 of 54 seed-fits (median −0.0006
  nats per atom). Adopted.
- Across three seeds a cell's excess spread by a median 0.0007; averaging the seeds gained 0.0003
  at three times the cost. One seed.
- Hidden width 128 was worse than 256 in 28 of 36 cells and only 21% faster.
- Training on MPS and on the CPU chose the same σ and checkpoint in every cell compared.
  Gradients with respect to the *input* coordinates, though, are wrong on MPS in torch 2.10 —
  forward passes and weight gradients are not — so anything needing them runs on the CPU.

### 3.4 A weak gradient penalty neither helps nor harms

λ·mean‖∇f‖² on the logit, in physical units, over each batch, for λ ∈ {1e-4, 1e-3, 1e-2} at the
cell's selected σ: the median excess moved by at most 0.00004 nats per atom. The largest λ
worsened rims in 15 of 24 cells — the direction the roadmap predicted — by a median 0.00023,
and validation would have chosen λ = 0 in only 4 of 24 cells. At these strengths the penalty is
1–3% of the loss, so the ablation says only that weak smoothness regularisation is inert.

<!-- RESULTS: sections 4 (test results), 5 (discussion), 6 (corrections), 7 (conclusion), outputs, reproduce -->
