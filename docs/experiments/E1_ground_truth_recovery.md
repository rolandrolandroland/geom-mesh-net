# E1 — Ground-truth recovery

*Recovering two simulator parameters that were varied but never written to disk.*

[← back to README_detailed](../../README_detailed.md#7-experiment-walkthroughs) ·
Implemented by [`inference/recover_ground_truth.py`](../../inference/recover_ground_truth.py) ·
Runtime 6 s

---

## Abstract

The cluster simulator varies four parameters but writes only two of them to
`data/pattern_stats.npy`. The mean cluster radius `cr` and the radius spread `rb`
were passed to the simulator and then discarded. Without them no supervised
inference over the full parameter vector is possible.

Both are recoverable because the generator is seeded and draws in a fixed order.
Replaying that sequence reproduces the two *saved* parameters to exactly zero
difference across all 1,000 patterns, which certifies the two unsaved ones from
the same replay. The recovery was then confirmed independently by repairing the
generator and regenerating patterns from scratch: all four parameters matched to
zero difference.

The recovered values are written to disk with provenance so that nothing
downstream depends on the replay again — a guarantee that would fail silently if
the generator were edited.

---

## 1. Introduction

`data_factory.py` draws four parameters per simulation and passes them to
`clustersim`. It then records a nine-column statistics table containing measured
and true values for the overall solute fraction `pcp`, the in-cluster
concentration `rho_c`, and the matrix concentration `rho_b`.

`cr` and `rb` appear nowhere in that table. They were drawn, used, and dropped.

This is not merely inconvenient. Supervised inference needs the parameter vector
for every training pattern; two of the four being missing would reduce the
problem to `rho_c` and `rho_b`, and would silently exclude the cluster length
scale — which E5 later shows is the parameter the *K* statistics exist to carry.

---

## 2. Methods

### 2.1 The replay

The generator opens with

```python
rng = np.random.default_rng(42)
rho_c_vec = rng.uniform(low=pcp*2, high=1,       size=n_sims)
rho_b_vec = rng.uniform(low=0,     high=pcp*0.5, size=n_sims)
cr_vec    = rng.uniform(low=3,     high=15,      size=n_sims)
rb_vec    = rng.uniform(low=0,     high=0.5,     size=n_sims)
```

A seeded generator consuming draws in a fixed order is deterministic. Replaying
the same four calls in the same order reproduces all four vectors exactly — not
approximately, since these are the same float64 draws from the same generator
state.

### 2.2 The verification

The replay is checked against the two parameters that *were* saved. If the
replayed `rho_c` and `rho_b` match the recorded columns bit for bit, then the
generator state evolved identically, and `cr` and `rb` from the same replay are
equally exact.

The gate is **exact equality**, not approximate. Any nonzero difference means the
draw sequence has changed and the replay is void.

### 2.3 The independent confirmation

The replay argument is circular in one respect: it verifies against columns that
the same script wrote. A stronger check became available once the generator was
repaired (§8.5 of the roadmap — it had been unable to run at all since the package
restructure, due to a stale import).

Three patterns were regenerated from scratch with the repaired generator, now
writing all four parameters. Their saved values were compared against the replay.

### 2.4 Descriptors

The script also records per-pattern realised quantities needed downstream:
cluster count, realised radius mean and standard deviation, realised guest
fraction, and a degeneracy flag for patterns containing no clusters. These are
read from the `.npz` archives without loading the coordinate arrays, since
`.npz` members decompress individually.

---

## 3. Results

### 3.1 The gate

**Table 1.** Replay verified against the saved columns, all 1,000 patterns.

| Parameter | Saved? | Max absolute difference | Verdict |
| --- | --- | ---: | --- |
| `rho_c` | yes | **0.0** | exact |
| `rho_b` | yes | **0.0** | exact |
| `pcp` | yes, constant | — | 0.1 for every pattern; not a parameter |
| `cr` | **no** | — | certified by the above |
| `rb` | **no** | — | certified by the above |

### 3.2 The independent confirmation

**Table 2.** Three patterns regenerated from scratch with the repaired generator,
compared against the replay.

| Parameter | Column | Max absolute difference |
| --- | ---: | ---: |
| `rho_c` | 4 | 0.000e+00 |
| `rho_b` | 7 | 0.000e+00 |
| `cr` | 9 | **0.000e+00** |
| `rb` | 10 | **0.000e+00** |

This is the stronger result. `cr` and `rb` are confirmed against values produced
by actually rerunning the simulator, not merely against the two columns the old
script happened to save.

### 3.3 Recovered priors

**Table 3.** The recovered parameter vector, 1,000 patterns.

| Parameter | Recovered range | Declared prior |
| --- | --- | --- |
| `rho_c` | 0.201 – 0.999 | U(0.2, 1.0) |
| `rho_b` | 0.000 – 0.050 | U(0.0, 0.05) |
| `cr` | 3.012 – 14.997 | U(3.0, 15.0) |
| `rb` | 0.000 – 0.500 | U(0.0, 0.5) |

No value lies exactly on a prior bound, which matters for E3: the flow works in a
logit-transformed space that diverges at the bounds.

### 3.4 Dataset descriptors

**Table 4.** Realised quantities across all 1,000 patterns.

| Quantity | Value |
| --- | --- |
| Patterns | 1,000 |
| Points per pattern | 216,000 |
| Domain | 60 × 60 × 60 |
| Cluster count | min 0, median 8, max 735 |
| Realised guest fraction | 0.0967 ± 0.0237 (target `pcp` 0.1) |
| **Zero-cluster patterns** | **2** |

### 3.5 Identifiability expectations, registered in advance

Two relationships were measured before any model was fitted, so that Stage 3's
outcome could be judged against a prediction rather than rationalised afterwards.

| Relationship | Measured | Prediction |
| --- | ---: | --- |
| corr(`cr`, log cluster count) | **−0.906** | `cr` should be sharply identified |
| corr(`rb`, realised mean radius) | **+0.003** | `rb` should return near its prior |

`rb` governs the *spread* of cluster radii. With a median of eight clusters per
pattern there is almost no sample from which to estimate a spread, so its
posterior was predicted to stay close to its prior.

---

## 4. Discussion

### 4.1 The guarantee is fragile, so it was retired

The replay depends on `data_factory.py` keeping the number, order and
distributions of its four `rng.uniform` calls unchanged. Any edit — inserting a
draw, reordering two, changing a bound — silently produces different values with
no error raised.

That is why Stage 0 persists θ to disk with provenance rather than recomputing it
on demand. Everything downstream reads `inference/ground_truth/theta.npy`. The
replay is performed once, verified, and never relied upon again.

A warning to that effect is recorded in the provenance file, and the generator
now carries a comment at the draw site.

### 4.2 `pcp` is not a parameter

`pcp` is recorded as a "true" value in the statistics table, which invites
treating it as inferable. It is held constant at 0.1 across every simulation and
therefore carries no variation; nothing can be learned about it. It is excluded
from θ.

This is a real limitation rather than a technicality. In a physical measurement
the overall solute fraction is a primary unknown, so the inference problem as
posed is easier than the real one. See §5.2 of the main document.

### 4.3 Two degenerate patterns

Two of 1,000 patterns contain zero clusters, despite parameters specifying large
ones. These are legitimate draws from the prior predictive — the simulator placed
cluster centres and none survived the domain cut — not a bug.

They are flagged rather than dropped. Dropping them would bias the training set
toward large-`cr` patterns that happened to produce clusters. One of them,
pattern 613, later dominated a mean log-likelihood by itself (E3 §4.3) and forced
a robustness reckoning that surfaced a defect in a diagnostic (E7 §5.2).

---

## 5. Conclusion

Both unsaved parameters were recovered exactly and confirmed by independent
regeneration. The dataset is now a complete set of 1,000 (θ, x) pairs with an
exactly known prior — the input that every subsequent experiment requires.

The two most consequential incidental findings were that the data generator could
not run at all, and that two patterns are structurally degenerate. Both mattered
later.

---

## Outputs

| File | Contents |
| --- | --- |
| `inference/ground_truth/theta.npy` | (1000, 4) parameter matrix |
| `inference/ground_truth/descriptors.npz` | per-pattern realised quantities |
| `inference/ground_truth/provenance.json` | priors, verification, fragility warning |

## Reproduce

```bash
PYTHONPATH=. python inference/recover_ground_truth.py
```

Tests: `tests/test_ground_truth.py` — 13 tests covering replay determinism, prior
containment, draw-order sensitivity, and the registered identifiability
predictions.
