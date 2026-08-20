# Controlled failure — Chaos Day, category hidden

## Failure: one catalogue defect, symptoms only

Before running, write competing hypotheses across data, optimization,
gradient flow, architecture, and evaluation. Do **not** name a cause
from the seed, the filename, or a guess about the rotation.

Chaos Day selects one hidden defect from the catalogue (labels,
learning rate, blocked gradient path, capacity, or evaluation leakage).
The public report omits defect, category, learning rate, width, and
Dropout. The loop can still emit finite losses. That is the point.

Passing requires:

1. a ranked hypothesis list written before extra inspection
2. one discriminating experiment and an updated ranking
3. root-cause proof that a competing family cannot explain the same
   traces
4. the smallest repair on the prepared objects
5. regression evidence that the original symptom is gone

## Discriminators (use after the prediction, not instead of it)

- labels: fixture-block agreement on train rows; val/held-out labels
  stay honest
- scaling: one feature column's standard deviation dominates
- learning rate high: loss oscillates or explodes at fixed architecture
- learning rate low: almost no train-loss movement; tiny-subset overfit
  succeeds once the rate is restored
- blocked path: `fc1` does not move and `requires_grad` is false while
  `fc2` still trains
- capacity: full-set underfit while a restored width can learn
- evaluation leakage: claimed validation `n` matches train, not val
- train-mode eval: `model.training` is true on a frozen checkpoint

## Repair rule

Repair the broken object. Do not compare two independent healthy runs
and call that a fix. Do not change several knobs. Do not open a later
mission's mechanisms.

Submit prediction, hypothesis ranking, discriminator, root cause,
smallest repair, verification, and regression evidence.

A repair is rejected if it retrains to dodge an evaluation defect, or
if it redesigns the net to dodge a label or learning-rate fault.
