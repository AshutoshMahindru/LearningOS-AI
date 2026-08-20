# Controlled failure — wrong-axis softmax or omitted hidden activation

## Failure: plausible probabilities, wrong boundary

Before running, predict the teaching-fixture probabilities from the
hand-computed logits, including the uniform first row.

The defective path uses one named change:

- `softmax_axis_batch`: apply softmax along the batch axis, or
- `omitted_hidden_activation`: skip ReLU on the hidden pre-activation.

The code path, intended `X`, intended weights, intended biases, and
hidden activation **name** stay fixed. Only the softmax axis or the
hidden-activation boundary changes.

The defect can still emit finite, even “probability-looking,” numbers.
That is the point. Diagnosis comes from:

1. one-example versus multi-example parity, and
2. named-intermediate checks against the M22 first-layer reference.

Do not open a loss, a gradient, or a training loop.

## Discriminators

Wrong-axis softmax: a singleton `(1, n_classes)` run no longer matches
the corresponding batch row, because a one-row batch softmaxed on axis 0
is a vector of ones. Batch column sums become 1; row sums generally do
not.

Omitted hidden activation: singleton and batch still agree with each
other, but example 0's hidden activation is no longer ReLU of
`(0, -0.5)`. Intermediate mismatch localizes the defect before the
probabilities.

## Repair rule

The smallest repair restores hidden activation after the first affine
map and softmax along the class axis (`axis=-1`). Do not change the
teaching weights, do not add a third layer, and do not introduce
gradients.

Submit prediction, named defect, preserved invariants, observed
outputs, root cause, smallest repair, and the repaired rerun that
matches the hand-computed graph.

A repair is rejected if it opens M24-M25 mechanisms or changes several
variables at once.
