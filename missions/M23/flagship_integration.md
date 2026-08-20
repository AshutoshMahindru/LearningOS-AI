# V05 integration — reconstruct inference after the layer is opened

## M22 → M23 boundary

M22 hands over explicit layer equations, activation-after-affine
placement, row-batch shapes, and trusted hand-computed first-layer
outputs. M23 composes those layers into a forward pass the learner can
inspect node by node.

The observable V05 inference graph after M23 is:

```
x
  -- W1, b1 --> hidden_preactivation
  -- ReLU   --> hidden_activation
  -- W2, b2 --> logits
  -- softmax (class axis) --> probabilities
```

## What M23 must not change

M23 does not retrain the M21 estimator, does not inspect sklearn
`coefs_`, and does not rewrite M22's layer convention. If a later
mission needs `(n_out, n_in)` weights, it must say so. M23 keeps
`X @ W` with `W` shaped `(n_in, n_out)`.

## M23 → M24 handoff

M24 may assign blame with backpropagation only after the learner can
defend:

- named intermediates for every node in the graph
- class-axis softmax versus batch-axis softmax
- scalar-to-vectorized parity
- M22 dense-layer composition as a trusted forward-parity test
- a repaired wrong-axis or omitted-activation defect

M24 implements credit assignment. It must not invent a different
forward graph without stating the change. M23 does not compute
parameter gradients.
