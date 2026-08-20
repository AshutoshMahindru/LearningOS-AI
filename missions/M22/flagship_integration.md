# V05 integration — open the layer after the black box

## M21 → M22 boundary

M21 hands over an accepted black-box training run: split, scaler,
hidden-unit count as a **capacity knob**, optimizer settings, and
held-out evidence. M22 explains what one of those hidden units *is*
without claiming that ReLU caused M21's accuracy.

The observable V05 layer after M22 is:

`neuron: z = w·x + b → y = activation(z)`
`layer:  Y = activation(X @ W + b)` with `W` shaped `(n_in, n_out)`

## What M22 must not change

M22 does not retrain the M21 estimator, does not inspect sklearn
`coefs_`, and does not rewrite M21 held-out metrics to make a neuron
story look successful. M21 capacity comparisons stay observations.

## M22 → M23 handoff

M23 may reconstruct a multi-layer NumPy forward pass only after the
learner can defend:

- the affine-then-activate order
- row-batch shapes
- bias as a translation
- why stacked affine maps collapse
- a repaired orientation/boundary defect
- trusted hand-computed reference outputs

M23 implements inference composition. It must not invent a different
layer convention without stating the change.
