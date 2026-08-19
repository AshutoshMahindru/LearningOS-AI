# Controlled failure — layer orientation or activation boundary

## Failure: plausible numbers, wrong boundary

Before running, predict the outputs of a small square layer from a
hand-computed `X @ W + b` then ReLU.

The defective path uses one named change:

- `transposed_weights`: multiply by `W.T` instead of `W`, or
- `activation_before_affine`: apply the activation to `X` before the affine map.

The code path, intended `X`, intended `W`, intended `b`, and activation
**name** stay fixed. Only the orientation or the activation boundary
changes.

The defect can still emit finite numbers. That is the point. Diagnosis
comes from the hand-computed micro-case and from the declared shape
contract `(batch, n_in) @ (n_in, n_out)`, not from opening a training
loop.

## Repair rule

The smallest repair restores `Y = activation(X @ W + b)` with `W` shaped
`(n_in, n_out)`. Do not change the teaching activation, do not add a
second layer, and do not introduce gradients.

Submit prediction, named defect, preserved invariants, observed outputs,
root cause, smallest repair, and the repaired rerun that matches the
hand calculation.

A repair is rejected if it opens M23-M25 mechanisms or changes several
variables at once.
