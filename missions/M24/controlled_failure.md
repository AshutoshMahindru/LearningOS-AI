# Controlled failure — omitted branch or wrong activation derivative

## Failure: plausible backward numbers, one local rule wrong

Before running, predict whether a hidden value that feeds two loss heads
should receive **one** contribution or the **sum**, and predict whether a
dead ReLU unit (`z < 0`) should pass an upstream gradient.

The defective path uses one named change:

- `omitted_branch`: accumulate only one downstream path into the shared
  node, or
- `wrong_relu_derivative`: treat `relu'` as identity, including on
  non-positive pre-activations.

The forward graph, intended `X`, intended weights, intended biases,
softmax-NLL, and all other local derivatives stay fixed. Only the named
rule changes. The defect can still emit finite, even “gradient-looking,”
numbers. That is the point.

Diagnosis comes from central finite differences on the **true** loss,
parameter by parameter. Isolate the first mismatch. Do not open
`torch.autograd` or a training loop.

## Discriminators

Omitted branch: `dL/dW2` still matches finite differences because that
local map never used the dropped path. `dL/dH` (and therefore `dL/dW1`,
`dL/db1`) disagrees by exactly the missing contribution.

Wrong ReLU derivative: a unit with `z < 0` is a clean discriminator —
the true loss does not move when that pre-activation is nudged, so the
finite difference is ~0 while the buggy analytic gradient is not. A unit
sitting **exactly** at `z = 0` is a hinge; skip it as a check, not as a
repair.

## Repair rule

The smallest repair restores addition at the branch and `relu'(z) = 1`
only when `z > 0`. Do not change the teaching weights, do not add an
optimizer, and do not replace reverse mode with a framework.

Submit prediction, named defect, preserved invariants, first mismatch,
root cause, smallest repair, and the repaired rerun that matches
finite differences on smooth parameters.

A repair is rejected if it opens M25-M26 mechanisms or changes several
variables at once.
