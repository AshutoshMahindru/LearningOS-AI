# Controlled failure — softmax axis or mask order

## Failure: weights look like probabilities, the axis or order is wrong

Use the cash teaching sequence with identity projections. Predict,
before running, what the bank row of a **correct** attention matrix
must satisfy (row sum, nonnegativity). Then run one named defect.

The defective path uses one named change:

- `softmax_over_queries`: softmax over the query axis rather than keys, or
- `mask_after_softmax`: multiply by a 0/1 causal keep-mask after softmax
  without renormalizing.

Q, K, V, scale, and the intended mask stay fixed. Only the softmax
axis or the mask/softmax order changes.

The defect can still emit finite, even “probability-looking,” numbers.
That is the point. Diagnosis comes from:

1. row-sum invariant (each query distributes mass over keys),
2. a singleton query versus the corresponding batch row,
3. masked-key mass after a causal mask,
4. the hand-computed two-key case `scores = (1, 0)`.

## Discriminators

Wrong-axis softmax: key-columns sum to 1; query-rows generally do not.
A one-query matrix softmaxed on the query axis collapses.

Mask after softmax: forbidden positions can be zeroed, but surviving
weights no longer sum to 1, so the output is not a convex combination
of the allowed values.

## Repair rule

The smallest repair calls `repair_attention` on the **broken trace**
(same Q, K, V, mask, scale) so softmax runs over keys and the mask is
applied before softmax. Do not change the teaching vectors, do not add
heads, and do not introduce a residual or LayerNorm.

Submit prediction, named defect, preserved invariants, observed
weights, root cause, smallest repair, verification, and the regression
that the broken path still fails the invariant.

A repair is rejected if it opens M30-M33 mechanisms, if it is two
unrelated happy-path runs, or if it changes several variables at once.
