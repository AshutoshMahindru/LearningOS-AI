# Controlled failure — residual branch or norm boundary

## Failure: tensors look plausible, a skip or a norm is in the wrong place

Use the cash teaching sequence with the declared **pre-norm** block.
Predict, before running, which named checkpoint should still match a
golden pre-norm trace if only the residual addend is wrong, and which
checkpoint should move first if the first LayerNorm is skipped.

Then run one named defect.

The defective path uses one named change:

- `residual_wrong_branch`: add the attention output to `LN(x)` rather
  than to the original residual stream `x`, or
- `norm_wrong_boundary`: skip the pre-attention LayerNorm while still
  labeling the run `pre_norm`.

`x`, parameters, mask, and scale stay fixed. Only the residual addend
or the first-norm boundary changes.

The defect can still emit finite, even "block-looking," tensors. That
is the point. Diagnosis comes from:

1. named-checkpoint parity (`first_divergence`),
2. the residual identity `stream + sublayer` recomputed with numpy `+`,
3. whether `attn_norm` still equals `LN(x)`,
4. the hand residual `(1, 0, 2) + (0, 1, -1) = (1, 1, 1)`.

## Discriminators

Wrong residual branch: `q`, `head_outputs`, and `attn_projected` still
match golden; `attn_add` equals `attn_norm + attn_projected` and is the
first divergence.

Wrong norm boundary: `attn_norm` equals `x` rather than `LN(x)` and is
the first divergence; later attention checkpoints move because the
M29 call saw a different input.

Labeled post-norm is **not** this defect. Post-norm is an explicitly
named convention that also LayerNorms after each residual. Compare
`declared_convention` plus whether `attn_residual` was LayerNormed.

## Repair rule

The smallest repair calls `repair_block` on the **broken trace**
(same `x`, parameters, mask, declared convention) so the pre-norm
identities are restored. Do not change the teaching vectors, do not
add a training loop, and do not retitle heads.

Submit prediction, named defect, preserved checkpoints, first
divergence, root cause, smallest repair, verification, and the
regression that the broken path still diverges.

A repair is rejected if it opens M31-M32 mechanisms, if it is two
unrelated `defect="none"` runs, or if it changes several variables at
once.
