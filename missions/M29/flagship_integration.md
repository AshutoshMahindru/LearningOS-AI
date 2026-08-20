# V06 integration — freeze the single-head attention trace

## M16 / M28 → M29 boundary

M16 already maps batches with `X @ W` and treats transpose as a
convention repair. M28 already ranks frozen sentence vectors. M29 is
the first V06 mission that **lets a position's representation depend
on other positions**.

The observable V06 surface after M29 is:

`sequence X (T, d_model) → Q, K, V → scores (T_q, T_k) →
mask-before-softmax → weights over keys → output (T_q, d_v)`

with named checkpoints and explicit shape/row-sum/mask invariants.

## What M29 must not change

M29 does not split heads, add a residual, place LayerNorm, or run a
position-wise FFN. It does not train an LLM and does not expose a
search service. Those are M30, M31, and M33 measurements on top of
this trace.

## M29 → M30 handoff

M30 may compose this single head into a block. It must not relabel
attention weights as multi-head semantics or as a residual path.

M30 receives:

- Q, K, V, raw scores, scaled scores, mask, weights, output
- scale `1/sqrt(d_k)` as the teaching default
- softmax over keys
- causal/padding masks as additive fills applied before softmax
- the interpretation limit: weights are not intent

Reusable artifacts: `missions/M29/attention_core.py` checkpoints and
the cash/water teaching sequences.
