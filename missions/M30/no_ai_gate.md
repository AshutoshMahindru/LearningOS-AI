# No-AI gate — defend one transformer block from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers below.

## Fixture (fresh)

Declared convention: **pre-norm**. Activation: ReLU.

```
B=2, T=5, d_model=8, n_heads=4, d_head=2, d_ff=16
```

Residual micro-case: `x = (2.0, -1.0, 0.5)`, sublayer `= (-0.5, 1.5, 0.0)`.

LayerNorm micro-case over the last axis, `gamma=1`, `beta=0`, ignore
`eps` (treat it as 0): `v = (0.0, 4.0)`.

## Part A: shapes

Annotate shapes for `x`, `q_heads`, `head_outputs`, `attn_concat`,
`attn_projected`, `ffn_hidden`, and `output`.

## Part B: residual and LayerNorm by hand

1. Compute `x + sublayer`.
2. Compute mean, variance, and normalized `v`.
3. State the pre-norm residual identity in words.

## Part C: wrong placement

A block is labeled pre-norm. After attention, the residual-stream
checkpoint equals `LN(x) + attn_projected`. Which identity failed?

## Part D: what lives inside from M29

List the M29 pieces that still run inside each head. State, in one or
two sentences, why heads are not assigned human job titles.

Pass requires a block diagram from memory, shape annotations, residual
and norm arithmetic, a placement diagnosis, and an oral defense.
Leave all learner responses unfilled in the repository.
