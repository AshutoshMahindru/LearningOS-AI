# V06 integration — freeze one teaching transformer block

## M29 → M30 boundary

M29 already mixes values with softmax over keys. M30 is the first V06
mission that **composes that head into a block**: several heads in
parallel, an output projection, residual adds, a declared norm
placement, and a position-wise feed-forward.

The observable V06 surface after M30 is:

```
X (B, T, d_model)
  -> LN (pre-norm)
  -> split heads -> M29 attention -> merge -> W_O
  -> residual add with X
  -> LN
  -> FFN
  -> residual add
  -> Y (B, T, d_model)
```

with named checkpoints and an explicit convention label.

## What M30 must not change

M30 does not train an LLM, does not shift next-token targets, does not
decode, and does not choose adaptation routes. Those are M31 and M32
measurements on top of this block.

M30 does not rename heads as "syntax" or "anaphora."

## M30 → M31 handoff

M31 may attach a causal next-token objective to stacks of this block.
It must not relabel residual adds as a training trick without a
training experiment, and it must not silently change the declared
norm convention.

M31 receives:

- TRACE_CHECKPOINTS through one block
- declared teaching convention `pre_norm` plus labeled `post_norm`
- residual identity `stream + sublayer`
- head split/merge shapes and the no-mythology banner
- version `v06-teaching-block-1`

Reusable artifacts: `missions/M30/transformer_block.py` checkpoints
and the cash/water 4-D teaching sequences that wrap M29's 2-D subspace.
