# M23 — Implement a Forward Pass with NumPy

M22 opened one neuron and one dense layer as arithmetic. M23 reconstructs
**inference** from those equations: two affine maps, a hidden activation,
named intermediates, batches, logits, and stable class-axis softmax.

The useful whole is not a training loop. It is:

`X → Z1 = X @ W1 + b1 → H = activation(Z1) → logits = H @ W2 + b2 → probabilities = softmax(logits)`

with row-batch orientation inherited from M16/M22: `X` is `(batch, n_in)`,
`W1` is `(n_in, n_hidden)`, `W2` is `(n_hidden, n_classes)`.

This mission does **not** assign blame with backpropagation (M24) or
author a PyTorch training loop (M25). Changing one weight to see which
outputs move is an inference probe, not a gradient.

Canonical sources: `numpy-quickstart` and `3b1b-neural-networks` via
`data/source_registry.json`. Prefer those over autograd material.

Implementation status is not learner completion. Predictions, no-AI work,
ADR decisions, and competence remain intentionally unfilled.
