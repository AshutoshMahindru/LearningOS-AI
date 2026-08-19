# M22 — Open the Neuron and Layer

M21 trained a complete neural estimator as a black box. M22 opens **one
hidden unit and one dense layer** as arithmetic: weighted sum, bias,
activation, shapes, and batches.

The useful whole is not a training loop. It is:

`input features → z = w·x + b → activation(z) → neuron output`

lifted to a row-batch layer `Y = activation(X @ W + b)` with `W` shaped
`(n_in, n_out)`.

This mission does **not** implement a multi-layer NumPy inference stack
(M23), backpropagation (M24), or a PyTorch training loop (M25). The
`relu` string that M21 froze as a sklearn knob is now an observed
function, not a reason to open autograd.

Canonical sources: `3b1b-neural-networks` and `fastai-course` via
`data/source_registry.json`. Prefer those over later PyTorch or micrograd
material.

Implementation status is not learner completion. Predictions, no-AI work,
ADR decisions, and competence remain intentionally unfilled.
