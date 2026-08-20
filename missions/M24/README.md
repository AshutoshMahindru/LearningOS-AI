# M24 — Assign Blame with Backpropagation

M19 showed that a parameter moves downhill when we subtract learning-rate
times the gradient. M23 named the forward graph

`x → hidden_preactivation → hidden_activation → logits → probabilities`.

M24 asks **which stored value is to blame for the loss**. The useful whole
is reverse accumulation of local sensitivities: chain rule on a scalar
graph, addition at a branch, a tiny dense-layer/ReLU backward pass, a
central finite-difference check, and **one** declared update.

This mission does **not** implement framework autograd or a training loop
(M25). One small update is allowed as local evidence, not as convergence.

Canonical sources: `3b1b-neural-networks`, `3b1b-calculus`, and
`karpathy-micrograd` via `data/source_registry.json`. Prefer those over
PyTorch autograd material.

Implementation status is not learner completion. Predictions, no-AI work,
ADR decisions, and competence remain intentionally unfilled.
