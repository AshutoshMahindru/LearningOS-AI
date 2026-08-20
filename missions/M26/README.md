# M26 — Diagnose Deep Learning Failure

M25 handed over an instrumented CPU training loop:

`zero_grad → forward → loss → backward → step`

with protected splits, eval/no-grad, and checkpoints. M26 asks **why a
run that still emits numbers is the wrong run**. The useful whole is the
M25 trace first, then one fault at a time across data, optimization,
gradient flow, capacity, regularization, and evaluation.

This mission does **not** open language-model internals. It provides
P5 a reusable protocol:

symptom → competing hypotheses → cheapest discriminating experiment →
root cause → smallest repair → regression.

Canonical sources: `pytorch-basics`, `fastai-course`, and
`karpathy-micrograd` via `data/source_registry.json`. Recheck official
PyTorch training and debugging notes at implementation time; prefer
those over blog reproductions.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
