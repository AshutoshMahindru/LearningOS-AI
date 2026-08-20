# M25 — Train Networks in PyTorch

M24 assigned blame with reverse-mode gradients on the named graph

`x → hidden_preactivation → hidden_activation → logits → probabilities → loss`.

M25 asks **how a modern training loop uses those gradients**. The useful
whole is one CPU `nn.Module` plus

`zero_grad → forward → loss → backward → step`

with train/eval mode, no-grad inference, protected splits, a checkpoint,
and a declared reproducibility policy.

This mission does **not** catalogue multi-cause deep-learning failures
(M26). It provides an instrumented loop M26 can break independently.

Canonical sources: `pytorch-basics`, `fastai-course`, and
`karpathy-micrograd` via `data/source_registry.json`. Recheck official
PyTorch training-loop docs at implementation time; prefer those over
blog reproductions.

Implementation status is not learner completion. Predictions, no-AI work,
ADR decisions, and competence remain intentionally unfilled.
