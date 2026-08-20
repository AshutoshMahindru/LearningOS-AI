# M30 — Dissect a Transformer Block

M29 already traces one attention head. The useful whole here is a
**tiny transformer block** on a deterministic 4-D sequence:

```
x
  |- LN -> multi-head attention -> W_O
  +-------------------------------- attn residual
       |- LN -> position-wise FFN
       +---------------------- FFN residual / output
```

That diagram is the **declared teaching convention: pre-norm**. Post-norm
is a labeled alternative, not a silent default and not a universal law.

Heads are parallel learned projections. They do not come with human
job titles. Residual connections are elementwise adds of matching
shapes. The feed-forward sublayer is the same MLP at every position.

This mission uses NumPy and the trusted M29 attention call. Nothing is
downloaded. LLM training (next-token loss, contamination) stays closed
until M31. Inference and adaptation stay closed until M32.

Canonical sources: `hf-llm-course` and `karpathy-zero-to-hero` via
`data/source_registry.json`.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
