# M29 — Understand Attention Through Context

A stored vector from M28 does not change when its neighbors change. The
useful whole here is a **tiny single-head attention pass**:

`X → Q, K, V → scaled dots → mask → softmax over keys → weighted values`

The same middle token (`bank`) sits in two three-token sequences:

- `river bank cash`
- `river bank water`

Only the third representation changes. Attention weights and the **output
vector at `bank`** move. That is context-dependent representation, not a
new embedding table.

This mission uses NumPy on hand-sized arrays. Nothing is downloaded.
Weights are **not** a causal story of intent. Multi-head split/merge,
residuals, normalization, and the feed-forward sublayer stay closed
until M30.

Canonical sources: `hf-llm-course` and `karpathy-zero-to-hero` via
`data/source_registry.json`.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
