# M36 — Understand Vector Databases and Hybrid Retrieval

M35 already measured ranking on an exact in-memory index. The useful
whole here is a **retrieval-infrastructure choice**:

`M35 exact oracle → approximate teaching path → filters → sparse →
declared fusion → lifecycle → defended V09 store`

A vector database is more than an array of embeddings. It owns
payload filters, an approximate search-effort knob, sparse and dense
channels, a declared fusion policy, and insert/update/delete/rebuild.
At teaching scale the exact M35 path remains the correctness
reference. Approximate search is allowed to miss neighbors when
effort is low.

This mission does **not** require Qdrant, FAISS, or a managed service.
Canonical sources `qdrant-docs`, `hnsw-paper`, and
`sentence-transformers` are **named**, not installed. An optional
live adapter exists only as a fail-closed module.

This mission does **not** open tool/agent orchestration (M37/M38),
does **not** open a decoding lab (M32), and does **not** retune M34
generation.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
V09 does not close because this package exists.
