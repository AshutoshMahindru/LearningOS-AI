# M34 — Build RAG

M33 returns ranked evidence. The useful whole here is a **grounded
answerer**:

`query → M33 retrieval → budgeted context pack → extractive synthesis
→ citations → support check or abstention`

A high retrieval score is not an answer. A fluent sentence is not
grounded unless a cited span actually supports the claim. Unanswerable
queries must abstain. Retrieval misses, context-budget drops,
generation misuse, and unsupported citations are different failures.

This mission uses the **bundled** M33 exact retriever
(`v08-exact-memory` over `v06-teaching-meanpool` `v06.1`) and a
**local extractive** synthesizer. Nothing is downloaded. No paid API
is required. A live-model adapter exists only as an optional unused
module.

This mission does **not** rerank or retune chunk size (M35), does
**not** require ANN/HNSW/hybrid infrastructure (M36), and does **not**
open a decoding/sampling lab (M32). Inspect evidence before synthesis.

Canonical sources: `rag-paper` and `sentence-transformers` via
`data/source_registry.json`. Use the bundled fixtures.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
