# M33 — Build Semantic Search

M28 ranked bundled sentence vectors. The useful whole here is a
**search service**:

`query → embed (load, do not download) → exact cosine index → top-k
evidence with document/chunk IDs, scores, text spans, and provenance`

Every index carries an **identity**: embedding model/version/metric,
corpus version/hash, filter schema, and a stale-index policy. High
cosine is a retrieval score. It is not labeled relevance and it is not
an answer.

This mission uses the **bundled** M28 encoder (`v06-teaching-meanpool`
`v06.1`, mean pool, L2, cosine) and an **exact in-memory** baseline.
Nothing is downloaded. Qdrant is an optional local adapter, not a CI
dependency.

This mission does **not** generate answers (M34), does **not** rerank
or retune chunk size (M35), does **not** require ANN/HNSW/hybrid
infrastructure (M36), and does **not** treat Q/K/V attention as search
(M29). Inspect ranked evidence on a tiny labeled corpus first.

Canonical sources: `sentence-transformers` and `qdrant-docs` via
`data/source_registry.json`. Use the bundled fixtures; do not download
encoder weights.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
