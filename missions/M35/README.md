# M35 — Improve Retrieval

M34 already answers from retrieved evidence. The useful whole here is a
**measured retriever**:

`frozen eval → baseline candidate ranking → worst queries → one change
(chunking / candidate k / rerank)`

A high cosine is not a relevance label. A better average nDCG is not a
better system if the critical slice still ranks a trap first. Candidate
recall is not final ranking: a gold span that never entered the
candidate set cannot be rescued by a reranker.

This mission freezes `datasets/M34` questions and support/relevance
ids. It rechunks the **same** M33 source documents under versioned
corpus ids, scores exact M33 candidates, and applies a **deterministic
local** lexical reranker to identical candidate sets. Nothing is
downloaded. No paid API is required. A model-reranker adapter exists
only as an optional unused module.

This mission does **not** deploy Qdrant, HNSW, ANN, or hybrid/sparse
fusion (M36), does **not** open a decoding/sampling lab (M32), and
does **not** retune M34 extractive generation. Name the canonical
sources `sentence-transformers`, `qdrant-docs`, and `hnsw-paper`; do
not implement those stacks here.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
