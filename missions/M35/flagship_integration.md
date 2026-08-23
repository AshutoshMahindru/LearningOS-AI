# V09 integration — measure ranking before choosing infrastructure

## M34 → M35 boundary

M34 already grounds answers and freezes questions. M35 is the first
V09 mission that **measures retrieval quality** as a ranking problem:

`frozen labels → versioned chunks → M33 candidates → local rerank →
recall / MRR / nDCG → slices`

plus a candidate/rerank interface that M36 can consume.

The observable V09 surface after M35 is a measured exact-search
baseline: frozen eval version, known failure slices (ticket/invoice
traps, mixed large chunks, hard negatives), and numbers for candidate
k versus a latency proxy.

## What M35 must not change

M35 does not require Qdrant, HNSW, or hybrid fusion (M36). It does not
open temperature / top-p sampling (M32). It does not edit
`datasets/M34` labels to flatter a reranker. It does not treat
`sentence-transformers` as a required download.

## M35 → M36 handoff

M36 may choose exact versus approximate infrastructure only after the
learner can defend:

- frozen eval identity (`m34.eval.v1`)
- candidate set versus reranked order
- aggregate versus critical-slice metrics
- a latency proxy that does not pretend exact search scored only k rows
- known failures (neighbor tickets, mixed merged chunks, hard negatives)

Reusable artifacts: `generate_candidates` / `rerank_candidates`,
`ExperimentConfig.identity`, frozen M34 labels, `datasets/M35/expected.json`
fixture properties (not learner evidence), and slice names.
