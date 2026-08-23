# V09 integration — freeze the grounding contract

## M33 / M31 → M34 boundary

M33 already returns ranked evidence with identity. M31 already
separated training-time updates from inference-time consumption.
M34 is the first V09 mission that **grounds** an answer:

`query → M33 evidence → budgeted pack → extractive claim + citations
→ support check or abstention`

plus traces that M35 can freeze.

The observable V09 surface after M34 is a small evaluated RAG
baseline: fixed questions, relevance and support labels, abstention
on unanswerable items, and separable retrieval/context/generation/
citation failures.

## What M34 must not change

M34 does not rerank or retune chunk overlap (M35). It does not
require Qdrant, HNSW, or hybrid fusion (M36). It does not open
temperature / top-p sampling (M32). It does not relabel cosine as
truth.

## M34 → M35 handoff

M35 may optimize ranking and chunking only after the learner can
defend:

- source IDs on every answered trace
- abstention on unanswerable queries
- separable failure layers
- unsupported citations failing evaluation
- the frozen files in `datasets/M34/`

Reusable artifacts: `answer_query` / `RagTrace`, `questions.json`
labels, `expected.json` pipeline properties (not learner evidence),
and `classify_failure` layer names.
