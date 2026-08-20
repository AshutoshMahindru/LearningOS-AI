# V08 integration — freeze the retriever

## M28 → M33 boundary

M28 already attaches sentence vectors and refuses mixed provenance.
M33 is the first V08 mission that **indexes** those vectors as a
service: stable document/chunk identity, exact search, filters, labeled
evaluation, and fail-closed staleness.

The observable V08 surface after M33 is:

`query text → 12-D unit vector (v06-teaching-meanpool v06.1)
→ exact cosine against a hashed chunk index → ranked evidence
(document_id, chunk_id, score, text, span, provenance)`

plus metadata filters that change eligibility, not labels.

## What M33 must not change

M33 does not generate answers, cite them, or abstain (M34). It does
not rerank or retune chunk overlap (M35). It does not require Qdrant,
HNSW, or hybrid fusion (M36). It does not relabel cosine as attention
(M29).

## M33 → M34 handoff

M34 may pack these hits into a context window only after the learner
can defend:

- exact baseline parity
- IDs and spans as evidence
- score versus labeled relevance
- filter eligibility
- refused stale and mixed indexes
- the frozen files in `datasets/M33/`

Reusable artifacts: `semantic_search.search` / `RankedHit.as_evidence`,
`corpus.json` identity, `queries.json` labels, `incompatible_vectors.json`
as the negative embedding case, and `expected.json` ranking properties
(not learner evidence).
