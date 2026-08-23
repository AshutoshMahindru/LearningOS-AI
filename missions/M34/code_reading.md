# Code reading — retrieve, pack, synthesize, cite, abstain, evaluate

Read `retrieve`, `pack_context`, `synthesize`, `verify_support`,
`classify_failure`, and `answer_query` in
`missions/M34/rag_pipeline.py`. M34's code-reading target is the
**query-to-grounded-answer contract**:

1. normalize query whitespace without rewriting meaning
2. call M33 `search` and keep `RankedHit.as_evidence` rows
3. pack in retrieval rank order under a char/chunk budget
4. copy a span from the pack, or abstain
5. map citations to packed chunk IDs, spans, and index identity
6. fail `verify_support` when the cited text does not support the claim
7. record inference config (frozen synthesizer, no weight update)
8. classify retrieval, context, generation, and citation separately

Before running the code-reading cell, predict:

- whether `as_evidence` rows still carry `index_id` and `source_hash`
  after packing
- whether lowering `max_chars` reorders hits or only drops the tail
- whether `verify_support` fails when the answer text is gold but the
  citation points at a neighbor that lacks the claim stems

Do **not** look for a reranker, a chunk-overlap tuner, HNSW, or a
temperature sampler. Those are M35, M36, and M32. If a failure can be
diagnosed from packed IDs versus gold support IDs, stay at that layer.
