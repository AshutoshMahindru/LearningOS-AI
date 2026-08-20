# Code reading — ingest, index, query, filter, score, ties, freshness

Read `compose_document`, `build_index`, `search`, `exact_cosine_rank`,
`assert_fresh`, and `rebuild_index` in `missions/M33/semantic_search.py`.
M33's code-reading target is the **query-to-evidence contract**:

1. stable `document_id` / `chunk_id` and recovered text spans
2. load bundled vectors (or encode with the M28 teaching encoder)
3. store embedding fingerprint plus `source_hash`
4. encode the query with the same provenance
5. apply metadata filters **before** scoring (eligibility)
6. cosine on every eligible row
7. sort by `(-score, chunk_id)` and cut to top-k
8. return ids, score, span, text, provenance
9. refuse a live corpus whose hash does not match the index

Before running the code-reading cell, predict:

- whether changing `top_k` from 1 to 8 changes `scored_candidates`
  on an unfiltered exact search
- whether `doc-account-access.text[span]` equals the chunk text
- whether `search` on a mutated live corpus raises before it returns
  hits when `enforce_freshness=True`

Do **not** look for softmax, Q/K/V, a generator, a reranker, or HNSW.
Those are M29 and M34–M36. If a failure can be diagnosed from index
metadata, stay at that level.
