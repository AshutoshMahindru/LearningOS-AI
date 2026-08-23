# Code reading — store schema, effort, filters, fusion, lifecycle

Read `open_teaching_store`, `build_adjacency`, `approximate_search`,
`filter_placement_trace`, `sparse_search`, `reciprocal_rank_fusion`,
`fuse_channels`, `mix_raw_scores`, `insert_chunk`, `update_chunk_text`,
`delete_chunk`, `rebuild_store`, and `HybridHit.as_evidence` in
`missions/M36/hybrid_retrieval.py`. M36's code-reading target is the
**retrieval-infrastructure contract**:

1. keep M35 exact search as the oracle (`index_id=v08-exact-memory`)
2. build a local navigable graph with an `ef` effort knob
3. apply payload filters before a small top-k, not after
4. score sparse BM25 as its own channel
5. fuse **ranks** with declared RRF; never add cosine to BM25
6. insert/update/delete mark the store dirty; rebuild restores freshness
7. evidence rows keep chunk ids, ranks, fusion method, and store identity

Before running the code-reading cell, predict:

- whether `HybridHit.as_evidence` includes a `fusion` key
- the store graph `entry_id` and `degree_m`
- how many neighbors `build_adjacency` stores for that entry
- whether `build_adjacency`'s docstring calls the graph a production
  hierarchical index

Do **not** look for a production hierarchical index implementation,
a live Qdrant client, a tool executor, or a temperature sampler.
Those stay named as sources/deferred missions. Probe **objects**
(candidate ids, filter-eligible misses, fusion ranks, generation /
dirty flags), not misleading substring booleans.
