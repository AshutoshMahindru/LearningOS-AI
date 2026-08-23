# Code reading — chunk, retrieve, label, rerank, score, identity

Read `rechunk_corpus`, `generate_candidates`, `project_labels`,
`rerank_candidates`, `ndcg_at_k`, `evaluate_set`, and
`ExperimentConfig.identity` in `missions/M35/retrieval_eval.py`.
M35's code-reading target is the **frozen-eval ranking contract**:

1. load M34 labels with `eval_version` and original chunk ids
2. rebuild a versioned corpus from the same source documents
3. call M33 `search` and keep `RankedHit.as_evidence` rows
4. rerank without adding or dropping candidate members
5. project labels by identity when original ids survive, else span overlap
6. compute recall@k / MRR / nDCG from ranked ids and grades
7. keep experiment-config identity stable for a named change
8. refuse to treat cosine as a relevance label

Before running the code-reading cell, predict:

- whether ticket `4412` cosine ids equal lex-reranked ids, or only the
  member set matches
- whether `ndcg_at_k` is a Python float computed here, not a library call
- whether a leaked corpus keeps a different `source_hash` after repair
  of a separate clean object

Do **not** look for HNSW, Qdrant, hybrid fusion, or a temperature
sampler. Those are M36 and M32. Probe **objects** (metric values,
candidate ids versus reranked ids, version ids), not misleading
substring booleans.
