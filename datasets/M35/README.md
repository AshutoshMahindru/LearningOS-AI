# M35 fixtures

Offline ranking/chunking fixtures for **Improve Retrieval**.

These files are synthetic, deterministic, and authored for M35. They
reuse the M33 teaching documents and the **frozen** M34 questions in
`datasets/M34/questions.json`. They are not a production index, not a
Qdrant snapshot, and not a license to relabel after seeing scores.
They require no download and no network.

- `chunk_versions.json` — declared chunk size/overlap versions over the
  same source documents. New chunk ids are projected onto frozen labels
  by span overlap.
- `hard_negatives.json` — high-similarity irrelevant extras. They are
  not added to M34 relevant/support lists.
- `expected.json` — fixture metric properties (not learner evidence).
- `transfer.json` — a fresh 3-hit list for the no-AI gate.
- `freeze_expected.py` — regenerates `expected.json` from the frozen
  M34 labels plus the M35 teaching pipeline.

Do not copy-edit `datasets/M34` labels to fit a reranker. M36 must not
treat this directory as a vector-database deployment.
