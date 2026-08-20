# M28 fixtures

Offline teaching embeddings for **Search Meaning with Embeddings**.

These files are synthetic, deterministic, and authored for M28. They are
not Sentence Transformers weights, not a model-hub checkpoint, and not a
quality benchmark. They require no download and no network.

- `token_table.json` — frozen 11-D token semantics plus a hashed residual
  axis (`v06-teaching-meanpool`, version `v06.1`). Mean-pool then L2.
- `catalog.json` — corpus and query texts, experiment tags, and fixture
  ranking properties (not learner evidence).
- `embeddings.json` — precomputed 12-D unit vectors with provenance
  (model, version, metric `cosine`, normalization `l2`, pooling `mean`).
- `mismatch.json` — same document ids with incompatible provenance
  (`v06.2`, `v06-teaching-meanpool-alt`, normalization `none`, account/print
  axes swapped). Used only for the controlled failure.
- `transfer.json` — a fresh 4-D hand-authored set for the no-AI gate.
- `generate_embeddings.py` — regenerates the JSON from the frozen token
  table. Canonical tests load the frozen files; they do not download a
  model.

M29 may reuse the vector intuition. M33 may reuse the embedding, metric,
normalization, and provenance contract plus these retrieval fixtures.
M33 must not treat this directory as a production index.
