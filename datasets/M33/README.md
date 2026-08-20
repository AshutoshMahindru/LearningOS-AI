# M33 fixtures

Offline teaching corpus and exact index for **Build Semantic Search**.

These files are synthetic, deterministic, and authored for M33. Chunk
vectors are **copies** of `datasets/M28` bundled embeddings with
explicit provenance. They are not Sentence Transformers weights, not a
Qdrant snapshot, and not a quality benchmark. They require no download
and no network.

- `corpus.json` — documents, stable chunk IDs, metadata, and texts.
  Chunk spans are recovered from concatenated document text.
- `vectors.json` — precomputed 12-D unit vectors (`v06-teaching-meanpool`
  `v06.1`, cosine, L2) keyed by chunk id, copied from M28.
- `queries.json` — labeled queries. Labels are relevance judgments, not
  cosine ranks. Do not relabel after seeing scores.
- `expected.json` — fixture ranking properties (not learner evidence).
- `incompatible_vectors.json` — same chunk ids with M28 `v06.2`
  swapped-axis unnormalized vectors. Used only for fail-closed mixing.
- `transfer.json` — a fresh 3-D hand-authored set for the no-AI gate.
- `generate_index.py` — regenerates the JSON from M28 plus the M33
  wrapping. Canonical tests load the frozen files.

M34 may call the retriever interface. M35/M36 must not treat this
directory as a production index, rerank benchmark, or vector-database
deployment.
