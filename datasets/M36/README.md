# M36 fixtures

Offline hybrid-retrieval fixtures for **Understand Vector Databases and Hybrid Retrieval**.

These files are synthetic, deterministic, and authored for M36. They
reuse the M33 teaching documents, the frozen M34 questions, and the
M35 exact ranking oracle. They are not a production vector database,
not a Qdrant snapshot, and not a license to mix cosine with BM25.
They require no download and no network.

- `expected.json` — fixture identities, channel ids, filter misses,
  and effort comparisons (not learner evidence).
- `transfer.json` — a fresh RRF / late-filter / effort packet for the
  no-AI gate.
- `freeze_expected.py` — regenerates `expected.json` from the teaching
  store.

Do not edit `datasets/M34` or `datasets/M35` from this mission. Do not
treat this directory as a managed ANN deployment.
