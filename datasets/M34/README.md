# M34 fixtures

Offline labeled questions for **Build RAG**.

These files are synthetic, deterministic, and authored for M34. They
reuse the M33 teaching corpus through the M33 retriever. They are not
a production knowledge base, not a paid-API log, and not an M35
rerank benchmark. They require no download and no network.

- `questions.json` — frozen questions with relevance, support spans,
  unanswerable flags, and a held-out split. Labels are not cosine
  ranks. Do not retune the synthesizer on `split=holdout`.
- `expected.json` — fixture pipeline properties (not learner evidence).
- `transfer.json` — a fresh 3-chunk pack for the no-AI gate.
- `freeze_expected.py` — regenerates `expected.json` from the M33
  retriever plus the M34 extractive pipeline. Canonical tests load
  the frozen file and compare it to a live run.

M35 may freeze these queries and support labels. M36 must not treat
this directory as a vector-database deployment.
