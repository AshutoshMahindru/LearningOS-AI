# M33 Formal Engineering Review Brief

## Review decision requested

Approve M33 as the V08 semantic-search package: an offline,
deterministic exact retriever that inherits the M28 embedding contract
and explicitly defers generation to M34, rerank/chunk tuning to M35,
ANN/hybrid infrastructure to M36, and attention to M29.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M33.

## System under review

- bundled `v06-teaching-meanpool` v06.1 vectors copied under `datasets/M33/`
- exact in-memory cosine index with document/chunk IDs and text spans
- top-k, metadata filters, labeled hit/recall, latency recording
- stale-index `IndexStaleError` repair via rebuild-or-reject
- incompatible `v06.2` store fail-closed
- NumPy exact baseline; no secrets; no network; no encoder download
- optional Qdrant module that is **not** imported on the required path

## Required reviewer challenges

- verify M33 is blocked by M28 and hands a retriever interface to M34;
- independently brute-force cosine-rank `q-password` on the frozen unit vectors;
- confirm top-k changes the window without changing `scored_candidates` when unfiltered;
- confirm `topic=device` scores only printer chunks for a password query;
- confirm negation's hard neighbor scores high without being labeled relevant;
- reproduce stale serving of indexed password text after a live corpus edit,
  then `IndexStaleError`, then rebuild from those objects;
- reproduce mixed `v06.2` printer ranking and the provenance gate;
- search code cells for model-hub downloads, softmax, Q/K/V, rerankers,
  generators/citations, faiss, and required qdrant;
- verify source notebook IDs are unique, outputs empty, and labs-cwd
  import works;
- confirm learner evidence, ADR decisions, and completion remain
  unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `validate_repo.py` still reports M01-M22 executable labs;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
