# M36 Formal Engineering Review Brief

## Review decision requested

Approve M36 as the V09 retrieval-infrastructure package: an offline,
deterministic hybrid-retrieval lab that treats M35 exact search as
the oracle, ships a local graph effort adapter (not a production
ANN library), teaches filter placement and declared RRF, fail-closes
optional live stores, and keeps V09 phase-end honest.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M36. M36 is listed in
`data/review_cadence.yaml`.

## System under review

- Frozen `datasets/M34/questions.json` (`eval_version=m34.eval.v1`)
- M35 `evaluate_set` / `generate_candidates` as the exact oracle
- M33 `search` / `RankedHit.as_evidence`
- local teaching graph with `ef` and comparison counts
- BM25 sparse channel and declared RRF
- FilterTrace for pre-filter versus late-after-small-top-k
- insert/update/delete/rebuild with fail-closed dirty search
- optional Qdrant module that is **not** imported on the required path
- NumPy via M33; no secrets; no network; no paid API

## Required reviewer challenges

- verify M36 is blocked by M35 and helpful-on M33;
- independently check RRF on `datasets/M36/transfer.json`;
- confirm ticket 4412 dense top-1 is `doc-tickets::c1` and sparse top-1 is `doc-tickets::c0`;
- confirm `rag-ceo` at ef=1 misses exact neighbors and ef=4 recovers them;
- confirm `Please reset` late-filter at k=1 misses `doc-account-access::c1` while pre-filter returns it;
- confirm raw-sum mix ranks `doc-refund-policy::c1` above the password gold and RRF repair from the same objects does not mutate the mix object;
- confirm dirty insert raises before rebuild;
- search code cells for paid APIs, model-hub downloads, faiss/qdrant/hnsw implementations, and tool/agent labs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated;
- confirm `validate_repo.py` still reports M01-M34 executable labs (do not "fix");
- confirm this branch does not mark M36 repository-executable;
- confirm package imports (`missions.M36.hybrid_retrieval`) share M35 classes (no dummy-module dual load).

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `validate_repo.py` still reports M01-M34 executable labs;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
