# M34 Formal Engineering Review Brief

## Review decision requested

Approve M34 as the V09 RAG package: an offline, deterministic
extractive pipeline that consumes the M33 retriever, records an
inference-time synthesizer config, and explicitly defers rerank/
chunk tuning to M35, ANN/hybrid infrastructure to M36, and decoding
controls to M32.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M34.

## System under review

- M33 `search` / `RankedHit.as_evidence` as the only retriever
- local extractive synthesizer `extractive-span-v1`
- context budget, citations, abstention, support checker
- frozen `datasets/M34` questions with holdout split
- unsupported-citation repair via `repair_grounding`
- NumPy via M33; no secrets; no network; no paid API
- optional live-model module that is **not** imported on the required path

## Required reviewer challenges

- verify M34 is blocked by M31 and M33 and hands a baseline to M35;
- independently check that `as_evidence` fields survive packing and citations;
- confirm closed-book canonical answers abstain on the same questions;
- confirm password-procedure k=1 is a retrieval miss and budget=80 is a context miss;
- confirm ticket 4412 at k=1 abstains while naive top-1 answers 4413;
- confirm CEO weather score is high and the gated policy still abstains;
- reproduce unsupported citation of the login-reset claim, failed support, then rebind-or-abstain from those objects;
- search code cells for paid APIs, model-hub downloads, rerankers, nDCG/MRR as the skill, faiss/qdrant/hnsw, and sampling labs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated;
- confirm `validate_repo.py` still reports M01-M31 executable labs.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `validate_repo.py` still reports M01-M31 executable labs;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
