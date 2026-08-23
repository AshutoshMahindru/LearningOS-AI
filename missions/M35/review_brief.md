# M35 Formal Engineering Review Brief

## Review decision requested

Approve M35 as the V09 retrieval-quality package: an offline,
deterministic ranking/chunking lab that freezes M34 labels, consumes
the M33 exact retriever, ships a local lexical reranker, and explicitly
defers ANN/Qdrant/HNSW/hybrid infrastructure to M36 and decoding
controls to M32.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M35.

## System under review

- Frozen `datasets/M34/questions.json` (`eval_version=m34.eval.v1`)
- Versioned chunking over the same M33 source documents
- M33 `search` / `RankedHit.as_evidence` as candidate generation
- local lexical reranker `lex-overlap-v1` on identical candidate sets
- teaching implementations of recall@k, MRR, nDCG
- leakage repair via `repair_eval_boundary`
- NumPy via M33; no secrets; no network; no paid API
- optional model-reranker module that is **not** imported on the required path

## Required reviewer challenges

- verify M35 is blocked by M34 and hands a baseline to M36;
- independently check nDCG/MRR/recall on the transfer toy list;
- confirm ticket 4412 cosine ranks 4413 first and lex rerank ranks 4412 first with the same member set;
- confirm k=1 candidate recall is below k=5 while scored_candidates stays corpus-sized;
- confirm critical-slice nDCG is below aggregate nDCG;
- confirm leaked phrasing raises nDCG and repair restores the clean source_hash while the leaked object stays leaked;
- confirm relabel-after-results inflates metrics with identical ranked ids and is a separate named change;
- search code cells for paid APIs, model-hub downloads, faiss/qdrant/hnsw, and sampling labs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated;
- confirm `validate_repo.py` still reports M01-M31 executable labs;
- confirm this branch does not mark M35 repository-executable.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `validate_repo.py` still reports M01-M31 executable labs;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
