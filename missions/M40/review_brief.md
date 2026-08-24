# M40 Formal Engineering Review Brief

## Review decision requested

Approve M40 as the V11 evaluation package: an offline, deterministic
local harness over M34 and M39 traces, with a versioned eval pack,
deterministic graders for schema/citation/termination/idempotency, a
calibrated rubric boundary, severity slices, regression detection,
observability fields, and fail-closed optional LLM-as-judge.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M40 (V11 evaluation boundary).

## System under review

- frozen pack `m40.eval.v1` (synthetic, downloaded=false)
- teaching systems: M34 extractive RAG and M39 robust agent
- named governance defects `hidden_critical` and `contaminated_pack`
- repair from the broken trace via `repair_run`
- optional LLM-as-judge raises `OptionalLLMJudgeUnavailable`
- no secrets; no network; no pretrained download; no paid eval SDK

## Required reviewer challenges

- verify M40 is blocked by M34 and M39 and hands a versioned suite to M41;
- independently confirm baseline case ids and that the canonical gate passes on healthy fixtures;
- confirm `missions.M34.rag_pipeline` and `missions.M39.robust_agent` package identity (`__name__` / `sys.modules`);
- confirm citation, schema, termination, and idempotency failures localize to issue kind and object id;
- confirm rubric disagreement is measured and cannot replace invariant graders;
- confirm a high outcome average can coexist with a non-zero critical slice;
- confirm regression injection keeps `eval_version` and case ids unchanged while the canonical gate fails;
- confirm ablating `used_memory_ids` or `citation_ids` blocks a named diagnosis;
- confirm `hidden_critical` aggregate-only gate passes and repair from the same report fails canonical gates;
- confirm `contaminated_pack` uses `m40.eval.tuned-dev` and repair reloads `m40.eval.v1`;
- confirm `repair_run` uses the broken object's report or pack, not two unrelated happy-path runs;
- confirm M34/M39 inference remains unevaluated as a training run (`weights_updated` is not flipped here);
- search code cells for openai/anthropic/langgraph SDKs, paid eval SDKs, torch, faiss, qdrant, and M41 architecture diagrams as implementations;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm MAP cells do not print the ranked/gate answers asked next;
- confirm learner evidence, ADR decisions, V11 closure, and completion remain unpopulated;
- confirm `validate_repo.py` still reports M01-M36 and status does not mark M40 repository-executable.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `python3 tools/validate_repo.py` still reports M01-M36 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
