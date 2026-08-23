# M37 Formal Engineering Review Brief

## Review decision requested

Approve M37 as the V10 tool-calling package: an offline, deterministic
local model-call fixture runtime with strict schemas, validation
before execution, two-tool selection plus no-tool, structured
schema-versus-tool errors, approval and idempotency on a side-effecting
mock, bounded repair retry, and a consume of M32's InferenceConfig
(`weights_updated=False`).

This is an implementation review, not learner sign-off. Formal
engineering review is required at M37 (V10 start).

## System under review

- three-tool local registry (`compute_vat`, `lookup_catalog_price`,
  `post_ledger_entry`)
- independent VAT `80 * 0.25` → tax `20`, total `100`
- catalog `SKU-7` → `42.0`
- named defects `malformed_reaches_side_effect` and
  `duplicate_side_effect`
- repair from the broken trace via `repair_run`
- optional live adapter raises `LiveAdapterUnavailable`
- separate `optional_live_llm.py` is not imported on the required path
- no secrets; no network; no pretrained download; no LangGraph/Anthropic SDK

## Required reviewer challenges

- verify M37 is blocked by M32 and hands a registry/trace to M38;
- independently multiply `80 * 0.25` and match `compute_vat`;
- confirm invalid proposals never increment `session.executions`;
- confirm bool is rejected as a number;
- confirm `SKU-ZZ` is a tool error after validation, while missing
  `rate` is a schema error that never executes;
- confirm same-key ledger replay leaves `effect_count == 1`;
- confirm sticky repairs exhaust `max_attempts=3` with
  `retry_exhausted` and `execution_reached is False`;
- confirm `malformed_reaches_side_effect` posts once with a `str`
  amount and repair refuses to execute that proposal;
- confirm `duplicate_side_effect` posts twice and repair from the
  broken snapshot posts once;
- confirm `repair_run` uses the broken trace's proposal and initial
  ledger snapshot;
- confirm M32 `InferenceConfig` identity (`type.__module__` is
  `missions.M32.inference_adaptation`) and `weights_updated is False`;
- search code cells for openai/anthropic/langgraph SDKs, RAG,
  Qdrant, torch, and sampling labs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd
  import works;
- confirm learner evidence, ADR decisions, and completion remain
  unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `python3 tools/validate_repo.py` still reports M01-M31 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
