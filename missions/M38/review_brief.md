# M38 Formal Engineering Review Brief

## Review decision requested

Approve M38 as the V10 stateful-workflow package: an offline,
deterministic local reference state machine around M37 tools, with an
explicit serializable state schema, allowed transitions, terminals,
loop bounds, checkpoint/resume that composes ledger idempotency, a
human approval node, recorded-trace replay, and fail-closed optional
LangGraph/live adapters.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M38 (V10 workflow boundary).

## System under review

- nodes `start`, `decide`, `validate`, `approve`, `execute`,
  `assimilate` plus terminals `complete`, `denied`, `failed`,
  `loop_exhausted`
- teaching task: lookup `SKU-7` (price `42.0`) then post that amount
- named defects `infinite_loop` and `replayed_side_effect`
- repair from the broken trace via `repair_run`
- optional LangGraph adapter raises `OptionalLangGraphUnavailable`
- separate `optional_langgraph.py` is not imported on the required path
- no secrets; no network; no pretrained download; no LangGraph/Anthropic SDK

## Required reviewer challenges

- verify M38 is blocked by M37 and hands a serializable workflow to M39;
- independently confirm `SKU-7` price `42.0` and the posted amount;
- confirm resume after lookup restores `last_tool_result` and does not
  duplicate `lookup_catalog_price` or the ledger post;
- confirm unresolved lookup at `max_steps=3` terminates
  `loop_exhausted` with `model_turn == 3`;
- confirm deny leaves `effect_count == 0` and grant posts once;
- confirm `start -> execute` raises `InvalidTransition` and leaves
  `node == start`;
- confirm a recorded proposal replay matches the original terminal and
  `effect_count`;
- confirm `infinite_loop` exceeds `max_steps` without
  `loop_exhausted` and repair from the initial checkpoint terminates;
- confirm `replayed_side_effect` posts twice with distinct entry ids
  and repair from the broken snapshot posts once;
- confirm `repair_run` uses the broken trace's initial checkpoint;
- confirm M37 `RuntimeSession` identity (`type.__module__` is
  `missions.M37.tool_runtime`) and `weights_updated is False`;
- search code cells for openai/anthropic/langgraph SDKs, RAG,
  Qdrant, torch, memory stores, and sampling labs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd
  import works;
- confirm learner evidence, ADR decisions, and completion remain
  unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `python3 tools/validate_repo.py` still reports M01-M34 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
