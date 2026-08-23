# M39 Formal Engineering Review Brief

## Review decision requested

Approve M39 as the V10 robustness package: an offline, deterministic
local memory/router/fallback layer wrapping M38, with
provenance-bearing persistent memory, explicit routes including
no-match, a bounded fallback ladder, a circuit breaker, an explicit
degraded terminal, and fail-closed optional live/LangGraph adapters.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M39 (V10 robustness boundary).

## System under review

- working/ephemeral M38 fields versus persistent catalog facts
- teaching task: retrieve, route, and wrap M38 for `SKU-7` (price `42.0`)
- named defects `stale_memory_trusted` and `fallback_oscillation`
- repair from the broken trace via `repair_run`
- optional live retrieve raises `LiveAdapterUnavailable`
- optional LangGraph store raises `OptionalLangGraphUnavailable`
- no secrets; no network; no pretrained download; no LangGraph/Anthropic SDK

## Required reviewer challenges

- verify M39 is blocked by M38 and hands declared memory/route/fallback traces to M40;
- independently confirm `SKU-7` price `42.0` and a healthy posted amount;
- confirm `missions.M38.agent_workflow` package identity (`type.__module__`);
- confirm irrelevant ids are excluded under a fixed purchase task;
- confirm an expired or superseded entry is flagged/excluded at `now=1000`;
- confirm the frozen case set routes to `catalog_purchase`, `catalog_lookup`, and `no_match`;
- confirm no-match does not wrap M38;
- confirm primary failure of purchase yields `degraded=True` and does not claim `complete`;
- confirm all-rung failure opens the circuit within `max_attempts`;
- confirm `stale_memory_trusted` posts `99.0` as `complete` and repair from the initial store posts the catalog amount;
- confirm `fallback_oscillation` exceeds `max_attempts` without `circuit_open` and repair opens the circuit;
- confirm `repair_run` uses the broken trace's initial store and task;
- confirm M37 inference remains `weights_updated is False`;
- search code cells for openai/anthropic/langgraph SDKs, RAG, Qdrant, torch, eval harnesses, and sampling labs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm MAP cells do not print the ranked/route/stale answers asked next;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `python3 tools/validate_repo.py` still reports M01-M35 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
