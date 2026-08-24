# V10 integration — freeze the memory / route / fallback boundary

## M38 → M39 boundary

M38 already runs a serializable state machine around M37 tools. M39
is the last V10 mission and **owns robustness**:

```
M38 run_workflow / AgentState / checkpoint / approval / loop_exhausted
  -> working vs persistent memory
  -> retrieve with provenance, expiry, scope
  -> explicit routes plus no-match
  -> wrap M38 as primary
  -> bounded fallback and circuit
  -> degraded terminal
```

The observable V10 surface after M39 is a small robust agent: SKU-7
purchase or lookup, scoped memory, frozen routes, a two-rung ladder,
and traces M40 can evaluate.

## What M39 must not change

M39 does not build a systematic eval harness (M40). It does not
retrieve, cite, or abstain (M34). It does not open temperature /
top-p sampling (M32). It does not open ANN/Qdrant (M36). It does not
import LangGraph or Anthropic SDKs. It does not edit M38 or M37.

## M39 → M40 handoff

M40 may evaluate the system only after the learner can defend:

- working state is not dumped into persistent memory
- persistent rows have provenance, time, and scope
- retrieval excludes irrelevant, expired, and superseded rows
- routes are explicit and testable, including no-match
- fallbacks stop at an attempt or circuit bound
- degraded success is not silent incorrect success
- traces expose retrieved ids, route, attempts, and degraded
- M38/M37 config remains recorded with `weights_updated=False`

Reusable artifacts: `MemoryStore` / `retrieve_memory`, `select_route`,
`run_robust_task` / `run_fallback_ladder`, `pipeline_with_defect` /
`repair_run`, and `handoff_contract()`.
