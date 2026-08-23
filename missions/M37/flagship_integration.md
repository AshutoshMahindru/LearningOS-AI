# V10 integration — freeze the tool-execution trust boundary

## M32 → M37 boundary

M32 already records inference-provider configuration and an adaptation
hierarchy that *names* tools as the lever for deterministic
computation. M37 is the first V10 mission that **executes** that
lever:

```
InferenceConfig (training_time=False, weights_updated=False)
  -> declared intent
  -> model-call fixture
  -> parse / select
  -> strict schema
  -> approval / idempotency
  -> execute
  -> structured result
  -> ToolTrace (selection | validation | execution | result)
```

The observable V10 surface after M37 is a small validated registry:
two read tools, one side-effecting mock, typed errors, bounded
retry, and traces M38 can consume.

## What M37 must not change

M37 does not open a persistent multi-step state machine (M38). It
does not add memory, routing, or fallbacks (M39). It does not build
a systematic eval harness (M40). It does not retrieve, cite, or
abstain (M34). It does not open temperature / top-p sampling (M32).
It does not import LangGraph or Anthropic SDKs.

## M37 → M38 handoff

M38 may wrap this runtime in an explicit state machine only after
the learner can defend:

- invalid calls never reach execution
- schema failure ≠ tool failure
- side-effect replay is keyed
- retries are bounded
- traces separate selection, validation, execution, and result
- M32 config is recorded with `weights_updated=False`

Reusable artifacts: `run_tool_call` / `ToolTrace`, the default
registry, `pipeline_with_defect` / `repair_run`, and
`handoff_contract()`.
