# V10 integration — freeze the workflow state/checkpoint boundary

## M37 → M38 boundary

M37 already executes tools behind a schema, an approval flag, and an
idempotency key. M38 is the first V10 mission that **owns multi-step
control flow**:

```
M37 run_tool_call / ToolTrace / ledger idempotency
  -> explicit AgentState
  -> nodes and allowed transitions
  -> approval node
  -> checkpoint / resume
  -> terminals including loop_exhausted
```

The observable V10 surface after M38 is a small reference machine:
lookup then approved post, serializable state, interrupt/resume
without a duplicate ledger row, and traces M39 can consume.

## What M38 must not change

M38 does not open a persistent memory store, a router, or a fallback
ladder (M39). It does not build a systematic eval harness (M40). It
does not retrieve, cite, or abstain (M34). It does not open
temperature / top-p sampling (M32). It does not import LangGraph or
Anthropic SDKs.

## M38 → M39 handoff

M39 may add memory and routing only after the learner can defend:

- state is explicit and serializable
- illegal transitions do not mutate
- resume preserves last_tool_result and keyed idempotency
- loops terminate at a configured bound
- approval is a node, not a prompt
- traces separate orchestration from model fixtures
- M37 config remains recorded with `weights_updated=False`

Reusable artifacts: `AgentState` / `apply_transition`,
`run_workflow` / `resume` / `checkpoint`, `pipeline_with_defect` /
`repair_run`, and `handoff_contract()`.
