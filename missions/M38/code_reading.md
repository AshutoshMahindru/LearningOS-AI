# Code reading — state, nodes, transitions, checkpoint, resume, bound

Read `apply_transition`, `run_workflow`, `checkpoint`, `resume`, and
`repair_run` in `missions/M38/agent_workflow.py`. M38's code-reading
target is the **reference state machine**:

1. `AgentState` is explicit and serializable (`node`, `last_tool_result`,
   `completed_effect_keys`, `model_turn`, ledger snapshot)
2. `apply_transition` consults `ALLOWED_TRANSITIONS` and raises
   `InvalidTransition` instead of mutating
3. `decide` asks a model fixture; it does not call the tool
4. `validate` uses M37 `parse_proposal` / `validate_proposal`
5. `approve` is a node; deny is a terminal
6. `execute` calls M37 `run_tool_call` and composes ledger idempotency
7. `assimilate` writes `last_tool_result` before the next `decide`
8. `checkpoint` / `resume` restore state without replaying completed
   side effects
9. `model_turn >= max_steps` becomes `loop_exhausted`

Before running the code-reading cell, predict:

- whether `start -> execute` mutates `state.node`
- what `resume` restores from a lookup checkpoint (`last_tool_result`,
  ledger `effect_count`, current node)
- what `repair_run` reuses from the broken object (initial checkpoint
  versus module defaults)
- the live `effect_count` after an approved SKU-7 purchase and after a
  denied purchase

Do **not** look for a LangGraph SDK graph, a memory store, a RAG pack,
a Qdrant client, or a temperature sampler. Those are later or parallel
missions. If a failure can be diagnosed from `state.node`,
`ledger.effect_count`, or `model_turn`, stay at that boundary.

Do not print substring membership of a later helper. Probe the live
objects: current node, last_tool_result, effect_count, model_turn.
