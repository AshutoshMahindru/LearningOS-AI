# M38 — Build a Stateful Agent Workflow

M37 already handed a validated tool runtime: schemas, typed errors,
approval, idempotency, and staged traces. The useful whole here is a
**stateful workflow**, not another single tool call and not a memory
router:

```
explicit AgentState
  -> decide (model fixture)
  -> validate (M37)
  -> approve (human gate)
  -> execute (M37)
  -> assimilate (last_tool_result)
  -> complete | denied | failed | loop_exhausted
```

Chat history is not a state schema. Resume must restore
`last_tool_result`, completed effect keys, and the ledger snapshot so
a completed side effect is not replayed. Loops end at an explicit
bound. Illegal edges raise; they do not silently mutate.

Canonical path: local deterministic model-call fixtures plus this
reference state machine. Nothing is downloaded. No paid API. Do not
import LangGraph or Anthropic SDKs. An optional LangGraph adapter, if
present, is fail-closed and is not required.

Canonical sources: `langgraph-docs` and `anthropic-agents` via
`data/source_registry.json` (named, not imported).

Content bundle: `tool-using-agents` (M37+M38+M39). This mission opens
the state machine; M39 opens memory, routing, and fallbacks.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
