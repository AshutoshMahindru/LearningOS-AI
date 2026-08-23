# M39 — Add Memory, Routing, and Fallbacks

M38 already handed an explicit stateful workflow: serializable
`AgentState`, allowed transitions, checkpoints, approval, and loop
bounds around M37 tools. The useful whole here is a **robust wrapper**,
not another node and not an eval harness:

```
working vs persistent memory
  -> retrieve (provenance, expiry, scope)
  -> select route (catalog_purchase | catalog_lookup | no_match)
  -> wrap M38
  -> bounded fallback / circuit
  -> complete | degraded | no_match | circuit_open
```

Workflow fields such as `node`, `step`, `pending_action`,
`last_tool_result`, `approval`, and `history` stay working/ephemeral.
Facts like a catalog price can become persistent memory only with
provenance, time, and scope. Retrieval is deliberate: irrelevant and
expired rows are excluded. Degraded success is labeled. Using stale
memory as if it were current is not success.

Canonical path: local deterministic fixtures plus this robustness
layer wrapping `missions.M38.agent_workflow`. Nothing is downloaded.
No paid API. Do not import LangGraph or Anthropic SDKs. An optional
live / LangGraph adapter, if present, is fail-closed and is not
required.

Canonical sources: `langgraph-docs` and `anthropic-agents` via
`data/source_registry.json` (named, not imported).

Content bundle: `tool-using-agents` (M37+M38+M39). This mission closes
V10 robustness; M40 opens systematic evaluation.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
