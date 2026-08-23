# M37 — Make an LLM Call Tools Reliably

M32 already handed an inference-provider contract: `InferenceConfig`,
frozen weights (`training_time=False`, `weights_updated=False`), and a
fail-closed live adapter. The useful whole here is a **validated tool
call**, not another decoder and not a multi-step agent:

```
declared intent → model-call fixture → parse → select
  → validate schema → permission / idempotency → execute
  → structured result + staged trace
```

A parseable JSON blob is not a reliable tool call. Invalid arguments
must fail closed **before** the tool runs. Schema failure is not tool
failure. Side-effecting tools need approval and an idempotency key so
a timeout-retry cannot post twice.

Canonical path: local deterministic model-call fixtures. Nothing is
downloaded. No paid API. Do not import LangGraph or Anthropic SDKs.

Canonical sources: `anthropic-agents` and `langgraph-docs` via
`data/source_registry.json` (named, not imported).

Content bundle: `tool-using-agents` (M37+M38+M39). This mission opens
the tool contract; M38 opens persistent state machines; M39 opens
memory, routing, and fallbacks.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
