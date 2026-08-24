# M40 — Evaluate AI Systems Systematically

M34 already grounds answers. M39 already routes, remembers, and
fails closed. The useful whole here is an **evaluation harness**, not
another model and not an M41 architecture diagram:

```
frozen eval pack (RAG + agent + tool)
  -> invoke (M34 / M39 / M37 traces)
  -> deterministic graders
  -> rubric only where needed
  -> slices, proxies, traces
  -> release gate
```

Write the cases before optimizing either system. A high average is
not permission to ship a rare unsupported citation, a schema-invalid
tool call, a mislabeled `complete`, or a double ledger post. Rubric
and LLM-as-judge sit behind that line.

Canonical path: local deterministic fixtures plus this harness
importing `missions.M34.rag_pipeline` and `missions.M39.robust_agent`
as packages. Nothing is downloaded. No paid API. Do not import OpenAI,
Anthropic, LangGraph, or paid eval SDKs. An optional LLM-as-judge
adapter, if present, is fail-closed and is not a required grader.

Canonical sources: `anthropic-evals` and `anthropic-agents` via
`data/source_registry.json` (named, not imported).

Content bundle: `ai-system-evals`. This mission **opens** V11
evaluation; it does **not** close V11. Architecture remains M41.
P7 phase-end here is the evaluation opening, not learner completion.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
