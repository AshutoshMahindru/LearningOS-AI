# Code reading — memory schema, retrieval, routes, ladder, circuit

Read `retrieve_memory`, `select_route`, `run_fallback_ladder`,
`run_robust_task`, and `repair_run` in `missions/M39/robust_agent.py`.
M39's code-reading target is the **robustness layer** around M38:

1. Persistent entries carry provenance, `written_at`, `expires_at`,
   scope, and optional `superseded_by`
2. `retrieve_memory` excludes scope/sku mismatches, expired rows, and
   superseded rows
3. `select_route` applies frozen predicates; `no_match` does not wrap
   M38
4. `run_robust_task` wraps `missions.M38.agent_workflow.run_workflow`
5. `run_fallback_ladder` counts attempts and records a circuit breaker
6. A degraded purchase is `terminal="degraded"` with `degraded=True`,
   not `complete`
7. `repair_run` recomputes from the broken object's initial store

Before running the code-reading cell, predict:

- whether an expired SKU-7 price appears in `included_ids`
- which route name `select_route` returns for a haiku with no SKU
- what `attempts` and `circuit_open` are when primary and fallback both
  fail
- whether a lookup-only fallback of a purchase is `degraded` or
  `complete`
- what `repair_run` reuses from a broken object (initial store versus
  module defaults)

Do **not** look for a LangGraph SDK store, a RAG pack, a Qdrant
client, a temperature sampler, or an M40 eval harness. If a failure
can be diagnosed from `retrieved_ids`, `route`, `attempts`, or
`degraded`, stay at that boundary.

Do not print substring membership of a later helper. Probe the live
objects: retrieved ids, route name, attempt count, degraded flag.
