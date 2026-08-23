# M37 ADR Prompt — V10 Tool-Execution Trust Boundary

Use `templates/ADR.md`. The decision is not pre-selected. M37 needs a
policy for the V10 artifact M38 will inherit: how strict schemas are,
when a human must approve a side effect, how idempotency keys work,
how far retries may go, and which trace fields are required.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Schema strictness (required fields, extra keys, types, bool-as-number):** [UNFILLED BY LEARNER]
- **Permissions / confirmation:** [UNFILLED BY LEARNER]
- **Idempotency:** [UNFILLED BY LEARNER]
- **Retry limits:** [UNFILLED BY LEARNER]
- **Required trace fields:** [UNFILLED BY LEARNER]
- **Live-provider fallback:** [UNFILLED BY LEARNER]
- **Teaching-scale limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V10 default for tool-execution trust (strict schemas,
fail-closed validation before side effects, approval on mutating
tools, keyed idempotency, bounded repair, staged traces). Do **not**
claim the three-tool fixture is a production gateway, and do not
implement M38 state machines, M39 memory/routing, or M40 eval in
this ADR.

## Alternatives that must be compared

1. Strict schemas (`additionalProperties: false`, reject bool-as-number);
   validate before every execution; side-effecting tools require
   approval and an idempotency key; retries bounded at 3 attempts
   without unsafe coercion; traces always record selection,
   validation, execution, and result.
2. Trust parseable JSON from the model; coerce missing/wrong-type
   fields; treat retries as unbounded; log only the final string so
   traces stay small.
3. Require a live vendor model and a LangGraph graph as the CI path
   for every tool call, including sampling controls.

## Evidence required

Use the schema table, the three-intent selection set, the ledger
replay, the retry-exhausted trace, the unknown-SKU tool error, and
the malformed/duplicate repair. Do not use a vendor model card as
evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a real provider replacing local
fixtures, an M38 state machine wrapping this runtime, adding memory
or fallbacks (M39), attaching an eval harness (M40), or any path that
executes a tool before validation or posts twice on the same key.
