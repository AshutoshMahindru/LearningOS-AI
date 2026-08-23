# M39 ADR Prompt — V10 Memory, Routing, and Fallback Policy

Use `templates/ADR.md`. The decision is not pre-selected. M39 needs a
policy for the V10 artifact M40 will inherit: what is persistent,
how provenance and retention work, how retrieval is scoped, how
routes take precedence, how far a fallback may run, when output is
degraded, and when a circuit opens.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Persistence (working vs durable facts):** [UNFILLED BY LEARNER]
- **Provenance / retention:** [UNFILLED BY LEARNER]
- **Retrieval scope:** [UNFILLED BY LEARNER]
- **Route precedence (including no-match):** [UNFILLED BY LEARNER]
- **Retry / fallback bounds:** [UNFILLED BY LEARNER]
- **Degraded outputs:** [UNFILLED BY LEARNER]
- **Circuit breaking:** [UNFILLED BY LEARNER]
- **Trace surfaces retained for M40:** [UNFILLED BY LEARNER]
- **Live-provider / LangGraph fallback:** [UNFILLED BY LEARNER]
- **Teaching-scale limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V10 default for robustness (persistent facts are
provenance-bearing and scoped; retrieval excludes expired and
irrelevant rows; routes are explicit predicates with a no-match;
primary wraps M38; fallbacks are bounded; degraded success is labeled;
a circuit opens after consecutive failures). Do **not** claim the
teaching layer is a production agent runtime, and do not implement
M40 eval in this ADR.

## Alternatives that must be compared

1. Explicit persistent memory with provenance/expiry/scope; retrieve
   by sku and scope; frozen route predicates including `no_match`;
   wrap M38 as primary; one lookup-only fallback; `max_attempts=3`;
   circuit after 2 consecutive failures; degraded purchase is not
   `complete`.
2. Append the whole transcript as memory; let the model choose a
   route in prose; retry forever; treat any lookup as success.
3. Require a live vendor memory store and a LangGraph checkpointer as
   the CI path for every run, including an M40 eval harness.

## Evidence required

Use the field-classification map, the relevance exclusion, the stale
or superseded flag, the frozen route case set, the primary-failure
degraded run, the circuit-bound run, and the stale-trust /
oscillation repairs. Do not use a vendor model card as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a real provider replacing local
fixtures, attaching an eval harness (M40), trusting expired memory as
`complete`, checkpointing mid-execute in M38, or any path that
oscillates between fallbacks without a hard stop.
