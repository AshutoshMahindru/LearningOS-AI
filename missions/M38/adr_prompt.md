# M38 ADR Prompt — V10 Workflow State and Checkpoint Policy

Use `templates/ADR.md`. The decision is not pre-selected. M38 needs a
policy for the V10 artifact M39 will inherit: the state schema, where
state is persisted, how resume works, how far a loop may run, where a
human must approve, and which traces are retained.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **State schema (required fields, last_tool_result, effect keys):** [UNFILLED BY LEARNER]
- **Persistence boundary (what a checkpoint contains):** [UNFILLED BY LEARNER]
- **Resume semantics (replay of completed side effects):** [UNFILLED BY LEARNER]
- **Loop limits:** [UNFILLED BY LEARNER]
- **Approval points:** [UNFILLED BY LEARNER]
- **Trace retention:** [UNFILLED BY LEARNER]
- **Live-provider / LangGraph fallback:** [UNFILLED BY LEARNER]
- **Teaching-scale limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V10 default for workflow state (explicit serializable schema,
checkpoint after a finished node, resume restores last_tool_result and
the ledger snapshot, bounded model turns, approval node on mutating
tools, retain transition history at teaching scale). Do **not** claim
the teaching graph is a production agent runtime, and do not implement
M39 memory/routing or M40 eval in this ADR.

## Alternatives that must be compared

1. Explicit serializable `AgentState`; checkpoint after assimilate;
   resume restores `last_tool_result`, `completed_effect_keys`, and the
   M37 ledger snapshot; `max_steps` is enforced at `decide`;
   side-effecting tools pause at `approve`; traces record every
   transition.
2. Treat chat history as state; skip checkpoints; leave loops to the
   model; ask the model to "remember" approvals; log only the final
   assistant string.
3. Require a live vendor model and a LangGraph checkpointer as the CI
   path for every run, including a long-term memory store.

## Evidence required

Use the happy-path transition list, the lookup checkpoint, the
unresolved-loop terminal, the approve/deny pair, the rejected
`start -> execute`, and the infinite-loop / replayed-post repair. Do
not use a vendor model card as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a real provider replacing local
fixtures, adding memory or fallbacks (M39), attaching an eval harness
(M40), checkpointing mid-execute, resuming without `last_tool_result`,
or any path that posts twice on the same key.
