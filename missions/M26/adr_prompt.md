# M26 ADR Prompt — V05 Failure-Triage / Release Policy

Use `templates/ADR.md`. The decision is not pre-selected. M26 needs a
policy for how V05 triages a failed training run before anyone is
allowed to change architecture.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Required sanity checks:** [UNFILLED BY LEARNER]
- **Escalation order:** [UNFILLED BY LEARNER]
- **Rollback triggers:** [UNFILLED BY LEARNER]
- **Evidence required before architecture changes:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose how V05 decides that a deep-learning failure is data, optimization,
gradient flow, evaluation, or (last) architecture. CPU is canonical. Do
**not** claim the choice is globally optimal for every later model or
hardware path.

## Alternatives that must be compared

1. Change architecture first whenever train loss looks high. Fast, and
   it hides label bugs, learning-rate bugs, and leaked evaluation.
2. Require a fixed sanity order: known-good trace, tiny-subset overfit,
   label/scale integrity, learning-rate probe, per-parameter gradient
   check, honest split metrics — and only then a capacity change.
   (Proposed default to discuss.)
3. Gate every release on a Chaos Day with a hidden defect, a filled
   diagnosis record, and a regression test. Highest discipline, highest
   process cost.

## Evidence required

Use the named catalogue runs, the hidden practice fault, and Chaos Day.
Do not use a single validation screenshot as a substitute for triage
policy.

## Monitoring and revisit conditions

Specify what would force a revisit: a new data modality, mixed precision,
distributed training, a non-CPU device, or evidence that the sanity order
systematically missed a failure family.
