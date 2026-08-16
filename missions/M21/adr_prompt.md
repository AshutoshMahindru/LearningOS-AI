# M21 ADR Prompt — Black-Box Neural Training Acceptance Policy

Use `templates/ADR.md`. The decision is not pre-selected. M21 needs a policy for deciding
whether a black-box neural training run is acceptable evidence before M22 opens the
network's internals.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Acceptance metrics and thresholds:** [UNFILLED BY LEARNER]
- **Seed and replay policy:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Monitoring:** [UNFILLED BY LEARNER]
- **Rollback triggers:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Define the acceptance gate for a V05 black-box neural run: baseline improvement, held-out
metrics, allowed training/validation evidence, seed replay expectations, tolerated seed
sensitivity, confusion-matrix review, and conditions that block progression to M22.

## Alternatives that must be compared

1. Accept primarily on one held-out headline metric.
2. Require baseline + accuracy + macro F1 + learning/validation curves + error profile.
3. Require the multi-signal package plus exact seed replay and a second-seed sensitivity run.

## Evidence required

Use the reference run, majority baseline, loss and validation traces, same-seed replay,
different-seed sensitivity, tiny-capacity comparison, undertraining failure and repair,
shuffled-label failure and repair, confusion-matrix error analysis, and no-AI transfer.

## Monitoring, rollback, and revisit conditions

Specify what blocks acceptance for near-baseline performance, non-finite metrics, severe
seed sensitivity, stalled loss, degrading validation evidence, target-integrity failure,
changed dataset/split policy, or a materially different compute budget. State the known
safe configuration to restore and who can approve progression to M22.
