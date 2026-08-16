# M20 ADR Prompt — Optimizer and Learning-Rate Policy

This ADR is consequential because M21 will inherit its training behavior, monitoring,
and failure response. Use `templates/ADR.md`. Compare the required alternatives and
complete every field with learner-produced evidence; do not treat this prompt as a
pre-selected answer.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Chosen optimizer and learning-rate regime:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Monitoring:** [UNFILLED BY LEARNER]
- **Rollback triggers:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose the initial optimizer and learning-rate regime for M21's black-box neural-network
training, including how the system detects poor dynamics and responds safely.

## Alternatives that must be compared

1. Exact/full-batch gradient descent with a fixed learning rate.
2. Seeded mini-batch SGD with a fixed rate or explicit schedule.
3. Classical momentum with declared coefficient and learning-rate policy.
4. Adam with declared betas, epsilon, learning rate, and any weight-decay boundary.

The comparison must address objective and horizon dependence, compute/memory cost,
noise, overshoot, coordinate scaling, tuning burden, reproducibility, interpretability,
and the risk of making a fixture-specific optimizer ranking universal.

## Evidence required

Use the controlled learning-rate sweep, stability-boundary calculation, GD/SGD trace,
momentum and Adam state traces, at least one seed-sensitivity comparison, both controlled
failures and repairs, and the no-AI transfer defense. Clearly separate deterministic
fixture evidence from expectations about M21.

## Monitoring, rollback, and revisit conditions

Define monitored loss and update signals, observation window, threshold owner, and
response. Include triggers for non-finite values, outward oscillation, stalled progress,
unstable seed sensitivity, changed curvature/scale, changed batch regime, different
compute budget, new generalization evidence, and any M21 incident. State what rolls
back, to which known-safe regime, and who may approve resumption.

