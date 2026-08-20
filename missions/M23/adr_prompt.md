# M23 ADR Prompt — V05 Inference-Parity and Numerical-Tolerance Policy

Use `templates/ADR.md`. The decision is not pre-selected. M23 needs a
policy for how V05 compares a NumPy forward pass to a trusted reference
before M24 treats that graph as the starting point for credit assignment.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Intermediates compared and why:** [UNFILLED BY LEARNER]
- **Dtype assumption:** [UNFILLED BY LEARNER]
- **Tolerance (rtol / atol):** [UNFILLED BY LEARNER]
- **Failure handling when parity breaks:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose how V05 declares two forward implementations the same. Compare
named intermediates, not only argmax. Do **not** claim the choice is
globally optimal for every later model.

## Alternatives that must be compared

1. Require bit-identical `float64` agreement (`rtol=0`, `atol=0`) on every named array.
2. Require `float64` with a small absolute tolerance (for example `atol=1e-12`) on intermediates and probabilities, and treat argmax as a secondary check.
3. Compare only final class predictions / argmax, and treat intermediate drift as out of scope.

## Evidence required

Use scalar-to-vectorized parity, M22 reference parity, softmax shift
invariance, batch reordering, and the repaired wrong-axis failure.
Do not use M21 headline accuracy as a substitute for intermediate
parity.

## Monitoring and revisit conditions

Specify what would force a revisit: a later dtype change, a fused
kernel that drops intermediates, a documented softmax-axis convention
change, or evidence that argmax-only checks hid a wrong-axis defect.
