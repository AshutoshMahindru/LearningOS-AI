# M24 ADR Prompt — V05 Gradient-Verification Policy

Use `templates/ADR.md`. The decision is not pre-selected. M24 needs a
policy for how V05 checks reverse-mode gradients before M25 treats them
as the autograd reference.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Parameters and cases checked:** [UNFILLED BY LEARNER]
- **Epsilon and stencil:** [UNFILLED BY LEARNER]
- **Tolerance (rtol / atol):** [UNFILLED BY LEARNER]
- **When checks run:** [UNFILLED BY LEARNER]
- **Failures that block release:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose how V05 declares a reverse-mode implementation trusted. Check
parameter gradients against central finite differences of the true loss.
Do **not** claim the choice is globally optimal for every later model.

## Alternatives that must be compared

1. Check every parameter entry on every batch, including ReLU hinges, with a tight absolute tolerance.
2. Check a declared subset (smooth entries, one active example, plus at least one dead ReLU unit and one branch) with `float64`, central epsilon near `1e-5`, and stated atol/rtol; treat `z = 0` as an excluded hinge.
3. Skip finite-difference checks and trust analytic reverse mode (or later autograd) alone.

## Evidence required

Use the scalar-chain check, branch-sum check, example-1 full-parameter
check, epsilon sweep, ReLU-at-zero hinge, omitted-branch isolation, and
the repaired defect. Do not use M21 headline accuracy as a substitute
for a gradient check.

## Monitoring and revisit conditions

Specify what would force a revisit: a fused kernel that drops
intermediates, a change of loss or reduction, mixed precision, a
documented ReLU-at-zero convention change, or evidence that
argmax-only or W2-only checks hid an omitted branch.
