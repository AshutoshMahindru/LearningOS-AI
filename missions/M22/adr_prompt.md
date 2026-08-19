# M22 ADR Prompt — Teaching Activation and Width Policy

Use `templates/ADR.md`. The decision is not pre-selected. M22 needs a
policy for the V05 **teaching** component: which activation and which
hidden width to keep as the default explanation surface before M23
builds a multi-layer forward pass.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Activation choice and why:** [UNFILLED BY LEARNER]
- **Width choice and why:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a teaching default for V05 (for example ReLU at the M21 width, or
a smaller width used only for hand calculation). Do **not** claim the
choice is globally optimal.

## Alternatives that must be compared

1. Keep M21's ReLU / 64-unit knob as the teaching default.
2. Teach with a 2-4 unit layer that is fully hand-computable, and treat 64 as a later scale-up.
3. Prefer sigmoid or tanh as the teaching default because the maps are smooth.

## Evidence required

Use the reference neuron, bias ablation, activation sweep, batch shape
trace, linearity collapse, and the repaired layer-boundary failure.
Do not use M21 headline accuracy as a causal argument for ReLU.

## Monitoring and revisit conditions

Specify what would force a revisit: a later mission needing a different
convention, a documented shape-contract change, or evidence that the
teaching width hides the collapse experiment.
