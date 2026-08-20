# M29 ADR Prompt — V06 Attention-Trace Observability Policy

Use `templates/ADR.md`. The decision is not pre-selected. M29 needs a
policy for the V06 teaching component M30 will inherit: which tensors
are checkpointed, which summaries are logged, which invariants must
hold, and what a weight must not be claimed to prove.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Checkpointed tensors:** [UNFILLED BY LEARNER]
- **Shape contract:** [UNFILLED BY LEARNER]
- **Mask logging:** [UNFILLED BY LEARNER]
- **Score summaries:** [UNFILLED BY LEARNER]
- **Invariants (row-sum / masked mass / softmax axis):** [UNFILLED BY LEARNER]
- **Over-interpretation limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V06 default for teaching attention traces (what to log, which
invariants gate a trace as valid, how to talk about weights). Do **not**
claim the choice is a production profiler, and do not open an M30 block.

## Alternatives that must be compared

1. Checkpoint Q, K, V, raw scores, scaled scores, mask, weights, and
   output; log shapes, scale, row sums, masked mass, and a fixed
   "weights are not intent" banner; refuse a trace that breaks those
   invariants.
2. Log only the output tensor so the trace stays small.
3. Persist full weight matrices as a causal explanation of the model's
   "focus" for later training or search work.

## Evidence required

Use the cash/water context change, the causal mask, the scale
comparison, the value-only split, and the repaired softmax-axis
failure. Do not use a pretrained transformer as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a multi-head split (M30), a change
to scale, a new mask convention, a residual/norm boundary, or any
pipeline that treats weights as intent.
