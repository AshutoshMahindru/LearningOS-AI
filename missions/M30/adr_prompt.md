# M30 ADR Prompt — V06 Teaching Transformer-Block Convention

Use `templates/ADR.md`. The decision is not pre-selected. M30 needs a
policy for the V06 teaching component M31 will inherit: pre-norm versus
post-norm placement, activation, head dimensions, checkpointed
intermediates, and version identity.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Norm placement (pre / post):** [UNFILLED BY LEARNER]
- **Activation:** [UNFILLED BY LEARNER]
- **Head dimensions (`d_model`, `n_heads`, `d_head`, `d_ff`):** [UNFILLED BY LEARNER]
- **Checkpointed tensors:** [UNFILLED BY LEARNER]
- **Version identity:** [UNFILLED BY LEARNER]
- **Head-interpretation limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V06 default for the teaching transformer block (norm
placement, activation, which tensors are checkpointed, how heads may
be talked about). Do **not** claim the choice is a production LLM, and
do not open an M31 trainer.

## Alternatives that must be compared

1. Declare pre-norm (`x + MHA(LN(x))`, then `h + FFN(LN(h))`), ReLU,
   `n_heads=2`, checkpoint the TRACE_CHECKPOINTS list, version the
   fixture, and refuse head mythology.
2. Declare post-norm as the teaching default because the original
   transformer paper used it.
3. Skip named residuals and log only the block output so traces stay
   small.

## Evidence required

Use the cash/water whole-block run, the head split, the residual
ablation, the labeled pre/post comparison, the golden parity table,
and the repaired residual/norm failure. Do not use a pretrained
transformer as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a stack of blocks under a training
objective (M31), a change to RMSNorm, a different head width, or any
pipeline that treats a head as a linguistic role.
