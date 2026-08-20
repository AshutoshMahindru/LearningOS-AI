# M25 ADR Prompt — V05 Training-Loop Reproducibility / Checkpoint Policy

Use `templates/ADR.md`. The decision is not pre-selected. M25 needs a
policy for how V05 seeds, saves, resumes, and rolls back a training run
before M26 treats the loop as a breakable reference.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Seeds covered:** [UNFILLED BY LEARNER]
- **Saved state:** [UNFILLED BY LEARNER]
- **Optimizer state:** [UNFILLED BY LEARNER]
- **Validation evidence required to accept a checkpoint:** [UNFILLED BY LEARNER]
- **Restart expectations:** [UNFILLED BY LEARNER]
- **Rollback rule:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose how V05 declares a training run reproducible and resumable. CPU
is canonical. Do **not** claim the choice is globally optimal for every
later model, GPU run, or distributed worker.

## Alternatives that must be compared

1. Save only `model.state_dict()` weights. On resume, reseed from scratch
   and rebuild a fresh optimizer. Held-out inference is possible; exact
   continuation of SGD/momentum is not.
2. Save model weights, optimizer state, epoch, split seeds, architecture
   widths, batch size, learning rate, and CPU RNG. Resume is exact on the
   canonical CPU path. Held-out evaluation loads this payload, calls
   `eval()` / `no_grad`, and never `step`. (Proposed default to discuss.)
3. Save the entire pickled `nn.Module`, DataLoader iterator position, and
   every device RNG (including CUDA/MPS). Fail closed if any piece is
   missing. Highest fidelity, highest coupling to class paths and
   hardware.

## Evidence required

Use the teaching autograd check, the two-step reset experiment, the
Dropout train-mode evaluation, the disjoint-split fixture, and the
checkpoint round-trip on held-out logits. Do not use M21 headline
accuracy as a substitute for a loop-reproducibility policy.

## Monitoring and revisit conditions

Specify what would force a revisit: a non-CPU device, a fused optimizer,
changing Dropout at eval time, mixed precision, a DataLoader with
`num_workers>0`, or evidence that saving weights alone hid a momentum
buffer that later training needed.
