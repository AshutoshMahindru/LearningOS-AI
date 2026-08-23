# M31 ADR Prompt — V07 Training-Data and Evaluation Provenance

Use `templates/ADR.md`. The decision is not pre-selected. M31 needs a
policy for the V07 artifact M32 will inherit: dataset versions, split
lineage, checkpoint identity, adaptation stage, and minimum audit
metadata.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Dataset version / corpus identity:** [UNFILLED BY LEARNER]
- **Split lineage (train/eval ids, leak rule):** [UNFILLED BY LEARNER]
- **Checkpoint identity:** [UNFILLED BY LEARNER]
- **Adaptation stage label:** [UNFILLED BY LEARNER]
- **Minimum audit metadata:** [UNFILLED BY LEARNER]
- **Teaching-scale limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V07 default for training-data and evaluation provenance
(what must be recorded on every checkpoint, when eval is invalid, how
adaptation stage is labeled). Do **not** claim the teaching table is a
production LLM, and do not open an M32 decoder.

## Alternatives that must be compared

1. Record dataset version, authored train/eval ids, split hash,
   checkpoint id, seed, steps, objective, adaptation stage, and a
   contamination flag; reject eval when train ∩ eval is non-empty.
2. Log only final train loss and a model filename so traces stay small.
3. Treat every new prompt as a new training stage.

## Evidence required

Use the pair construction, the tiny training run, the held-out NLL, the
leak lineage report, and the repaired shift/leak failure. Do not use a
vendor pretrain card as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: stacking real transformer blocks
under the same objective, an instruction-tuning stage, a preference
stage, a tokenizer version change, or any pipeline that samples tokens
(M32) without recording the checkpoint id.
