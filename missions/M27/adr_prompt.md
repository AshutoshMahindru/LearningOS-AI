# M27 ADR Prompt — V06 Tokenization, Version, and Context-Budget Policy

Use `templates/ADR.md`. The decision is not pre-selected. M27 needs a
policy for the V06 **teaching** component: which tokenizer identity to
treat as canonical, how specials and truncation are counted, what is
logged, and what forces a migration before M28 attaches vectors.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Tokenizer identity and version:** [UNFILLED BY LEARNER]
- **Special-token handling:** [UNFILLED BY LEARNER]
- **Truncation and budget unit:** [UNFILLED BY LEARNER]
- **Logging:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V06 default for teaching tokenization (for example BPE as the
model-like scheme with word lookup as the contrast, or word lookup as
the explanation default). Do **not** claim the choice is a production
LLM tokenizer.

## Alternatives that must be compared

1. Canonicalize `v06-teaching-bpe` v06.1; keep word lookup as a contrast only.
2. Canonicalize `v06-teaching-word` because it matches human words on the teaching corpus.
3. Count context in characters or words "for simplicity" and treat tokenizer length as an implementation detail.

## Evidence required

Use the surface-variation traces, rare-string decomposition, padding
masks, truncation suffix, scheme comparison, and the repaired
token-budget failure. Do not use a downloaded Hugging Face or tiktoken
encoding as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a vocab or special-token change, a
truncation-side change, a new tokenizer version, or any pipeline that
budgets context in characters or words.
