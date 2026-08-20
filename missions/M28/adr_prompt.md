# M28 ADR Prompt — V06/V08 Embedding Provenance and Compatibility Policy

Use `templates/ADR.md`. The decision is not pre-selected. M28 needs a
policy for the V06 teaching component and the contract M33 will inherit:
which embedding identity is canonical, how vectors are normalized,
which metric is declared, what is logged, and what refuses a mix.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Embedding model and version:** [UNFILLED BY LEARNER]
- **Normalization and pooling:** [UNFILLED BY LEARNER]
- **Similarity metric:** [UNFILLED BY LEARNER]
- **Compatibility / mix policy:** [UNFILLED BY LEARNER]
- **Logging:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V06 default for teaching retrieval (model identity, L2 vs none,
cosine vs inner product, refuse-vs-renormalize on mismatch). Do **not**
claim the choice is a production encoder.

## Alternatives that must be compared

1. Canonicalize `v06-teaching-meanpool` v06.1 with mean pooling, L2, and
   cosine; reject any differing model, version, width, metric, or
   normalization before scoring.
2. Accept any same-width vectors and L2-normalize at query time so
   stores can be mixed.
3. Score inner product on raw pooled vectors and treat normalization as
   an implementation detail.

## Evidence required

Use the paraphrase rankings, lexical/semantic disagreement, hard cases,
the L2-vs-sum inner-product comparison, and the repaired mixed-store
failure. Do not use a downloaded encoder as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a new model or version, a pooling
change, a metric change, a width change, or any pipeline that scores
across fingerprints.
