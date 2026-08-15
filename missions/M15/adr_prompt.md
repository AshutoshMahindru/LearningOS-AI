# ADR prompt — choose a vector similarity contract for V04

Complete an ADR using `templates/ADR.md`. The decision is not “which metric is best?” It is “which invariant must this V04 measurement preserve for this use case?”

## Context to record

- What does each vector dimension mean?
- Does magnitude carry signal, noise or arbitrary scale?
- Is the application comparing direction, absolute closeness or magnitude-sensitive alignment?
- Are vectors guaranteed to be nonzero?
- Where and how is normalization applied?
- What evidence comes from the controlled disagreement fixture and the semantic ranking?

## Options to evaluate

1. cosine similarity on nonzero vectors;
2. Euclidean distance on original vectors;
3. Euclidean distance on L2-normalized vectors;
4. raw dot product when magnitude is intentionally part of the score.

## Decision record requirements

State the chosen representation, preprocessing, metric, ranking direction, zero-vector policy and deterministic tie policy. Include at least one rejected option and a consequence of rejection. Define an observable V04 measurement and a condition that would trigger revisiting the decision.

The ADR fails if it names a metric without connecting it to an intended invariant or if it silently assumes normalization.
