# M12 ADR Prompt — Select an Ensemble Trial

Write a new ADR using `templates/ADR.md`. Do not copy a model ranking from the notebook; use your executed evidence and an explicit operating context.

## Decision to make

Choose one of these for the next V03 validation trial:

- retain the limited tree;
- bagged trees;
- random forest;
- gradient-boosted trees.

Specify estimator count, tree depth, learning rate when applicable, random-seed policy, validation design, and the metric threshold that must be met.

## Context constraints to state

Define the expected training volume, prediction latency budget, model-size budget, retraining cadence, interpretability requirement, class-cost assumptions, label-quality risk, and available parallelism. If a value is unknown, record it as uncertainty rather than inventing it.

## Evidence required

Reference the limited baseline, bootstrap disagreement, held-out comparison, sequential-correction trace, estimator-count curve, depth sweep, and controlled failure. Separate facts observed on the synthetic fixture from inferences about V03.

## Alternatives and trade-offs

For every rejected alternative, state one credible advantage and the evidence or constraint that makes it less suitable. Discuss accuracy, generalization gap, stability, training cost, inference cost, memory, parallelism, explainability, and sensitivity to noisy labels.

## Revisit conditions

Include measurable triggers such as drift, class-balance change, latency regression, model-size growth, label-quality degradation, confidence-interval overlap, or failure to reproduce gains under cross-validation. Give an owner and review date in the learner-authored ADR.

## Status

Leave the ADR `Proposed` until the formal engineering review in `review_brief.md` is complete.
