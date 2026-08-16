# M08 ADR Prompt — Regression Model and Evaluation Protocol

Create a learner-authored ADR after completing the experiments. Do not choose the model from a single metric, and do not copy notebook narration as the decision record.

## Decision

Which safe regression candidate and evaluation protocol should V02 adopt as its current baseline?

State:

- the selected model or explicit decision to retain the simple baseline;
- the permitted prediction-time feature allow-list;
- the train/test and training-only cross-validation protocol;
- the primary metric and the supporting diagnostics;
- the rule that excludes post-outcome features.

## Context

Describe the prediction moment, user or system affected, target units, dataset limitations and why the decision is consequential.

## Alternatives considered

Compare at least:

1. training-mean baseline;
2. capacity-controlled decision tree or linear regression;
3. random-forest pipeline;
4. the leaky high-scoring model, marked inadmissible.

## Evidence

Include a compact table with held-out MAE, RMSE and R², cross-validation MAE mean and variability, train/CV gap, residual observations and runtime. Distinguish observed evidence from interpretation.

## Trade-offs

Address accuracy, variability, interpretability, computation, maintenance, feature availability and the cost of large errors. Explain why permutation importance does not establish causality.

## Revisit conditions

Name measurable triggers such as material data drift, a persistent residual pattern, a defined MAE degradation, new prediction-time features, changed decision costs or representative temporal data becoming available.

## Status

Use one of: `proposed`, `accepted`, `superseded`, `rejected`.

## Required challenge questions

- What evidence could falsify this decision?
- Which result depends most on the synthetic data design?
- Why will the leakage control still work if a proxy column has a harmless name?
- What evaluation split would replace a random split for time-dependent deployment?
