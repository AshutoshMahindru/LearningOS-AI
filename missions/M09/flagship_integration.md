# M09 → V02 Predictive Decision System

M09 adds the binary-decision path to V02. It turns a held-out feature row into a probability, then makes the threshold and consequence policy visible rather than treating `predict()` as an unexplained final answer.

## Contribution to V02

- a binary target contract;
- a deterministic train/holdout boundary;
- a probability-producing classifier;
- an explicit probability-to-class policy;
- confusion-matrix and accuracy/precision/recall diagnostics;
- an imbalance check against the majority baseline;
- threshold comparison without model refitting;
- consequence-aware selection and calibration intuition;
- a learner-authored ADR governing the selected threshold policy.

## Hand-off to M10

M10 deepens the metric and cost layer. M09 should hand over the same held-out probabilities, targets, candidate thresholds, and TP/TN/FP/FN counts so the next mission can examine richer decision metrics without quietly changing the model or evaluation population. It must also carry the learner-authored threshold ADR: selected threshold and comparison rule, FP/FN cost assumptions, operating capacity, alternatives, accepted trade-offs, owner/status/date, monitoring, rollback and revisit triggers.

## V02 integration check

The flagship must make these two components separable:

1. **probability model:** features → estimated probability;
2. **decision policy:** probability + threshold + consequences/capacity → action.

If a threshold changes, V02 should be able to show changed decisions while confirming that the learned coefficients and held-out probabilities did not change.

V02 must not treat the notebook's lowest toy cost as approval. The operating policy is governed only when its ADR connects the choice to evidence and capacity, records alternatives and trade-offs, and defines measurable conditions for review or rollback.
