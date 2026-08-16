# M10 Controlled Failure — Optimize Accuracy, Harm Utility

## Setup

The validation data contains 6 imminent failures and 24 non-failures. Missing a failure costs 18 units; an unnecessary inspection costs 2 units. At most half the validation population can be alerted under the exercise's capacity assumption.

The deliberate mistake is:

> Sweep thresholds, select the one with the highest validation accuracy, report the attractive percentage, and omit the confusion matrix and consequence table.

## Prediction before action

Before running the selection cell, predict:

- whether accuracy will favor a high or low threshold;
- which confusion outcome the selected threshold will concentrate;
- whether the most accurate feasible threshold will minimize expected cost.

## Diagnosis protocol

1. Record the threshold selected by accuracy and its TP, FP, TN and FN.
2. Translate FP and FN into operational outcomes.
3. Compute `2 × FP + 18 × FN`.
4. Select the lowest-cost threshold subject to the capacity limit.
5. Compare accuracy, recall, alert rate, expected cost, and value versus no alerts.
6. State the root cause: the optimized objective assigned equal value to every correct row and equal penalty to every incorrect row, contrary to the decision contract.
7. Repair selection by making the cost matrix and constraints explicit.
8. Verify the repair on validation data, lock it, and only then inspect test data.

## Guard against a superficial repair

Replacing accuracy with F1 is not automatically sufficient. F1 excludes TN and balances precision and recall symmetrically. The learner must compare the F1-selected threshold with consequence-selected utility and explain any difference.

## Prevention

- Require a consequence table and confusion matrix beside any threshold metric.
- Version costs, constraints, dataset window, threshold, tie-breaks, and owner in an ADR.
- Monitor outcome prevalence, FP/FN counts, action volume, and realized cost after deployment.
- Revisit the decision when costs, capacity, calibration, prevalence, or score distribution changes materially.
