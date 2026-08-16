# M10 — Make Metrics Reflect Consequences

## Mission objective

Choose evaluation metrics and a decision threshold from the consequences of action and inaction. The mission starts with the decision, not a dictionary of metrics:

> An asset-risk score triggers a manual inspection. A false positive spends scarce inspection capacity and costs 2 units; a false negative misses an imminent failure and costs 18 units.

The score is fixed. M10 changes only how it is evaluated and converted into an action. Validation evidence is used to select and lock a threshold; the test split is opened once to estimate the locked policy's behavior.

## Consequence-first route

1. Name the positive event and the action triggered by a positive prediction.
2. Write the TP, FP, TN and FN operational meanings before calculating a metric.
3. State cost assumptions and the inspection-capacity constraint.
4. Derive accuracy, precision, recall, specificity and F1 from the confusion matrix.
5. Compare ROC and precision-recall views of the same threshold sweep.
6. Expose the imbalanced all-negative baseline.
7. Deliberately optimize accuracy and observe the utility failure.
8. Select a threshold by empirical expected cost/value on validation data.
9. Lock the threshold, evaluate once on test data, and run cost-sensitivity checks.
10. Record the consequential decision in an ADR and defend it in formal review.
11. Pass the no-AI gate on a fresh scenario with different costs and capacity.

## Metric meanings in this mission

- **Accuracy** measures the fraction of all decisions that are correct. It can be dominated by the majority class.
- **Precision** answers: among issued alerts, what fraction were real imminent failures?
- **Recall (sensitivity)** answers: among real imminent failures, what fraction were alerted?
- **Specificity** answers: among non-failures, what fraction avoided an unnecessary alert?
- **F1** is the harmonic mean of precision and recall. It ignores true negatives and treats precision and recall symmetrically; neither property automatically matches decision value.
- **ROC points** compare recall with false-positive rate across thresholds. **PR points** expose the precision/recall trade-off and are often more revealing when positives are rare.
- **Expected cost/value** attaches explicit consequence weights to confusion-matrix outcomes. Its assumptions must be documented and revisited.

## Guardrails

- Never tune on the test split.
- Do not interpret a score as a calibrated probability without evidence.
- Do not hide capacity, latency, group-level error, or distribution-shift constraints inside one scalar metric.
- Report the confusion matrix with any aggregate metric.
- Treat threshold changes as versioned policy changes with owners and monitoring.

Learner evidence is defined in `evidence_contract.yaml` and is intentionally not prefilled.
