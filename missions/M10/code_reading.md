# M10 Code Reading — Follow the Decision Contract

Read the notebook as an executable decision policy, not as a collection of formulas.

## Pass 1 — Locate consequences before metrics

Trace `load_consequence_costs(...)` and the capacity constant. Record:

1. which rows define TP, FP, TN and FN;
2. which assumptions are empirical and which are policy inputs;
3. what units the costs use;
4. what important effects the toy cost matrix omits.

Explain why a metric cannot decide whether 2 cost units for FP and 18 for FN are appropriate.

## Pass 2 — Verify confusion orientation

Follow `confusion_counts(...)` for one positive and one negative record on each side of a threshold. Explain the `score >= threshold` boundary and verify that actual labels, predictions, TP/FP/TN/FN, and the action meaning are not transposed.

## Pass 3 — Derive every metric

Follow `metrics_from_counts(...)`. Recompute accuracy, precision, recall, specificity and F1 by hand from the threshold-0.50 counts. Identify every possible zero denominator and the notebook's explicit convention.

## Pass 4 — Trace threshold selection

Follow `threshold_table(...)`, `accuracy_choice`, `f1_choice`, and `utility_choice`.

- Which split supplies the candidate thresholds?
- Which tie-break rule is used?
- Where is the capacity constraint applied?
- Why is the test split absent from selection?
- Which selected row would change if FN cost changed?

## Pass 5 — Separate ranking from action

Follow `roc_auc_pairwise(...)` and `average_precision(...)`. Explain why these values summarize ranking across thresholds but do not deploy an action threshold or encode the stated cost matrix.

## Pass 6 — Find the controlled failure

Trace the assertion comparing accuracy-selected cost with consequence-selected cost. Explain why this is a failure of objective alignment rather than a bug in the accuracy formula.

## Transfer prompt

An unfamiliar fraud model has a strong ROC AUC, weak alert precision, a fixed analyst queue, and losses that differ by transaction value. Identify the minimum code and evidence needed to select a defensible threshold without retraining the model.
