# M10 → V02 Flagship Integration

M10 completes the **Predictive Decision System** by converting the M09 classifier's continuous scores into an explicit operating policy.

## V02 decision boundary

```text
M09 fixed score
  → M10 versioned threshold policy
  → alert / no-alert action
  → observed FP and FN consequences
  → monitoring and review
```

The model score and the action policy are separate artifacts. Retraining may change score quality; M10 demonstrates that the threshold can also change when costs, capacity, prevalence, or policy changes, even while the model remains fixed.

## M10 contribution

- consequence-oriented definitions for positive class and action;
- confusion-matrix instrumentation at an operating threshold;
- accuracy, precision, recall, specificity and F1 with explicit limitations;
- ROC/PR threshold-sweep interpretation under imbalance;
- validation-only threshold selection using expected cost/value and capacity;
- a locked policy evaluated once on test evidence;
- an ADR, formal review, monitoring plan, and revisit conditions.

## V02 release check

V02 is not ready merely because regression and classification notebooks run. The release must name the deployed threshold, decision owner, cost/constraint version, validation window, tie-break rule, action volume, realized FP/FN monitoring, and rollback or revisit trigger.
