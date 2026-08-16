# M12 → V03 Flagship Integration

M12 supplies V03 with an evidence-backed ensemble-selection workflow for tabular tree models. The contribution is not a hard-coded winning algorithm; it is a repeatable comparison that begins with a limited baseline, exposes instability, and treats model capacity as an engineering decision.

## What M12 contributes

- a fixed, reproducible baseline and held-out evaluation path;
- an empirical view of prediction variation under bootstrap resampling;
- comparable bagging, random-forest, and boosting implementations;
- estimator-count and tree-depth sensitivity evidence;
- a controlled failure demonstrating the limits of added capacity;
- an ADR and formal-review boundary before a consequential ensemble choice.

## V03 integration check

Before V03 adopts an ensemble, the learner must state the workload constraints, choose the metric and validation design, compare against the limited tree, quantify diminishing returns, and define a revisit trigger for drift, latency, or label-quality changes.
