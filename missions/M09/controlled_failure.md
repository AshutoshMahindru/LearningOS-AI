# M09 controlled failure — Accurate enough, harmful by default

The seeded failure is a decision process, not a syntax error:

> The team accepts threshold `0.50` because it is the library-style default and reports only accuracy. It does not compare the majority baseline, inspect false negatives, or declare the consequences of a missed disengagement.

On this imbalanced holdout, a high accuracy can coexist with low recall because predicting the majority negative class correctly is comparatively easy. Another threshold can even have the same rounded accuracy while producing a materially different mix of false positives and false negatives.

## Diagnose before changing anything

1. Compute the all-negative baseline confusion matrix and accuracy.
2. Compute TP, TN, FP and FN at `0.50`.
3. State which cell represents a learner who disengages but receives no proactive outreach.
4. Compare accuracy, precision and recall at `0.20`, `0.30`, `0.50` and `0.70`.
5. Identify any thresholds with similar accuracy but different recall.
6. Declare a false-positive cost and a false-negative cost before choosing a threshold.
7. Compare total consequence cost for every candidate threshold.
8. Recommend a threshold and name the operational trade-off it accepts.
9. State what additional evidence would be required before a real deployment.

## Repair standard

The repair is not “always use a lower threshold.” The repair is to treat threshold selection as an explicit, reviewable policy based on held-out evidence and stated consequences. A good diagnosis also notes that probability quality, capacity, distribution shift, group-level impact, and calibration would require deeper validation outside this toy mission.
