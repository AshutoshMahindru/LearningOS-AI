# M10 ADR Prompt — Consequential Metric and Threshold Policy

Create a new ADR using `templates/ADR.md`. Do not copy a threshold merely because the notebook produced it; reconstruct the decision from the evidence available at decision time.

## Decision

Specify:

- positive event and triggered action;
- primary operating objective and supporting metrics;
- locked threshold and exact comparison rule (`score >= threshold` or otherwise);
- validation window and candidate-search method;
- tie-break rule and capacity constraint;
- owner and effective version.

## Context

Describe asymmetric FP/FN consequences before metric definitions. State population, prevalence, action capacity, outcome delay, cost units, and whether scores are calibrated probabilities or only ranking signals.

## Alternatives considered

At minimum compare:

- default threshold 0.50;
- maximize accuracy;
- maximize F1;
- satisfy a recall or precision floor;
- minimize expected cost/value subject to capacity;
- no automated alert policy.

## Evidence

Include validation confusion matrices and action volumes for material alternatives, the all-negative baseline, cost/value calculations, sensitivity analysis, and the untouched test result after the threshold was locked. Separate evidence from assumptions.

## Trade-offs and consequences

Explain who bears FP and FN harm, what F1/ROC/PR omit, operational load, distribution-shift exposure, calibration limitations, subgroup risks, delayed feedback, and failure/rollback behavior.

## Monitoring

Name owners, cadence, and thresholds for at least:

- prevalence and score-distribution shift;
- TP, FP, TN, FN when outcomes mature;
- recall, precision, specificity and action rate;
- realized consequence cost/value;
- capacity breaches and stale labels.

## Revisit conditions

Use quantitative triggers: for example, a material cost-ratio change, capacity change, prevalence or score drift beyond a declared bound, recall below a safety floor, or enough mature outcomes to overturn the validation ranking.

## Status

Choose Proposed, Accepted, Superseded, or Rejected and include date and decision owner. A notebook result alone does not make an ADR Accepted.
