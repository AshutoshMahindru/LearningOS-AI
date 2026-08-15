# M10 synthetic decision datasets

All files are deterministic, synthetic teaching fixtures. They do not represent real equipment reliability, food safety, monetary value, or production model performance.

## `asset_alert_scores.csv`

Sixty fixed risk scores for an asset-inspection scenario:

- 30 validation cases used for threshold selection;
- 30 test cases held back until the threshold is locked;
- 6 positive outcomes per split (20% prevalence);
- `failure_within_30d=1` is the positive event;
- `risk_score >= threshold` triggers an inspection.

The file contains no training split because M10 evaluates the decision layer of an already-scored classifier. The exercise does not claim that `risk_score` is calibrated probability.

## `consequence_matrix.csv`

The bounded teaching assumptions assign 2 cost units to an unnecessary inspection (FP), 18 to a missed imminent failure (FN), and zero marginal cost to TP/TN. Real deployments must include action, delay, heterogeneous loss, equity, and capacity effects as appropriate.

## `unseen_threshold_evidence.csv`

Aggregate validation evidence for the independent no-AI food-allergen audit scenario. It intentionally includes neither cost columns nor a selected/recommended threshold. The learner must apply the unseen consequences and capacity constraint in `missions/M10/no_ai_gate.md`.
