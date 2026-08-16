# M08 — Predict a Continuous Outcome

## Mission

Build a regression system that predicts a continuous housing sale price and earns trust through evaluation and diagnosis, not through one impressive score.

The whole-first path is:

**baseline → train/test split → fit → predict → metrics → residuals → diagnose**

Run that complete path before descending into model internals.

## System boundary

- **Input:** facts available before a sale, such as floor area, age and access scores.
- **Target:** `sale_price_k`, a continuous value in thousands of currency units.
- **Output:** one predicted sale price per held-out transaction.
- **Evaluation boundary:** model fitting and cross-validation happen only on the training partition; the test partition is reserved for final generalization evidence.
- **Availability boundary:** `post_sale_assessment_k` is recorded after the outcome and is forbidden in a deployable predictor.

The data is deterministic and generated locally. It is deliberately synthetic: results demonstrate regression mechanics and evaluation discipline, not real property-market performance.

## What the lab covers

- a mean-prediction baseline;
- a deterministic train/test split;
- a fitted random-forest regression pipeline;
- MAE, RMSE and R² with units and interpretation;
- predicted-versus-actual and residual plots;
- residual slices that reveal where errors concentrate;
- cross-validation on training data;
- shallow, moderate and unrestricted trees to expose underfitting and overfitting;
- held-out permutation importance interpreted cautiously;
- a controlled target-leakage failure;
- an invalid training-set evaluation that looks better than cross-validation;
- an explicit repair based on feature availability.

## Evaluation rules

1. Compare every candidate with the same simple baseline.
2. Never fit transformations or models on the test target.
3. Use MAE for typical error magnitude, RMSE to emphasize large misses and R² only relative to target variance.
4. Inspect residuals; a scalar metric cannot show systematic error.
5. Use cross-validation to estimate variability during model development.
6. Treat feature importance as model-dependent predictive influence, not causal proof.
7. Reject features unavailable at the moment a prediction must be made, even when they improve validation scores.

## Source policy

`sklearn-guide` is the authoritative implementation reference and `stanford-cs229` is the conceptual regression reference already registered in `data/source_registry.json`. The notebook has no runtime network dependency.

## Engineering controls

M08 requires a formal engineering review and a learner-authored ADR. The review must cover the architecture, evaluation boundary, failure analysis, decision evidence, uncertainty and operational guardrails. Learner completion evidence is intentionally not populated by this source package.

## V02 connection

M08 begins the V02 Predictive Decision System by establishing a continuous-outcome prediction and evaluation contract. M09 adds classification and M10 makes metric choices consequence-aware.
