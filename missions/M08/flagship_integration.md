# V02 Flagship Integration

M08 establishes the regression layer of the **V02 Predictive Decision System**.

The mission contributes a reusable prediction contract:

- declare the decision and prediction timestamp;
- separate prediction-time features from the continuous target;
- compare against a baseline;
- fit with deterministic controls;
- evaluate held-out predictions with MAE, RMSE and R²;
- diagnose residual behavior and generalization variability;
- reject unavailable features regardless of score;
- preserve model and evaluation choices in an ADR.

M09 extends this contract to categorical outcomes. M10 adds consequence-aware metric and threshold decisions. The M08 output should therefore expose evaluation evidence and feature-availability controls that later V02 components can reuse rather than only a fitted estimator.
