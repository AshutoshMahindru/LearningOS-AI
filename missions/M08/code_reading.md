# M08 Code Reading

Before changing the notebook, spend 10–15 minutes tracing this path:

`CSV row → feature list → train/test split → fitted estimator state → predict(X_test) → residual vector → metric aggregation`

Record answers without running the code first:

1. Which column is the target, and which column is forbidden because it exists only after the sale?
2. Which rows can influence `model.fit(...)`?
3. Where is randomness controlled?
4. What shape enters `fit`, and what shape leaves `predict`?
5. What sign convention does `residual = actual - predicted` use?
6. Why are cross-validation folds drawn only from the training partition?
7. Which objects learn state and which functions only summarize predictions?
8. Where would a future preprocessing step need to live to avoid leakage?

Then trace the controlled-failure path and identify the earliest point where the prediction-time availability contract is violated.
