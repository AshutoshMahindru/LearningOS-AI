# Controlled failure — independent encoding drift

The lab deliberately calls `pandas.get_dummies` independently on the training
frame and on an inference frame containing unseen categories. The two calls
produce different feature columns. The already-fitted estimator then rejects
the inference matrix because its feature names and width no longer match the
matrix used during fit.

This is a controlled failure: the exception is expected, captured and
inspected; Restart + Run All still succeeds.

Required diagnostic sequence:

1. record a prediction before running the failing inference call;
2. capture the exception type and message;
3. compare training and inference feature names and shapes;
4. state whether the defect is in the estimator or preprocessing boundary;
5. identify the independent `get_dummies` calls as the smallest root cause;
6. replace the duplicated inference transformation with the fitted pipeline;
7. retry the unseen-category row;
8. verify that training and inference transformations have identical width;
9. explain why `handle_unknown="ignore"` preserves shape without inventing a
   learned coefficient for the unseen category.

Do not “repair” the failure by deleting the new category, editing the inference
row, or fitting preprocessing again on inference data. Those actions hide the
boundary defect or leak future information.
