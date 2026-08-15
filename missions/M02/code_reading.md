# M02 Code Reading — Find the Learning Boundaries

Read the baseline cells as a data-flow trace. Do not start by explaining logistic regression internals.

| Boundary | Exact call | Input | Output/state | Integrity question |
|---|---|---|---|---|
| Split | `train_test_split(...)` | `X`, `y` | train/test rows and labels | Are class proportions preserved and row sets disjoint? |
| Fit | `baseline_model.fit(X_train, y_train)` | training features and labels | fitted scaler and classifier state | Did test labels or test-derived preprocessing enter fitting? |
| Prediction | `baseline_model.predict(X_test)` | held-out features | one predicted class per row | Does prediction require `y_test`? It must not. |
| Evaluation | `accuracy_score(y_test, baseline_predictions)` | held-out truth and predictions | scalar score | Are truth and predictions distinct, aligned arrays? |

## Reading task

1. Mark the first line where model state can change.
2. Mark the line where the fitted state is consumed without new labels.
3. Mark the line where truth is allowed to re-enter.
4. Explain why `accuracy_score(predictions, predictions)` is executable but invalid evidence.
5. Trace one misclassified row from raw CSV values through selected features to its predicted and true labels.

A strong explanation names inputs, outputs, learned state, and forbidden information flow at each boundary. A weak explanation merely recites function names.
