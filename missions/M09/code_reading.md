# M09 code reading — Find the policy boundary

Before executing the classifier code, trace these functions in the notebook:

1. `fit_logistic_classifier`
2. `predict_probabilities`
3. `classify`
4. `confusion_counts`
5. `metric_summary`

For each function, identify its inputs, output, learned state, and whether changing a threshold can change it.

Then answer before running:

- Which function learns from target labels?
- Which function produces continuous scores?
- Which function turns a score into an action-oriented class?
- Why can the confusion matrix change without retraining the classifier?
- In what order does `confusion_counts` report TN, FP, FN and TP?
- Which denominator belongs to precision, and which belongs to recall?

Trace one held-out probability through two thresholds. Record the class at each threshold and name the confusion-matrix cell after comparing it with the true label. The point is to locate the model/policy boundary in executable code, not to memorize metric formulas in isolation.
