# M02 Controlled Failures

The notebook contains two seeded failures. They are expected to execute successfully; the failure is epistemic, not a Python exception.

## Failure A — corrupted training labels

A deterministic permutation breaks the relationship between each training row and its class while preserving label counts. Predict the likely held-out behavior before fitting. Diagnose with the permutation disagreement rate, class counts, the honest held-out score, and comparison with the intact-label baseline.

Root cause: the fitting call received semantically corrupted supervision. Repair: restore row-label alignment, refit from scratch, and reevaluate on untouched held-out truth. Do not “repair” the result by tuning until noise happens to score well.

## Failure B — invalid self-comparison evaluation

`accuracy_score(baseline_predictions, baseline_predictions)` returns `1.0`. The number is mathematically correct for its inputs but invalid as model-quality evidence because no ground-truth labels participate.

Root cause: the evaluation boundary compares predictions to themselves. Repair: compare `y_test` with predictions from the corresponding `X_test` rows and retain row alignment. Confirm the repaired score matches the recorded honest baseline.

## Required diagnosis

For each failure, record hypothesis, observed evidence, root cause, smallest repair, verification, and one monitoring check that could catch recurrence. “The score is bad” or “the score is perfect” is an observation, not a diagnosis.
