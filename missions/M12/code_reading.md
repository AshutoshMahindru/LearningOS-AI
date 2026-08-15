# M12 Code Reading — Follow the Ensemble State

## Pass 1 — The baseline

Locate `limited_tree`. Record its maximum depth, the split it sees, and the metrics produced by `metric_row`. Explain what is deliberately limited and what evidence would distinguish high bias from high variance.

## Pass 2 — Bootstrap variation

Trace one iteration of `fit_resampled_trees`. Identify the sampled row indices, the fitted tree state, and the held-out prediction vector. Then explain the two aggregation axes in `bootstrap_predictions` and why disagreement is only an empirical sensitivity proxy.

## Pass 3 — Parallel averaging

Compare `BaggingClassifier` and `RandomForestClassifier`. Find where each receives row randomness and where the forest receives feature randomness. Trace how many fitted estimators contribute to one prediction. Explain why independent fitting permits parallelism.

## Pass 4 — Sequential correction

Trace `GradientBoostingClassifier.staged_predict`. For two consecutive stages, locate corrected cases and newly wrong cases. Explain why stage `k + 1` depends on the ensemble at stage `k`, even though the public API hides residual or gradient bookkeeping.

## Pass 5 — Capacity controls

Follow `run_size_sweep` and `run_depth_sweep`. List what is fixed and what changes. Check whether comparisons use the same held-out records and metric. Identify why estimator count cannot be interpreted independently from depth and learning rate.

## Transfer prompt

A production team proposes 2,000 unpruned trees because validation accuracy rose between 10 and 100 trees. Identify the missing evidence, the computational trade-offs, and the stopping or revisit criteria you would require before approving the change.
