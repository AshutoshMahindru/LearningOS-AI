# M12 No-AI Gate — Defend an Ensemble Choice

Complete this gate without AI assistance and before rereading the notebook.

## Part A — Reconstruct the mechanisms

From memory, draw three training diagrams:

- bootstrap sampling followed by parallel tree fitting and averaged votes;
- random-forest row and feature randomization;
- boosting stages in which the next weak learner responds to the current ensemble's errors or loss gradient.

Annotate what can run in parallel, what must be sequential, and what state is aggregated.

## Part B — Explain bias and variance carefully

Explain aloud how tree depth can affect approximation error and sample sensitivity. Then explain why disagreement among resampled models is useful evidence but is not, by itself, a formal bias/variance decomposition.

## Part C — Fresh decision

Without using the M12 fixture, choose an approach for a tabular classification service with a strict latency budget, retraining every day, noisy labels, and a requirement to explain individual decisions. State:

1. the baseline you would preserve;
2. the ensemble you would trial first;
3. the estimator-count and depth ranges you would test;
4. the metrics and uncertainty checks you would use;
5. the condition that would make you reject the ensemble.

## Pass standard

Pass only if the learner can distinguish parallel averaging from sequential correction, reason from held-out evidence, challenge “more trees always fixes it,” and make trade-offs explicit without AI-generated prose.
