# M21 Implementation Review Brief

## Review decision requested

Approve M21 as the first V05 implementation layer: an offline, reproducible, black-box
neural-training mission that preserves M20's monitoring discipline while explicitly
deferring neuron/layer mechanisms to M22.

This is an implementation review, not learner sign-off and not the periodic formal
engineering review milestone.

## System under review

- bundled digits dataset with stratified held-out split
- training-only `StandardScaler` inside a scikit-learn pipeline
- one declared `MLPClassifier` reference configuration
- learning loss and validation-score histories
- majority baseline, train/test accuracy, macro F1, and confusion matrix
- controlled model-seed replay/sensitivity
- undertraining and shuffled-label failures
- tiny-capacity comparison

## Required reviewer challenges

- verify M21 is blocked by M20 and hands off to M22 without opening neuron internals;
- reproduce the reference run and confirm it materially beats majority baseline;
- verify same-seed replay is exact and a changed model seed changes the trace;
- reproduce both controlled failures and their smallest repairs;
- verify the held-out test set is not used for fitting or early stopping;
- search notebook and training harness for weight/activation/forward-pass/backprop inspection;
- verify source notebook IDs are unique, execution counts are null, outputs are empty, and execution is offline/CPU-only;
- confirm learner evidence, ADR decisions, scores, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds with zero code-cell errors;
- mission unittest and pytest pass in the M21 environment;
- repository validator and repository test suite remain green under the mission-only diff;
- reference held-out accuracy and macro F1 exceed a meaningful declared floor and majority baseline;
- one-iteration undertraining and shuffled-label corruption fail observably and repair by restoring only the named contract;
- no shared registry, tracking, root README, workflow, global validator, or lab-status file is changed.
