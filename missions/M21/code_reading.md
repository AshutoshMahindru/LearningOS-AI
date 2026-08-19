# Code reading — the training harness, not the neuron

Read `training_core.py` from `train_black_box` outward. M21's code-reading target is the
**orchestration contract** around a neural estimator:

1. dataset load and stratified train/test split
2. optional controlled training-label corruption
3. training-only preprocessing through a pipeline
4. declared estimator hyperparameters and model seed
5. fit call
6. train and held-out predictions
7. learning-curve, validation, accuracy, macro-F1, and confusion-matrix evidence

Trace one reference run with concrete configuration values. Name which data is allowed to
influence fitting and early stopping and which data is held out until evaluation.

Do **not** inspect `coefs_`, `intercepts_`, individual activations, manual matrix products,
gradients, or backpropagation. Those are the explicit M22-M24 descent boundary. If a
black-box failure can be diagnosed from split integrity, labels, budget, seed, loss,
validation, or held-out errors, stay at that level.
