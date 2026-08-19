# No-AI gate — defend a neural training run from whole-system evidence

Complete this gate from a blank page without AI-generated code, calculations, prose, or
diagrams.

## Part A: interpret a learning run

Given a table containing majority baseline, training accuracy, validation-score history,
training-loss history, held-out accuracy, macro F1, and a 10x10 confusion matrix:

1. decide whether the run has learned useful signal beyond baseline;
2. identify two observations that support or weaken a generalization claim;
3. state the most-confused true/predicted class pair from the matrix;
4. name evidence you still need before accepting the run for the flagship.

## Part B: isolate a controlled failure

You receive two runs with identical split, architecture knobs, optimizer settings, and
seed. One has `max_iter=1`; the other uses the declared reference budget. Predict which
observable traces differ and explain the smallest valid repair without referring to
neurons, gradients, or backpropagation.

## Part C: data integrity transfer

A new run falls near majority baseline after training. Propose checks that distinguish a
bad target mapping from ordinary seed sensitivity or insufficient budget. State a rollback
trigger and the evidence required before resuming training.

Pass requires independent reasoning, explicit fit/test boundaries, quantitative evidence,
and an oral defense. Leave all learner responses unfilled in the repository.
