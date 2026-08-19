# Controlled failures — undertraining and corrupted targets

## Failure A: one-iteration undertraining

Before running, predict the training loss and held-out accuracy relative to the reference
run. Change only `max_iter` from the declared reference budget to `1`. The code path,
data split, preprocessing, architecture knobs, learning rate, optimizer family, and seed
remain fixed.

Diagnose the symptom from the one-point loss curve, low held-out performance, and the
known training-budget change. The smallest repair restores the reference budget and reruns
from the original split and seed. Do not change hidden units or labels to hide the failure.

## Failure B: shuffled training labels

Before running, predict held-out behavior when the training targets are permuted with the
recorded label seed while test labels remain correct. Change only the mapping between
training examples and training labels.

Diagnose the failure as a broken target contract. A larger network, different model seed,
or longer budget is not an acceptable repair. Restore the original training labels and
rerun the same reference configuration.

## Evidence and repair rule

For each failure submit prediction, changed variable, preserved invariants, observed
metrics, root cause, smallest repair, and repaired rerun. A repair is rejected if it opens
neural internals or changes several variables at once.
