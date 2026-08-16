# Controlled failures — stagnation and outward oscillation

## Prediction before action

Before either run, record a loss-curve sketch, the expected sign pattern for the
high-curvature coordinate, and a numeric claim that could prove the prediction wrong.

## Failure A: too-small learning rate

Run exact gradient descent from the reference initialization with `learning_rate =
1e-5` for the fixed step budget. The code, objective, and gradient are correct; the
controlled cause is update scale. Diagnose stagnation from relative progress and
update magnitudes rather than from an arbitrary single loss threshold.

Small progress is not zero progress. The repair must preserve the objective,
initialization, gradient, and budget, then choose a rate supported by the sweep.

## Failure B: too-large learning rate

Run the same experiment with `learning_rate = 0.21`. The high-curvature coordinate
multiplier is `1 - learning_rate * 10`, so the parameter crosses zero while its
magnitude expands. The controlled cause is the learning rate crossing the stability
boundary; do not blame the gradient inherited from M19.

## Evidence and smallest repair

1. Compare each failure with the stable run from the identical start.
2. Trace `gradient → optimizer state → update → parameters → next loss`.
3. Count coordinate sign changes and compare successive magnitudes.
4. Name only the learning rate as the seeded root cause.
5. Restore a rate inside the observed stable region and rerun from the original start.
6. Verify the repaired trace with loss, parameter path, and update magnitude.

A repair is rejected if it changes curvature, initialization, step budget, loss
function, gradient, or optimizer to hide the failure.

