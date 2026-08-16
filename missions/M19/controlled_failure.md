# Controlled failure — wrong gradient sign

## Prediction

Before running the failure, record whether `weight + learning_rate * gradient` should move uphill or downhill from `weight = 1.0`, and predict the next loss.

## Seeded root cause

The faulty update adds the gradient instead of subtracting it. All earlier stages—parameter, prediction, loss, and gradient—remain unchanged so the failure has one controlled cause.

## Evidence and repair

1. Compare correct and faulty runs from the same starting weight.
2. Trace the first step as `parameter → prediction → loss → gradient → update`.
3. Check the gradient sign against the plotted loss curve.
4. Identify the update operator as the smallest faulty unit.
5. Replace `+` with `-` and rerun from the original weight.
6. Verify that the repaired first step lowers loss and that repeated repaired steps approach the minimum.

A repair is not accepted if it changes the data, learning rate, starting point, or loss function to hide the failure.
