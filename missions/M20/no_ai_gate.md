# No-AI gate — optimizer traces from a blank page

Complete this gate without AI-generated code, calculations, prose, or diagrams.

## Part A: learning-rate dynamics by hand

For `loss(theta) = 0.5 * 5 * theta^2`, gradient `5 * theta`, and initial `theta = 2`:

1. Predict and calculate the first four exact-gradient updates for learning rates
   `0.1`, `0.3`, and `0.5`.
2. Record the parameter sign and loss after every update.
3. Classify each trace as converging, oscillating while converging, or diverging.
4. Explain the classification from the multiplier `1 - learning_rate * 5`.

## Part B: momentum trace

Using the same scalar objective, `learning_rate = 0.1`, `momentum = 0.8`, and
initial velocity zero, calculate four classical-momentum updates. Show gradient,
velocity, parameter, and loss at each step. Explain one benefit and one risk visible
in the trace.

## Part C: transfer and policy defense

Given an unfamiliar training trace with noisy batch losses, state what additional
instrumentation distinguishes ordinary SGD noise from divergence. Propose a monitored
learning-rate response, a rollback trigger, and evidence that would make you revisit
the optimizer choice.

Pass requires independent arithmetic, full intermediate state, prediction before
calculation, a correct dynamics explanation, and an oral defense. Leave all responses
unfilled in the repository.

