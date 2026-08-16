# Code reading trace

Read `one_parameter_step` in `gradient_core.py` in this order:

1. **parameter** — `weight` is the current adjustable value.
2. **prediction** — `predict_one_parameter` multiplies each input by that weight.
3. **loss** — `one_parameter_loss` aggregates squared prediction errors.
4. **gradient** — `analytic_weight_gradient` measures local loss change with respect to the weight.
5. **update** — `update_parameter` subtracts `learning_rate * gradient`.

For each name, point to the exact value passed to the next stage. Then explain why changing `-` to `+` changes the behavior even though the prediction, loss, and gradient calculations remain correct.

Do not summarize the loop as “the model learns.” State which parameter changed, why it moved in that direction, and what evidence the next loss supplies.
