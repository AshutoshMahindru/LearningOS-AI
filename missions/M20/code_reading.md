# Code reading — from gradient to learning dynamics

Read `run_optimizer` in `optimization_core.py` while following one trace record:

1. **objective and parameters** — the same curvature tuple and initialization define
   every controlled comparison.
2. **gradient** — exact GD, momentum, and Adam receive `quadratic_gradient`; SGD
   receives that gradient plus one seeded zero-mean component perturbation.
3. **optimizer state** — GD/SGD use the gradient directly, momentum updates velocity,
   and Adam updates bias-corrected first and second moments.
4. **applied update** — learning rate scales the optimizer-adjusted direction.
5. **parameters after** — the update is subtracted from the previous parameters.
6. **next loss** — the unchanged objective evaluates the resulting point.

For one step from each optimizer, write the concrete gradient, state before and after,
applied update, parameters, and next loss. Then identify which values can change when
only the learning rate changes.

Do not summarize the trace as “Adam learns faster” or “momentum is better.” State the
objective, horizon, hyperparameters, evidence metric, and any overshoot or noise that
the summary would hide.

