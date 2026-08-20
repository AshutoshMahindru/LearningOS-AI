# Code reading — the forward graph, not the gradient tape

Read `two_layer_forward`, `stable_softmax`, and `validate_stack_shapes` in
`missions/M23/forward_core.py`. M23's code-reading target is the
**inference contract**:

1. input coercion to a 2-D row batch
2. `W1` rows matching `n_in` and columns matching `n_hidden`
3. `W2` rows matching `n_hidden` and columns matching `n_classes`
4. bias lengths matching the corresponding column counts
5. affine maps `X @ W + b` with trailing-axis broadcast
6. hidden activation **after** the first affine map
7. logits from the second affine map with **no** output activation
8. stable softmax along the **class** axis (`axis=-1`)
9. declared `float64` / `atol=1e-12` parity against the M22 stack

Before running the code-reading cell, predict:

- what happens if softmax is asked to reduce `axis=0` on a `(batch, n_classes)` array
- what happens to example 0 if the hidden ReLU is omitted
- whether a 1-D `x` is treated as a batch of one row
- why subtracting `max(logits)` along the class axis does not change the probabilities

Do **not** look for `backward`, `grad`, `loss`, or a training step.
Those are M24-M25. If a failure can be diagnosed from one-example versus
batch parity or a named intermediate, stay at that level.
