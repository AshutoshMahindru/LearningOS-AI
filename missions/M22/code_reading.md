# Code reading — the dense layer, not the training loop

Read `dense_forward` and `validate_dense_shapes` in
`missions/M22/neuron_layer_core.py`. M22's code-reading target is the
**layer contract**:

1. input coercion to a 2-D row batch
2. `W` rows matching `n_in` and columns matching `n_out`
3. bias length matching `n_out`
4. affine map `X @ W + b`
5. activation **after** that map
6. error paths for feature/weight mismatch

Before running the code-reading cell, predict:

- what happens if `W` is supplied as `(n_out, n_in)` and passed to `dense_forward`
- what happens if `b` has the wrong length
- whether a 1-D `x` is treated as a batch of one row

Do **not** look for `backward`, `grad`, `softmax`, or a stacked MLP class.
Those are M23-M25. If a failure can be diagnosed from shapes or a
two-number hand calculation, stay at that level.
