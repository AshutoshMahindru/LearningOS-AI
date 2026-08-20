# Code reading — project, score, mask, softmax, aggregate

Read `project_qkv`, `dot_product_scores`, `scale_scores`,
`apply_additive_mask`, `softmax_over_keys`, `aggregate_values`, and
`scaled_dot_product_attention` in `missions/M29/attention_core.py`.
M29's code-reading target is the **single-head dataflow**:

1. `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V` (row-batch, M16)
2. raw scores `Q @ K^T` with shape `(batch, queries, keys)`
3. multiply by `1/sqrt(d_k)` unless an experiment sets scale to 1
4. add the mask **before** softmax (`0` keep, `MASK_FILL` block)
5. softmax on the **key** axis (`axis=-1`)
6. `weights @ V` with output shape `(batch, queries, d_v)`
7. row sums of weights equal 1 when every query has an allowed key

Before running the code-reading cell, predict:

- the shapes of Q, K, V, scores, weights, and output for the cash
  sequence `(3, 2)` with identity projections
- whether softmax runs over keys or over queries
- whether a causal mask at position 0 can put mass on `cash`

Do **not** look for head splitting, residuals, LayerNorm, or a
feed-forward sublayer. Those are M30. If a failure can be diagnosed
from row sums or masked mass, stay at that level.
