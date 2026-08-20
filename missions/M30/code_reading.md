# Code reading — split, M29 call, merge, residual, norm, FFN

Read `split_heads`, `merge_heads`, `multi_head_attention`,
`layer_norm`, `residual_add`, `feed_forward`, and `transformer_block`
in `missions/M30/transformer_block.py`. M30's code-reading target is
the **pre-norm block dataflow**:

1. `attn_norm = LN(x)` over the last axis (declared pre-norm)
2. `Q, K, V = X @ W_*`, then reshape to `(batch, seq, n_heads, d_head)`
3. M29 `scaled_dot_product_attention` on a head-batch
4. merge heads and apply `W_O`
5. `attn_residual = x + attn_projected` (original stream, not LN(x))
6. `ffn_norm = LN(attn_residual)`
7. position-wise `ReLU(h @ W1 + b1) @ W2 + b2`
8. `output = attn_residual + ffn_projected`

Before running the code-reading cell, predict:

- shapes of `q_heads`, `head_outputs`, `attn_concat`, and `output`
  for cash `(3, 4)` with `n_heads=2`
- whether the residual add uses `x` or `attn_norm` in pre-norm
- whether the FFN mixes positions or only features
- which M29 function is called (not rewritten) inside the block

Do **not** look for a training loop, a next-token shift, temperature,
or an adaptation policy. Those are M31-M32. If a failure can be
diagnosed from the first diverging checkpoint, stay at that level.
