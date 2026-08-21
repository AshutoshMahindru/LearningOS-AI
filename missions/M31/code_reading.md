# Code reading — tokenize, window, shift, batch, loss, split, checkpoint, eval

Read `shift_tokens`, `pair_windows`, `run_causal_pipeline`,
`lineage_with_leak`, `repair_run`, and `make_checkpoint` in
`missions/M31/llm_training_core.py`. M31's code-reading target is the
**miniature causal training pipeline**:

1. M27 word tokenizer encodes each document (`[BOS] … [EOS]`)
2. `apply_context_length` keeps at most `C` tokens
3. `shift_tokens` builds `inputs = window[:-1]`, `targets = window[1:]`
4. train ids are taken from `used_lineage` (authored, unless leaked)
5. a `(V, V)` score table produces logits for each input id
6. mean NLL vs targets is the objective; SGD updates only that table
7. evaluation scores authored eval ids with **correct** alignment
8. `StageAwareCheckpoint` records dataset version, split hash, stage, seed, steps

Before running the code-reading cell, predict:

- the `(input, target)` pairs for a four-token window
- whether evaluation uses the defective alignment or the correct one
- which lineage field moves first if `e02` is leaked into train
- what M32 is allowed to do with `inference_ready=True`

Do **not** look for temperature, top-k, a generate loop, or a RAG
index. Those are later missions. If a failure can be diagnosed from
constructed pairs or split overlap, stay at that level.
