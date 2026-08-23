# Code reading — prompt, scores, sampling, stop, seed, metadata, fallback

Read `run_inference`, `prepare_distribution`, `apply_top_k`, `apply_top_p`,
`config_as_evidence`, `repair_run`, and `optional_live_complete` in
`missions/M32/inference_adaptation.py`. M32's code-reading target is the
**inference wrapper**:

1. A produced M31 checkpoint is attached (`training_time=False`)
2. Prompt/context ids select a row of the local score table
3. `InferenceConfig` records temperature, top-k/p, seed, stop, max-tokens
4. Teaching filter order is temperature → top-k → top-p → softmax
5. Greedy uses argmax; sampling uses `random.Random(seed)`
6. Generation stops on a stop token or at `max_tokens`
7. Output metadata is the evidence that makes a run reproducible
8. Live adapters and training-time checkpoints fail closed

Before running the code-reading cell, predict:

- the teaching filter order inside `prepare_distribution`
- whether greedy `run_inference` applies `config.temperature` to the
  selected token
- what `repair_run` reuses from the broken object (reference config /
  signals vs module defaults)

Do **not** look for a vector index, a context pack, citations, or a
tool executor. Those are later missions. If a failure can be diagnosed
from `InferenceConfig` fields or from the adaptation signals, stay at
that level.
