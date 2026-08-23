# V07 integration — freeze inference configuration and adaptation order

## M31 → M32 boundary

M31 already maps the training system that produces a checkpoint. M32 is
the V07 phase-end mission that **consumes that checkpoint with frozen
weights** and controls token selection plus the choice of how to adapt.

The observable V07 surface after M32 is:

```
StageAwareCheckpoint (inference_ready, training_time=False)
  -> prompt / context ids
  -> local score table -> logits
  -> InferenceConfig (temperature, top-k/p, seed, stop, max_tokens)
  -> greedy or seeded sample
  -> tokens + config fingerprint
```

with a named adaptation hierarchy:

`prompt/context → retrieval → tools → parameters`

Retrieval, tools, and parameter-efficient fine-tuning (LoRA/QLoRA) are
**named**. They are not implemented here.

## What M32 must not change

M32 does not retrain the M31 table, does not build a semantic-search
service, does not assemble RAG context packs or citations, and does not
execute tools. Those are M33, M34, and M37 measurements on top of this
provider contract.

M32 does not claim a proprietary production decoder.

## M32 → M33 / M34 / M37 handoff

Later missions may consume `InferenceConfig` and the adaptation
hierarchy. They must not relabel a sampling change as a training-stage
change, must not treat temperature as quality, and must not skip
retrieval for a freshness problem by jumping to weights.

M33/M34/M37 receive:

- `InferenceConfig` fields and `config_fingerprint`
- attached M31 checkpoint identity (`v07-teaching-lm-1`)
- inference version `v07-teaching-inference-1`
- teaching filter order `temperature → top-k → top-p → softmax`
- the adaptation hierarchy sentence
- the training-time versus inference-time boundary from M31
- fail-closed optional live adapter (not required)

They do **not** receive a search index, a RAG pack, or a tool executor
from M32.

Reusable artifacts: `missions/M32/inference_adaptation.py` traces and
the local logits fixtures in that module.
