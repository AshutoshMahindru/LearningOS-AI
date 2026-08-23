# V07 integration — freeze a training-time objective and lineage

## M30 → M31 boundary

M30 already maps one transformer block. M31 is the first V07 mission
that **wraps that block in a training system**: a causal next-token
objective, explicit input/target shifting, a tiny teaching simulation,
train/eval lineage, and a stage map.

The observable V07 surface after M31 is:

```
Document (id, text, authored split, checksum)
  -> M27 word encode
  -> context window
  -> inputs = window[:-1], targets = window[1:]
  -> (V, V) score table  [teaching stand-in for stacked blocks]
  -> mean NLL
  -> StageAwareCheckpoint (dataset version, split hash, stage, seed)
```

with named pair checkpoints and a protected eval id list.

The M30 block may be unembedded for shape inspection. It is not trained
here. Teaching scale is a bigram table on eight documents.

## What M31 must not change

M31 does not decode, does not set temperature or top-k/p, does not
choose prompt versus retrieval versus tools versus weights, and does
not implement RAG. Those are M32 and M34 measurements on a produced
checkpoint.

M31 does not claim a proprietary production recipe.

## M31 → M32 handoff

M32 may consume a produced checkpoint (`inference_ready=True`,
`training_time=False`) and control **token selection**. It must not
relabel a sampling change as a training-stage change, and it must not
silently mix eval documents into train.

M32 receives:

- `StageAwareCheckpoint` identity and audit metadata
- authored versus used split lineage
- objective name `causal_next_token`
- adaptation stage `pretrained` on the teaching artifact
- the training-time versus inference-time boundary sentence
- version `v07-teaching-lm-1`

Reusable artifacts: `missions/M31/llm_training_core.py` traces and
`datasets/M31/corpus.json`.
