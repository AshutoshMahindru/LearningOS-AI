# M32 — Control Inference and Adaptation

M31 already produced a language-model identity: a stage-aware checkpoint
with `training_time=False` and `inference_ready=True`. The useful whole
here is **token selection and the adaptation choice**, not another
training run:

```
checkpoint → prompt/context → logits → temperature → top-k/top-p
          → greedy or seeded sample → stop/max-tokens
          → tokens + InferenceConfig evidence
```

Temperature and candidate filters change the **distribution over
already-scored tokens**. They are not quality knobs in isolation and
they are not a new model. Prompt/context changes what the frozen
checkpoint sees. Retrieval, tools, and parameter updates are later
levers; this mission records a **decision rubric**, not a search
service, RAG pack, or tool executor.

Canonical path: local deterministic logits/sampling fixtures. Nothing
is downloaded. No paid API.

Canonical sources: `hf-llm-course` and `karpathy-zero-to-hero` via
`data/source_registry.json`.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
