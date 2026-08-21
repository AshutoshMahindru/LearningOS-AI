# M31 — Understand LLM Training

M30 already traces one transformer block. The useful whole here is the
**system that produces a language model**:

```
tokens → window → shifted (input, target) pairs
                → next-token NLL
                → checkpoint
                → protected evaluation
```

Causal next-token prediction is the pretraining objective: at each
position the model scores the following token, not the current one.
A tiny NumPy bigram table is the teaching stand-in for a stacked-block
trainer. Scale limits are explicit. Falling training loss is not
evidence of a general LLM.

This mission uses the bundled M27 word tokenizer and a synthetic
eight-document corpus. Nothing is downloaded. Inference controls
(temperature, top-k/p, decoding) stay closed until M32. RAG stays
closed until M34.

Canonical sources: `hf-llm-course` and `karpathy-zero-to-hero` via
`data/source_registry.json`.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
