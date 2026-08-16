# M01 No-AI Gate — Reconstruct the System From Memory

Complete this gate without an AI assistant, search engine, or copied diagram.

## Part A — Blank-page map

On blank paper or an empty document, draw an AI/ML system from memory. Your map must include and connect:

- data;
- training;
- learned model state;
- inference;
- application;
- embeddings/retrieval;
- a tool boundary;
- memory/state;
- evaluation/observability;
- compute/infrastructure.

Use different arrow annotations for **data flow** and **control flow**.

## Part B — Explain the boundaries aloud

In your own words, explain:

1. what changes during training;
2. what does not normally change during inference;
3. how retrieval can change an answer without retraining the model;
4. how a tool differs from the model that decided to call it;
5. where memory lives and what state it preserves;
6. how evaluation or observability can influence a later improvement loop without automatically being training.

## Part C — Fresh transfer

Map this unfamiliar system without referring to the notebook:

> A user uploads a policy document. A service chunks and indexes it. Later, a chat request retrieves relevant chunks, sends them with the user's question to a language model, optionally invokes a calculator, stores the conversation state, and records latency plus answer-quality scores.

Identify where training, inference, retrieval, tools, memory, evaluation, and infrastructure occur. If the described system contains **no model-training step**, say so explicitly rather than inventing one.

## Pass standard

Pass only when the learner can produce a coherent map and defend ambiguous boundaries from system behavior. Naming every term correctly without explaining the flows is insufficient.
