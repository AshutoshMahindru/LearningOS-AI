# M01 Code Reading — Trace the System, Not the Syntax

Read the notebook's toy system as an architecture diagram expressed in code.

## Pass 1 — Locate state-changing code

Find `train_classifier(records)`. For every value it creates, answer:

1. What enters the function?
2. What learned state leaves it?
3. Which loop uses labels from training data?
4. If this function never runs again, can `predict(...)` still produce outputs?

Mark this function as **training** and justify the label from behavior, not from its name.

## Pass 2 — Trace inference

Follow one call to `predict(model, text)`.

- Identify the input.
- Identify where `tok(...)` converts raw text into the feature/representation consumed by the model.
- Identify the model state read.
- Identify the prediction output (the selected label and all label scores).
- Identify whether any learned state changes.
- Explain why calculating scores is inference even though computation occurs.

## Pass 3 — Trace retrieval

Follow `embed(...)` → `cosine_similarity(...)` → `retrieve(...)`.

Record the shape and meaning of the query vector, document vectors, similarity scores, and returned document. Explain why changing retrieved context is not the same thing as changing model weights.

## Pass 4 — Trace control flow

Follow `run_application(...)` for two inputs:

- a normal account request;
- an urgent billing request.

Draw arrows for both **data flow** and **control flow**. The controller reads a model output, may call retrieval, may call a tool, then mutates application memory. State which of those operations are model inference and which are application orchestration.

## Pass 5 — Separate evaluation from observability

Compare `evaluate(...)` with the trace events emitted by `run_application(...)`. Explain why labelled correctness is evaluation while event records are observability. For each trace event, identify what evidence it preserves about system behavior and one failure that would be hard to diagnose without it.

## Transfer prompt

You encounter an unfamiliar recommendation service with an offline feature pipeline, a trained ranking model, an online feature store, an inference service, a rules engine, an A/B-test platform, and telemetry. Without running code, map each component to the M01 system layers and identify one ambiguous boundary you would investigate further.
