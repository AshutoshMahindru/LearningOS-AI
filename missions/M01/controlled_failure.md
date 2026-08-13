# M01 Controlled Failure — The Architecture That Calls Everything “The Model”

## Faulty architecture

A deliberately confused team diagram claims:

```text
user request
  ↓
TRAINING: query embedding
  ↓
MODEL WEIGHTS: vector database
  ↓
INFERENCE: document retrieval
  ↓
MODEL: controller chooses calculator
  ↓
TRAINING: calculator returns result
  ↓
MEMORY: evaluation dashboard
  ↓
OUTPUT
```

It also draws an arrow from every production request directly into “weight updates,” without showing an objective, optimizer, training job, approval boundary, or newly versioned model artifact.

## Learner task

Before seeing any repair, diagnose the diagram using observable behavior:

1. Which labels conflate training and inference?
2. Which component stores learned model state, and which stores retrievable external data?
3. Which step is control flow rather than model inference?
4. Which component is a tool?
5. Which component is observability rather than memory?
6. What evidence would be required before claiming production requests update model weights?
7. Redraw the architecture with explicit data-flow and control-flow arrows.

## Expected diagnosis dimensions

A successful diagnosis should distinguish at least these ideas:

- embedding computation at request time is an inference/representation step unless a training procedure is actually updating parameters;
- a vector database stores indexed external representations, not the language model's learned weights;
- retrieval selects context and does not by itself retrain the generator;
- a controller/agent can decide to call a calculator, while the calculator remains an external tool;
- an evaluation dashboard is an observability/evaluation surface, not conversational memory;
- online data collection can feed a **future** training loop, but data collection is not automatically weight updating.

The goal is not to memorize these corrections. The learner should infer them from inputs, outputs, state mutation, and control flow.
