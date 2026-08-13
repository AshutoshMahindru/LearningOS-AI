# M01 — Map the AI/ML Landscape

## Mission objective

Build a whole-system map of modern AI/ML before descending into implementation details. The mission is complete when the learner can take an unfamiliar AI application, identify its major layers, and explain what information and control cross each boundary.

## Whole-first route

Start with the useful whole:

**data → training → model state → inference → application**

Then add the surrounding system:

- structured and unstructured data;
- classical machine learning, neural networks, and large language models;
- embeddings, retrieval, and retrieval-augmented generation (RAG);
- tools, agents/controllers, and memory/state;
- evaluation, observability, and feedback;
- compute and infrastructure.

The notebook implements a deterministic toy support system using only Python's standard library. It deliberately uses small, inspectable components rather than opaque services. The learner can therefore point to each input, output, piece of state, data-flow edge, and control-flow decision.

## Learning sequence

1. Draw a first-pass map from the six-layer skeleton above.
2. Run the toy system end to end before studying any component deeply.
3. Inspect how training converts examples into model state.
4. Verify that inference consumes model state without retraining it.
5. Build a tiny lexical embedding and retrieval index; observe how retrieval augments an answer path without changing model weights.
6. Trace a deterministic controller that can choose retrieval, call a tool, and update application memory.
7. Inspect evaluation and trace records as cross-cutting system layers.
8. Diagnose a deliberately mislabelled architecture in which training, inference, retrieval, tools, memory, and evaluation are conflated.
9. Pass the no-AI gate by redrawing and explaining the system from memory.
10. Complete transfer assessment on architectures not used in the notebook.

## Core distinction

A useful default boundary is:

- **Training** changes learned parameters or other learned model state using training data and an objective.
- **Inference** uses already-created model state to produce outputs for new inputs.
- **Retrieval** selects external information at run time; retrieval can affect an inference request without itself becoming model training.
- **Tools** perform actions or computations outside the model.
- **Memory** preserves application/session state across steps or runs.
- **Evaluation and observability** measure behavior and expose what the system did; feedback from them may later trigger a separate training or product-improvement loop.

Real systems can blur operational boundaries, but the map should always state explicitly which component changes learned model state and which components only participate at run time.

## Deliverables

Learner deliverables are defined in `evidence_contract.yaml`. This repository does **not** contain prefilled learner evidence.

## Source policy

M01 does not add a global content-registry entry. `content.yaml` records a small mission-local set of current primary/official sources and the exact sections to consult.
