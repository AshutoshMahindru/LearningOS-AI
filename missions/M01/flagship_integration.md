# M01 → V00 Flagship Integration

M01 supplies the architecture map used throughout V00. It is the orientation layer for every later experiment: before changing a model or pipeline, the learner should be able to locate the change in the larger system.

## What M01 contributes to V00

- a stable vocabulary for **data → training → model → inference → application**;
- explicit distinction between data flow and control flow;
- an operational training/inference boundary;
- placement of retrieval, tools, memory, evaluation, observability, and infrastructure around the model;
- a habit of asking what state changes at each boundary.

## Hand-off to M02

M02 runs and interrogates a concrete ML system. M01 should make M02 legible: the learner can point to the dataset, training call, learned estimator state, prediction call, evaluation metric, and application-facing output before studying their internals.

## V00 integration check

Before the V00 review, the learner should be able to take the M01 map and annotate the M02 execution path on top of it without adding new conceptual layers merely because the implementation uses new library names.
