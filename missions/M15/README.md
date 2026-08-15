# M15 — Represent Meaning and Data as Vectors

## Mission

Treat a vector as an operational representation before treating it as an algebraic object. The lab moves through numeric features, positions, directions and a small set of precomputed semantic embeddings. Every representation names its dimensions so the numbers retain meaning.

The learning loop is:

1. state what each dimension means;
2. predict the result of an operation or ranking;
3. compute it;
4. compare the result with the prediction;
5. diagnose any disagreement in terms of magnitude, direction, scale and metric;
6. explain which measurement belongs in the V04 instrumentation layer.

## Operations

The notebook uses vector addition and subtraction for motion, the L2 norm for magnitude, the dot product for alignment weighted by magnitude, normalization for unit direction, cosine similarity for angular agreement and Euclidean distance for geometric closeness.

Similarity is not an intrinsic property of two records. It is a measurement produced by a representation, preprocessing choice and metric.

## Controlled failure

One fixture is designed so cosine similarity and Euclidean distance select different nearest candidates. A second check shows why treating an unnormalized dot product as cosine is a normalization mistake. The learner must diagnose the intended invariant rather than declare one metric universally correct.

## Source policy

`numpy-quickstart` and `3b1b-linear-algebra` are the registered just-in-time references. The lab itself is CPU-only, deterministic, network-free and uses no API or secrets.

## V04 connection

M15 establishes the first instruments in the V04 Mathematical Instrumentation Layer: magnitudes, pairwise scores, distances, normalized directions, rankings and disagreement reports. Later V04 missions can add matrix, probability, statistics, gradient and optimization measurements without changing the observation-first discipline.

## Completion evidence

Completion requires prediction logs, correct vector operations, a semantic similarity ranking, a controlled-failure diagnosis, a metric ADR, a code-reading trace and a no-AI transfer task. Implementation artifacts do not pre-populate learner evidence.
