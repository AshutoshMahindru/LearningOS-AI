# M12 — Improve Weak Trees with Ensembles

## Mission objective

Compare bagging, random forests, and boosting by experiment. Start with a deliberately limited decision tree, measure what varies across bootstrap samples, and make an ensemble choice from held-out evidence rather than from the slogan that ensembles are always better.

## Experimental route

1. Fit a depth-limited tree and record train/test behavior.
2. Resample the training set repeatedly and observe where tree predictions disagree.
3. Average resampled trees with bagging and add feature randomness with a random forest.
4. Trace boosting stage by stage to see sequential correction and new mistakes.
5. Sweep ensemble size and base-tree depth while holding the split fixed.
6. Trigger the controlled failure: excess tree complexity and corrupted labels show why “more trees always fixes it” is false.
7. Write an ADR that selects an ensemble under stated latency, interpretability, and error constraints.
8. Defend the evidence in a formal engineering review.

## Bias/variance intuition

A single tree can move sharply when its training sample changes. Bootstrap disagreement is an observable proxy for that sensitivity, not a complete mathematical bias/variance decomposition. Bagging averages predictions from resampled trees to reduce variation. A random forest also randomizes the candidate features at splits, seeking less-correlated errors. Boosting is different: each stage is added sequentially to reduce the current ensemble's loss, so depth, learning rate, and number of stages jointly control capacity.

## Completion standard

Completion requires the notebook experiments, a diagnosis of the controlled failure, the no-AI transfer gate, a review using `review_brief.md`, and a learner-authored decision using `adr_prompt.md`. The repository intentionally contains no filled learner answers, review verdict, or ADR.
