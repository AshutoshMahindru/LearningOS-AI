# Architecture Decision Record prompt

Complete this record after the depth, leaf-size, and controlled-failure experiments. Do not choose a configuration from training accuracy alone.

## Decision

Which tree constraint configuration, if any, should be retained as the interpretable V03 baseline: shallow `max_depth`, larger `min_samples_leaf`, both, or no tree deployment?

## Context

State the intended use and non-use. Identify the unchanged train/test split, the synthetic-data limitation, the need for individual path explanations, and the prohibition on real learner decisions from this fixture.

## Alternatives considered

Compare at least the depth-3 baseline, the unconstrained tree, and one `min_samples_leaf` alternative. Include “do not deploy” as a valid alternative.

## Evidence

Record depth, node count, leaf count, smallest leaf, train accuracy, test accuracy, generalization gap, at least two path traces, and the observed sensitivity of feature importance or structure.

## Trade-offs

Discuss fit, held-out behavior, path length, small leaves, stability, maintenance burden, and the risk that apparent interpretability encourages causal or policy overreach.

## Revisit conditions

Name measurable triggers such as new representative data, a material test-score change, unstable paths, subgroup disparities, target-definition changes, or a proposed use affecting real learners.

## Status

Use `proposed`, `accepted for offline V03 comparison only`, or `rejected`. A status that authorizes consequential real-world learner decisions is out of scope for this mission.
