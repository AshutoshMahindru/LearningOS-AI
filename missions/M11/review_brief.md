# M11 Formal Engineering Review Brief

## Architecture

The notebook loads one versioned local CSV, creates one deterministic stratified train/test split, fits a shallow tree first, and then runs controlled comparisons through reusable metric and path helpers. The source notebook contains no execution outputs; runtime evidence is produced by Restart + Run All.

## Meaningful artifact

The primary artifact is an offline CPU-only notebook that links scikit-learn's fitted `tree_` arrays and `decision_path` output to learner-authored explanations of thresholds, branches, leaves, impurity, depth, and generalization.

## Consequential interpretation boundary

The fixture resembles learner-readiness data only to make the mechanics concrete. It is synthetic and must not be used to rank, diagnose, or intervene on real learners. Split selection and impurity-based feature importance are predictive properties of a fitted model, not causal effects or policy recommendations.

## Failure diagnosis

The controlled failure makes an unconstrained tree look attractive on training accuracy. Required evidence includes the unchanged held-out score, generalization gap, tree size, smallest leaf, and a constrained repair. A second failure asks the learner to reject a causal claim inferred from feature importance.

## Review checklist

- deterministic local data and split;
- source notebook has stable unique cell IDs and no stored outputs;
- every consequential experiment asks for a prediction before execution;
- shallow/deep and train/test comparisons share the same split;
- path explanations use actual fitted thresholds and node IDs;
- no network, secret, paid API, or non-CPU dependency;
- no prefilled learner evidence;
- causal and high-stakes use limitations are prominent;
- mission unittest, mission pytest, repository unittest, repository validator, and Restart + Run All are executed.

## Unresolved uncertainty

One small synthetic split cannot establish stability across populations or time. Any real deployment would require representative data, subgroup and calibration analysis, temporal validation, a documented decision owner, appeal/override controls, and causal evidence for intervention claims.
