# ADR prompt — define similarity for V03

Record a decision for the KNN component of V03 Model Comparison & Diagnostics.

## Decision question

How will V03 define and preserve meaningful similarity for neighbor-based predictions?

## Required alternatives

Compare at least:

1. raw features with Euclidean distance;
2. standardized features with Euclidean distance;
3. standardized features with Manhattan distance;
4. standardized informative features after removing the weak instrumentation feature;
5. a non-KNN model when trustworthy distance cannot be defined.

## Required evidence

- fixed-split performance across candidate `k` values;
- representative and boundary-query neighbor identities;
- per-feature distance contributions;
- behavior with the high-scale weak feature;
- scaling fitted on training data only;
- sensitivity to metric and `k`;
- interpretability and inference-cost implications.

## ADR sections

Use `templates/ADR.md` and include context, decision, alternatives, evidence, trade-offs, consequences, safeguards and revisit conditions.

Revisit the decision if feature units change, drift alters neighborhood composition, latency or data size makes exact search unsuitable, a new feature dominates distance, class balance changes materially, or representative query behavior conflicts with aggregate metrics.
