# M14 ADR Prompt — Clustering Geometry and Interpretation

Create an ADR using `templates/ADR.md`. Do not copy a preferred answer from the
lab. The ADR records your decision, not a universal best clustering pipeline.

## Decision title

Choose a title that names the data, geometry, algorithm, and scope, for example:
“Standardized Euclidean K-means for exploratory learning-session profiles.”

## Context to establish

- What question can an unlabelled partition help investigate?
- Which decisions are explicitly out of scope?
- What is one observation, what is the analysis population, and which time window
  does the fixture represent?
- Which columns were available, which became features, and which were excluded?
- What evidence suggests or contradicts center-based, roughly convex structure?

## Consequential decisions to record

1. Feature inclusion/exclusion and any weighting.
2. Scaling or transformation, including where preprocessing is fitted.
3. Distance geometry and K-means rather than at least one alternative such as
   robust scaling, hierarchical clustering, DBSCAN, or no clustering.
4. Candidate range for `k`, selected `k`, initialization count, and seed policy.
5. Internal diagnostics and their decision thresholds or interpretation.
6. Outlier policy: investigate, cap, transform, use robust preprocessing, or keep.
7. Visualization method and what the projection cannot establish.
8. Names, IDs, and downstream uses allowed for clusters.

## Evidence table

For every decision, provide:

| Decision | Evidence for | Evidence against | Alternative | Consequence |
|---|---|---|---|---|

Include the raw/scaled comparison, candidate-k table, within-cluster summary,
seed-stability result, and outlier stress result. A plot alone is insufficient.

## Consequences and reversal triggers

State expected benefits, limitations, operational costs, and at least four
conditions that force reconsideration. Include triggers for assignment
instability, changing feature distributions, small/empty clusters, degraded
within-cluster evidence, and failure of external validation.

End with this boundary in your own words: cluster assignments are model outputs
under chosen features, scaling, distance, algorithm, sample, and `k`; clusters
are not true classes discovered in the data.
