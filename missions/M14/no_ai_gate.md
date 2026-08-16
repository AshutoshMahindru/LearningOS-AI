# M14 No-AI Gate

Complete this gate without AI-generated code or prose. Use a fresh, unlabelled
numeric dataset that is not the M14 fixture.

1. Select three to seven defensible features and document every exclusion.
2. Predict the dominant raw-distance feature, then compute a scale diagnostic.
3. Compare at least four candidate values of `k` after justified preprocessing.
4. Choose one candidate using separation, stability, cluster size, and domain
   usefulness. Record evidence that argues against your choice.
5. Show original-unit center profiles plus a within-cluster diagnostic.
6. Stress the result with either a plausible outlier or a unit conversion.
7. Write one paragraph beginning: “These clusters are not true classes because …”
8. Produce an ADR from `missions/M14/adr_prompt.md`.

The gate passes only if another learner can rerun the work and distinguish
observations, assumptions, decisions, and interpretation. A colorful cluster
plot by itself does not pass.
