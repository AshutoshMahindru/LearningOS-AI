# V03 — Model Comparison & Diagnostics integration

M13 adds a transparent instance-based classifier to V03 Model Comparison & Diagnostics.

The comparison surface for V03 is:

- **trees:** learned feature thresholds and piecewise regions;
- **ensembles:** aggregated learned models and stability through voting/averaging;
- **neighbors:** stored training instances, distance, local membership and votes;
- **clustering:** unlabeled similarity structure explored in M14.

M13 evidence should be reusable in V03 to answer:

1. Why did this model predict this case?
2. Which training cases had local influence?
3. Which units and preprocessing decisions define similarity?
4. How does the boundary react to `k`, metric and feature set?
5. When is a KNN explanation clearer or less reliable than a tree explanation?

The flagship integration should preserve the scaler and classifier as one pipeline, record the chosen feature set, and expose neighbor-level diagnostics for representative and boundary queries.
