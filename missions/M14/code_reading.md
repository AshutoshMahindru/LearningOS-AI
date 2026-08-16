# M14 Code Reading — Trace a Candidate-k Diagnostic

Read the notebook's `evaluate_candidate` function before running it. Trace one
call for `k=3` and answer:

1. Which matrix defines the geometry: raw values or standardized values?
2. Which state is learned by `fit_predict`, and where are assignments stored?
3. Why does the function use `n_init=20` and a fixed `random_state`?
4. Why is inertia not comparable as “larger is better”?
5. What does the mean silhouette hide that `silhouette_samples` later exposes?
6. The stability score compares two partitions without true labels. Why are
   cluster-number permutations harmless to adjusted Rand index?
7. Which line would fail if `k=1`, and why is that candidate excluded?

Then sketch the data flow:

`CSV → selected numeric features → scaler state → standardized matrix → K-means centers/assignments → internal diagnostics → interpretation`

Mark every consequential choice on that flow. Those choices belong in the ADR.
