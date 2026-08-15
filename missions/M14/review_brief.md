# M14 Formal Engineering Review Brief

## Review outcome requested

Approve or reject the mission package as a reproducible, CPU-only clustering
investigation. Approval covers the learning artifact and diagnostic contract; it
does **not** approve the discovered clusters as real learner types or authorize
downstream decisions about people.

## System under review

```text
local unlabelled CSV
  → schema and finite-value checks
  → identifier exclusion + seven numeric features
  → StandardScaler fitted on the analysis fixture
  → deterministic KMeans candidates (k=2..6, k-means++, n_init=20)
  → inertia + silhouette + seed stability + size checks
  → selected fit
  → inverse-transformed centers + sample silhouettes + center distances
  → controlled failure stress tests + cautious interpretation
```

The fixture is synthetic, contains 54 observations, and has no target, class,
segment, or hidden answer column. `session_id` is retained only for traceability.

## Consequential choices

| Choice | Rationale | Main risk | Review evidence |
|---|---|---|---|
| Seven interpretable numeric features; exclude `session_id` | Identifiers do not express meaningful distance | Instrumentation count may still encode collection behavior | Range audit and raw-distance contribution |
| StandardScaler | Mixed units otherwise dominate Euclidean geometry | Sensitive to outliers; equal variance is not equal importance | Raw/scaled partition comparison and injected-outlier stress |
| K-means with Euclidean distance | Transparent centers and a small dense numeric fixture | Prefers roughly convex, center-described groups | Center profiles, sample silhouettes, limitations text |
| Candidate `k=2..6` | Wide enough to expose merge/split behavior on 54 rows | Search range may omit other structure | Diagnostics table and explicit reversal trigger |
| `n_init=20`, fixed seeds | Reduce initialization accidents and make runs reproducible | Stability across seeds is not stability across samples/time | Cross-seed adjusted Rand score and ADR prompt |
| PCA only for display | Two dimensions are inspectable | Projection can manufacture or hide separation | Explained variance plus full-space metrics |

## Invariants reviewers should verify

1. Restart + Run All succeeds without secrets, paid APIs, GPUs, or network.
2. Source cells have stable unique IDs, null execution counts, and no outputs.
3. The lab never reads or derives target labels.
4. Candidate fits use the declared standardized matrix; raw fitting is clearly a
   controlled failure.
5. `k` is defended with multiple diagnostics and counterevidence.
6. Centers are inverse-transformed before domain interpretation.
7. Per-sample evidence reveals structure hidden by aggregate silhouette.
8. Cluster numbers receive no ordinal or class semantics.
9. The ADR prompt captures alternatives, tradeoffs, and reversal conditions.

## Failure containment

- The fixture is local and synthetic; no user data or personal action is in scope.
- Assertions stop execution on schema drift, missing values, non-finite values,
  duplicate IDs, or an unexpected selected-k diagnostic.
- Raw-scale, forced-k, outlier, and projection traps are named and quantified.
- The final interpretation boundary requires external validation before clusters
  influence product or educational decisions.

## Open review questions

1. Is equal-variance scaling defensible for all seven features, or should
   `activity_events` be excluded/weighted after a domain review?
2. Is cross-seed stability sufficient for this mission, or should the next
   version add bootstrap/subsample stability?
3. Does the candidate range communicate that `k` is scoped rather than natural?
4. What external outcome would make a cluster interpretation useful and safe?
5. Which drift check should V03 run before reusing centers on later cohorts?

## Required review evidence

- executed notebook produced from a clean restart;
- mission unittest and pytest results;
- full repository unittest and validator results;
- clean `git diff --check`;
- completed learner-authored ADR using `adr_prompt.md` (required for learner
  completion, deliberately not prefilled in the source package).
