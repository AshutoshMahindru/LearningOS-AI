# M14 learning-session fixture

`learning_sessions.csv` contains 54 deterministic, synthetic session records.
It represents no real person and is licensed as part of this repository solely
for instruction and testing.

## Schema

| Column | Type/unit | Role |
|---|---|---|
| `session_id` | string | traceability identifier; never a model feature |
| `active_minutes` | minutes | selected numeric feature |
| `practice_ratio` | fraction `[0, 1]` | selected numeric feature |
| `review_ratio` | fraction `[0, 1]` | selected numeric feature |
| `help_requests` | count | selected numeric feature |
| `context_switches` | count | selected numeric feature |
| `completion_fraction` | fraction `[0, 1]` | selected numeric feature |
| `activity_events` | instrumentation count | selected feature and deliberate scale trap |

There is intentionally no target, class, segment, cohort, or ground-truth
answer column. The rows contain broad profile structure for exercising internal
diagnostics, while `activity_events` cycles through widely separated magnitudes
independently of that structure. This makes unscaled Euclidean clustering look
clean for the wrong reason.

The notebook injects one corrupted, multi-feature observation in memory for the
outlier stress test; it does not mutate this fixture.
