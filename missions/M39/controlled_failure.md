# Controlled failure — stale memory trusted, or fallback oscillation

## Failure: complete with expired facts, or a ladder that never stops

Use the local robustness layer, seed `3901`, M38's teaching workflow,
and SKU-7 at catalog price `42.0`. Predict, before running, whether an
expired memory row at `99.0` may be posted as `complete`, and whether
primary plus fallback can exceed `max_attempts` when the circuit is
ignored.

Then run one named defect. Repair each named defect in its own
predict/act step. Do not treat fallback oscillation as a second
simultaneous repair of stale-memory trust.

The defective path uses one named change:

- `stale_memory_trusted`: retrieval skips expiry and the purchase posts
  the remembered amount as `complete`, or
- `fallback_oscillation`: the ladder ignores the circuit and the
  attempt bound, so injected failures keep alternating rungs.

The teaching catalog, route predicates, and M38 wrap stay fixed. Only
expiry enforcement or the termination bound is removed.

The defect can still post `99.0` as success or spin past
`MAX_ATTEMPTS`. That is the point. Diagnosis comes from:

1. `retrieved_ids` versus `excluded` reasons (`expired`, `superseded`),
2. `used_memory_ids` and `posted_amount` versus catalog `42.0`,
3. `route`, `attempts`, `circuit_open`, and rung names in `trace`,
4. `degraded` versus `terminal="complete"`,
5. provenance `written_at` / `expires_at` versus `now`.

## Discriminators

Stale trust: `used_memory_ids` contains `mem-sku7-price-stale`;
`posted_amount` is `99.0`; `degraded` is false; `terminal` is
`complete`; M38 executions skip lookup.

Oscillation: `circuit_open` is false; `attempts` exceeds
`MAX_ATTEMPTS`; `aborted_ceiling` is true; rungs alternate
`primary` / `lookup_only`.

Do not start with "call a bigger model" or "import LangGraph." Read
the retrieval and attempt traces first.

## Repair rule

The smallest repair calls `repair_run` on the **broken trace** so
either expiry is enforced on that store and `now`, or the same
initial store is replayed with the circuit and attempt bound. Do not
start two unrelated `defect="none"` runs, do not open an eval harness,
and do not open M40 implementations.

Submit prediction, named defect, preserved fields, `posted_amount` or
`attempts`, root cause, smallest repair, verification, and the
regression that the broken path still fails.

A repair is rejected if it opens M40 mechanisms, if it is two
unrelated happy-path runs, or if it changes several healthy variables
at once.
