# Controlled failure — an unbounded loop, or a replayed side effect

## Failure: the agent never hits a terminal, or it posts twice

Use the local reference machine, seed `3801`, M37's teaching registry,
and SKU-7 at price `42.0`. Predict, before running, whether an
unresolved lookup fixture should terminate at `loop_exhausted` when
`max_steps=3`, and whether resuming a ledger execute without consulting
idempotency should increment `effect_count`.

Then run one named defect. Repair each named defect in its own
predict/act step. Do not treat replayed-side-effect as a second
simultaneous repair of the missing loop bound.

The defective path uses one named change:

- `infinite_loop`: the decide node skips the `max_steps` check, so an
  unresolved lookup never becomes `loop_exhausted`, or
- `replayed_side_effect`: the execute node is rewound after the first
  post and run again without consulting M37 idempotency.

The teaching graph, catalog price, and M37 registry stay fixed. Only
the loop bound or the resume/idempotency gate is removed.

The defect can still spin past `max_steps` or append a second ledger
row. That is the point. Diagnosis comes from:

1. `state.node` and `state.terminal`,
2. `model_turn` versus `max_steps`,
3. `ledger.effect_count` and `session.executions`,
4. checkpoint fields (`last_tool_result`, `completed_effect_keys`),
5. the catalog fact `SKU-7 -> 42.0`.

## Discriminators

Infinite loop: `loop_bound_enforced` is false; `model_turn` exceeds
`max_steps`; `terminal` is not `loop_exhausted`; a safety ceiling
stops the test, not the policy bound.

Replayed post: `idempotency_consulted` is false; `effect_count` is 2;
the two entry ids differ; the graph was rewound to `execute`.

Do not start with "call a bigger model" or "import LangGraph." Read
the state and checkpoint first.

## Repair rule

The smallest repair calls `repair_run` on the **broken trace** so
either the loop bound is restored on that unresolved policy and
`max_steps`, or the same initial checkpoint is replayed with
idempotent execute. Do not start two unrelated `defect="none"` runs,
do not open a memory store, and do not open M39/M40 implementations.

Submit prediction, named defect, preserved fields, `effect_count` or
`model_turn`, root cause, smallest repair, verification, and the
regression that the broken path still fails.

A repair is rejected if it opens M39-M40 mechanisms, if it is two
unrelated happy-path runs, or if it changes several healthy variables
at once.
