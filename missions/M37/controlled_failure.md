# Controlled failure — malformed arguments or a doubled side effect

## Failure: the tool ran when it should not have, or it ran twice

Use the local three-tool registry, seed `3701`, and an attached M32
`InferenceConfig` (`training_time=False`, `weights_updated=False`).
Predict, before running, whether a side-effecting ledger post with a
non-numeric amount should increment `effect_count`, and whether a
timeout retry of the same approved post should increment it again.

Then run one named defect.

The defective path uses one named change:

- `malformed_reaches_side_effect`: validation is skipped and
  `post_ledger_entry` runs with a non-numeric amount, or
- `duplicate_side_effect`: a simulated timeout retry posts the same
  approved ledger entry without consulting the idempotency store.

The teaching registry, VAT schema, and catalog stay fixed. Only the
validation or idempotency gate is removed.

The defect can still append a ledger row. That is the point.
Diagnosis comes from:

1. `validation.ok` versus `execution_reached`,
2. `ledger.effect_count` and `session.executions`,
3. `error_kind` (`schema` vs `tool` vs `permission`),
4. the hand product `80 * 0.25 = 20`.

## Discriminators

Malformed reach: `validation_bypassed` is true; `effect_count` is 1;
a healthy `validate_proposal` on the same object is not ok; the
amount type is `str`.

Duplicate post: `idempotency_consulted` is false; `effect_count` is
2; the proposal itself is schema-valid; the two entry ids differ.

Do not start with "call a bigger model" or "add LangGraph." Read the
gates first.

## Repair rule

The smallest repair calls `repair_run` on the **broken trace** so
either validation is restored on that proposal or the same
idempotency key is consulted against the defective object's initial
ledger snapshot. Do not start two unrelated `defect="none"` runs, do
not open a state machine, and do not open M34/M36/M38/M39
implementations.

Submit prediction, named defect, preserved fields, `effect_count`,
root cause, smallest repair, verification, and the regression that
the broken path still posts.

A repair is rejected if it opens M38-M40 mechanisms, if it is two
unrelated happy-path runs, or if it changes several healthy variables
at once.
