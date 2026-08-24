# Controlled failure — hidden critical, or a tuned eval set

## Failure: a high average ships a rare severe miss, or the pack was tuned

Use the local eval harness, seed `4001`, frozen `m40.eval.v1`, and
current M34/M39 fixtures. Predict, before running, whether an
aggregate-only gate can pass while one unsupported citation is
critical, and whether a pack that dropped holdout/idempotency cases
can look like a clean holdout.

Then run one named governance defect. Repair each named defect in its
own predict/act step. Do not treat contamination as a second
simultaneous repair of hidden-critical.

The defective path uses one named change:

- `hidden_critical`: the suite injects M34 `unsupported_citation` on
  `rag-grounded-reset` and scores an aggregate-only gate, or
- `contaminated_pack`: the suite loads `m40.eval.tuned-dev`, which
  dropped hard cases and relabeled the rest as holdout.

The teaching systems and the canonical case list stay fixed. Only the
gate policy or the pack version is wrong.

The defect can still ship on a high average or on a tuned subset.
That is the point. Diagnosis comes from:

1. `task_success_rate` versus `critical_fail_rate` and slice rates,
2. `fail_reasons` on `decide_release_gate`,
3. localized citation/schema/idempotency strings,
4. `eval_version` and the `contaminated` / `tuned_against` flags,
5. `repair_run` using the broken object's report or pack path.

## Discriminators

Hidden critical: `n_critical_fail == 1`; citation slice > 0;
aggregate-only `passed` is true; canonical slice gates would fail;
the broken report's case ids still match `m40.eval.v1`.

Contamination: `eval_version` is `m40.eval.tuned-dev`;
`contaminated` is true; holdout email and idempotency replay are
absent; `n` is smaller than 12.

Do not start with "call a bigger judge" or "import an eval SaaS."
Read the suite report and pack version first.

## Repair rule

The smallest repair calls `repair_run` on the **broken trace** so
either canonical slice gates are applied to that same report, or the
clean `m40.eval.v1` pack is reloaded. Do not start two unrelated
`defect="none"` runs, do not retune M34, and do not open M41
architecture.

Submit prediction, named defect, preserved fields, rates or version,
root cause, smallest repair, verification, and the regression that
the broken path still fails.

A repair is rejected if it opens M41 mechanisms, if it is two
unrelated happy-path runs, or if it changes several healthy variables
at once.
