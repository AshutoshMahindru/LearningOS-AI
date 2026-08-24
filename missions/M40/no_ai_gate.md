# No-AI gate — design evals, slices, and a release blocker

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers and traces below.

## Fixture (fresh)

A lobby kiosk and a warehouse desk, not the notebook SKU-7 purchase
and not `rag-reset-login`:

```
title              = Atlas of Rivers
call number        = QH 441
floor              = 4
weather query      = Will it frost tonight in the valley?
bin                = BIN-12
occupancy          = 3
refund order       = 8802
poetry             = Write a sonnet about forklifts.
```

The kiosk must retrieve a floor for a titled book and must abstain on
weather. The warehouse agent may reserve `BIN-12` and must refuse
poetry. Deterministic graders exist for citation support, tool schema,
termination, and idempotency. A teammate reports:

```
n=20  task_success=0.95  critical_fail_rate=0.00
```

after dropping the only poetry-refusal case that failed last week.

Another teammate's trace:

```
case=reserve-bin-12  terminal=complete  effect_count=2
idempotency_key=reserve-bin-12  replayed=False
```

## Part A: five eval cases

Design five eval cases for this fresh requirement. Include at least
one RAG-like lookup/abstention case and one agent tool/termination
case. One short table: id, family, gold outcome.

## Part B: deterministic versus rubric

For two of those cases, choose deterministic versus rubric grading
and justify in one sentence each. Do not make a judge the only
grader for an invariant.

## Part C: slice versus aggregate

Using `n=20`, `task_success=0.95`, and one remaining severe
idempotency miss (`effect_count=2` on a single case), compute a small
slice/aggregate report: outcome success rate and critical-fail rate.
State whether an aggregate-only 0.80 gate would ship.

## Part D: contamination

Identify the eval-contamination example in the teammate report that
dropped the failing poetry case. One short paragraph.

## Part E: release blocker

Define one release-blocking failure that is independent of average
score. Use the `BIN-12` double-post or an unsupported citation.

Pass requires five cases, two grader choices, a slice/aggregate
report, a contamination identification, and one average-independent
blocker, plus an oral defense.
Leave all learner responses unfilled in the repository.
