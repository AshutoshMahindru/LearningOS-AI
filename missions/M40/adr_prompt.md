# M40 ADR Prompt — V11 Evaluation Governance

Use `templates/ADR.md`. The decision is not pre-selected. M40 needs a
policy for the V11 artifact M41 will inherit: who owns the eval
dataset, how versions work, which graders are required, how
severities slice, what the regression baseline is, which thresholds
block a release, when humans review, and how long traces are kept.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Dataset ownership / versioning:** [UNFILLED BY LEARNER]
- **Grader hierarchy (deterministic vs rubric vs judge):** [UNFILLED BY LEARNER]
- **Severities / slices:** [UNFILLED BY LEARNER]
- **Regression baseline:** [UNFILLED BY LEARNER]
- **Release thresholds:** [UNFILLED BY LEARNER]
- **Human-review triggers:** [UNFILLED BY LEARNER]
- **Audit / trace retention:** [UNFILLED BY LEARNER]
- **Contamination policy:** [UNFILLED BY LEARNER]
- **Live-judge / paid-eval-SDK fallback:** [UNFILLED BY LEARNER]
- **Teaching-scale limits / V11 closure:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V11 default for evaluation governance (the suite is versioned
and owned; deterministic graders cover schema, citation, termination,
and idempotency; rubric/judge is calibrated and cannot be the sole
required grader; critical slices block release even when the average
is high; a known regression must fail the gate on an unchanged pack;
contaminated packs are rejected; traces needed for diagnosis are
retained). Do **not** claim the teaching harness is a production eval
platform, do not close V11 in this ADR, and do not implement M41
architecture here.

## Alternatives that must be compared

1. Versioned `m40.eval.v1` owned by this mission; deterministic
   graders required for invariants; rubric on a frozen hand-labeled
   set; canonical slice gates with `max_critical_fail_rate=0`;
   LLM-as-judge fail-closed and optional; holdout remains untuned.
2. Optimize against the visible average; drop failing cases from the
   pack; use an LLM judge as the only scorer; treat any 80% outcome
   success as a ship.
3. Require a paid eval SDK and a live judge as the CI path for every
   run, including an M41 architecture diagram in this mission.

## Evidence required

Use the frozen pack baseline, localized grader failures, rubric
disagreement, slice-versus-aggregate comparison, regression injection
on an unchanged pack, trace ablation, and the hidden-critical /
contamination repairs. Do not use a vendor model card as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a real judge replacing local
checklists, attaching M41 architecture as if it were an eval skill,
trusting averages over critical slices, retuning M34 holdout ids, or
any path that ships after dropping the only failing case.
