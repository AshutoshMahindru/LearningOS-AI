# M18 ADR Prompt — Choose an Inference Decision Policy

An ADR is warranted because the experiment decision rule becomes shared V04
infrastructure: it affects product releases, future model comparisons, auditability,
and the incentive to search for favorable metrics. Write a fresh ADR using
`templates/ADR.md`. Do not treat this prompt as a pre-selected answer.

## Decision to make

What policy will turn an observed A/B difference into **ship**, **do not ship**, or
**collect more evidence**, while preserving useful exploratory analysis?

The decision must specify:

- population, randomization/unit of analysis, primary metric, estimand, and minimum practically important effect;
- fixed sample-size or sequential stopping rule;
- confidence level and the role (if any) of a hypothesis-test threshold;
- comparison-family definition and multiplicity control;
- required diagnostics for assignment, dependence, missingness, and instrumentation;
- how estimates, intervals, effect sizes, harms, costs, and uncertainty combine;
- separation of exploratory and confirmatory work; and
- owner, audit record, and revisit trigger.

## Alternatives that must be compared

1. Ship whenever the unadjusted primary p-value is below 0.05.
2. Use a pre-specified primary estimand with effect/interval and operational utility,
   controlling the declared comparison family.
3. Use a Bayesian decision rule with explicit prior and loss model.
4. Require a second confirmatory experiment for every explored finding.

You may choose, reject, or combine alternatives, but must evaluate false-positive
control, false-negative cost, sample efficiency, interpretability, auditability, and
resistance to p-hacking.

## Evidence required

Include the checkout fixture, sampling-variation experiment, analytic/bootstrap
intervals, permutation result, controlled-failure trace, at least one seed
sensitivity check, and the no-AI transfer scenario. State which evidence is
instructional simulation rather than production observation.

## Revisit conditions

At minimum consider clustered/repeated users, sequential monitoring, more than one
primary outcome, non-random assignment, rare harms, changed business loss, and an
observed false-positive or false-negative incident.
