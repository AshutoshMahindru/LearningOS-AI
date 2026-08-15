# M18 Formal Engineering Review Brief

## Review decision requested

Approve the M18 statistical-inference instrument as a deterministic teaching and
review surface for deciding whether an observed difference is signal, noise, or
unresolved. Approval covers the mission contract and implementation quality; it
does not certify learner completion or authorize a real checkout rollout.

## System under review

- **Input boundary:** two committed synthetic CSV fixtures. Grouped checkout counts
  are expanded to binary session outcomes; seasonal data demonstrate confounding.
- **Computation boundary:** transparent standard-library functions implement
  summaries, standard errors, confidence intervals, bootstrap resampling,
  permutation and z tests, effect sizes, correlations, and multiplicity simulation.
- **Interaction boundary:** the notebook requires predictions before computation and
  keeps learner answers outside source control.
- **Decision boundary:** the primary estimand is B conversion rate minus A conversion
  rate. Estimates, intervals, practical effect, assumptions, and the planned
  comparison family travel together; no p-value alone determines action.
- **Failure boundary:** the p-hacking cell is seeded, isolated to null simulations,
  and followed by a full-family audit and correction exercise.

## Evidence available to reviewers

1. Dataset contract: A = 72/600 and B = 84/600, an observed +0.02 effect.
2. Analytic and deterministic bootstrap intervals include zero for the fixture.
3. Approximate z and permutation p-values are non-significant at 0.05.
4. Repeated Bernoulli sampling shows smaller empirical standard error at n=400 than
   n=25, close to the theoretical relationship.
5. The seeded 20-comparison null family produces nominal wins that do not meet its
   Bonferroni threshold; repeated families demonstrate inflated familywise error.
6. Mission tests validate numeric invariants, deterministic behavior, source
   notebook hygiene, causal/multiplicity language, and package completeness.

## Material assumptions

- Checkout sessions are treated as independent units; repeated users or clustered
  traffic would make the naive standard errors too optimistic.
- Random assignment, stable measurement, no interference, and consistent exposure
  are assumed for causal interpretation of an A/B effect.
- Bootstrap samples represent the observed data-generating process; resampling does
  not remove bias, confounding, dependence, or metric corruption.
- Normal approximation is acceptable for these group sizes and event counts.
- The permutation test relies on exchangeability under the null.

## Risks and controls

| Risk | Consequence | Control | Residual risk |
| --- | --- | --- | --- |
| P-value threshold becomes an automatic ship rule | practically harmful rollout | effect, interval, stakes, and assumptions are required together | stakeholder incentives remain |
| Hidden comparison family | inflated false positives | full result audit, pre-specification, correction, fresh confirmation | undisclosed external analyses |
| Session dependence | intervals too narrow | explicit unit/independence review | fixture cannot model every cluster |
| Correlation becomes a causal story | invalid intervention | confounder fixture and identification gate | real systems may have subtler confounding |
| Seed mistaken for robustness | overconfidence in one Monte Carlo run | analytic comparison and seed-sensitivity prompt | finite simulation error remains |

## Required reviewer challenges

- Recalculate the fixture totals and both interval endpoints independently.
- Identify the estimand, unit of analysis, null, comparison family, and stopping rule.
- Change at least one resampling seed and verify that the decision is stable even
  though exact Monte Carlo values move.
- Explain what evidence would support equivalence rather than failure to reject.
- Inspect the entire controlled-failure result family before viewing the selected rows.
- Confirm that source cells have stable IDs, empty outputs, and no network/secrets code.

## Acceptance criteria

- Restart + Run All succeeds within the CPU timeout with no runtime network access.
- Mission unittest and pytest invocation both pass; repository tests and validator
  remain green.
- All numeric claims are reproducible from committed fixtures and deterministic seeds.
- The package never states that a confidence interval gives a posterior probability,
  that a p-value is the probability the null is true, or that correlation proves cause.
- Learner evidence remains intentionally unpopulated.

## Open decision

The learner must complete `adr_prompt.md`: choose and defend an operational
inference policy, including its comparison-family control and revisit conditions.
