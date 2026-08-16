# M20 Formal Engineering Review Brief

## Review decision requested

Approve M20 as the deterministic optimization-instrumentation layer that closes V04
and establishes the monitored training-policy handoff to M21. Approval covers package
correctness and pedagogy; it does not certify learner completion or choose the ADR.

## System under review

- **Objective boundary:** one transparent anisotropic quadratic with fixed curvature.
- **Gradient boundary:** the analytic mechanism inherited from M19 remains unchanged.
- **Optimizer boundary:** exact GD, seeded component-gradient SGD, classical momentum,
  and bias-corrected Adam expose their state and applied updates.
- **Interaction boundary:** a learner prediction cell precedes every experiment cell.
- **Failure boundary:** only learning rate changes in the stagnation/divergence failures.
- **Decision boundary:** optimizer selection is deferred to an unfilled learner ADR.

## Evidence available to reviewers

1. Deterministic reference traces cover too-small, stable, oscillatory, and too-large
   exact-gradient learning-rate regimes from the same start.
2. The zero-mean component-gradient fixture preserves the aggregate objective; equal
   seeds reproduce identical SGD traces and different seeds alter update order.
3. Momentum and Adam traces preserve their complete state and make overshoot, scale
   adaptation, tuning, and finite-horizon trade-offs observable.
4. Mission tests validate numeric mechanics, comparison controls, notebook order and
   hygiene, empty learner evidence, ADR fields, and the M19→M20→M21 handoff.

## Material assumptions

- The quadratic is an instrumentation fixture, not a claim about all loss landscapes.
- Fixed-budget final loss is one comparison metric; compute cost and robustness also matter.
- Seeded component perturbations illustrate SGD noise but do not model every batch process.
- Hyperparameters with the same numeric value need not be equally tuned across optimizers.
- A finite trace supports a bounded observation, not proof of asymptotic convergence.

## Risks and controls

| Risk | Consequence | Control | Residual risk |
| --- | --- | --- | --- |
| Final loss becomes a universal ranking | wrong default optimizer | require objective, horizon, hyperparameters, and multiple trace metrics | fixture remains simple |
| Oscillation mistaken for divergence | unnecessary rollback | inspect sign and magnitude trends together | noisy traces need smoothing judgment |
| SGD randomness hides reproducibility | irreproducible evidence | fixed seed plus seed-sensitivity run | real parallel kernels may differ |
| Optimizer state hides a gradient bug | wrong diagnosis | retain gradient and applied update separately | production frameworks add state |
| Implementation marked as learner completion | fabricated readiness | learner evidence and ADR remain unfilled | process enforcement remains human |

## Required reviewer challenges

- Recalculate the exact-gradient stability boundary for both coordinates.
- Independently reproduce at least one smooth, one oscillatory, and one divergent trace.
- Verify component-gradient noise averages to zero and seed replay is exact.
- Compare momentum and Adam at two horizons or tunings and reject a universal ranking.
- Trace one optimizer step from gradient through state to next loss.
- Confirm every experiment has a prediction cell before its action cell.
- Confirm source notebook cells have stable unique IDs, null execution counts, empty
  outputs, and no network, paid API, secret, accelerator, or hidden learner evidence.

## Acceptance criteria

- Restart + Run All succeeds in a fresh CPU kernel with zero code-cell errors.
- Mission unittest and pytest, repository discovery, repository validator, and notebook
  validation are green in their documented environments.
- Failures isolate learning rate and their repairs preserve declared controls.
- Comparisons use the same objective and initialization where applicable.
- Learner predictions, scores, no-AI responses, evidence, sign-off, and ADR decisions
  remain intentionally unpopulated in source control.
- The closure document preserves both M19's gradient boundary and M21's training handoff.

## Open decision

The learner must complete `adr_prompt.md` and defend an optimizer/learning-rate policy,
monitoring plan, rollback threshold, and revisit triggers. The source prompt must stay
unfilled until that evidence exists.

