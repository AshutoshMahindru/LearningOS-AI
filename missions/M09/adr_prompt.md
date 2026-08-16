# M09 ADR Prompt — Binary Classification Operating Threshold

Use `templates/ADR.md` to create a separate learner-authored decision record
after completing the threshold experiments. The notebook result is evidence,
not approval. Do not copy a threshold merely because one candidate has the
lowest toy cost.

This repository prompt must remain unfilled. The empty response blocks make
that boundary testable; put the completed ADR in the learner's evidence
location rather than turning this prompt into a fabricated completion record.

## Decision

State the positive event and triggered action, selected operating threshold,
exact comparison rule, decision owner, effective date and policy version.
Explain how the chosen threshold respects operating capacity.

<!-- BEGIN LEARNER RESPONSE: DECISION -->
<!-- END LEARNER RESPONSE: DECISION -->

## Context

Describe the decision population, class balance, action capacity, outcome
delay, and who experiences each consequence. Declare false-positive and
false-negative costs with their units, sources and uncertainty. Distinguish
the model's probability estimate from the threshold policy.

<!-- BEGIN LEARNER RESPONSE: CONTEXT -->
<!-- END LEARNER RESPONSE: CONTEXT -->

## Alternatives considered

Compare at least the default `0.50` threshold, a lower-recall/higher-precision
policy, a higher-recall/lower-precision policy, a capacity-constrained policy,
and no automated outreach. Include a reason for rejecting or deferring each
alternative; accuracy alone is not a sufficient reason.

<!-- BEGIN LEARNER RESPONSE: ALTERNATIVES -->
<!-- END LEARNER RESPONSE: ALTERNATIVES -->

## Evidence

Use held-out TP, TN, FP and FN counts; accuracy, precision and recall; action
volume; and consequence-cost calculations for material alternatives. Include
the majority baseline, calibration limitations and any assumptions not
established by the synthetic fixture. Separate observations from judgment.

<!-- BEGIN LEARNER RESPONSE: EVIDENCE -->
<!-- END LEARNER RESPONSE: EVIDENCE -->

## Trade-offs

Explain the accepted FP/FN balance, capacity load, people affected, uncertainty
in cost assumptions, calibration and distribution-shift exposure, and why the
decision is not a universal best threshold. Name rollback behavior if capacity
or harm exceeds the declared limit.

<!-- BEGIN LEARNER RESPONSE: TRADE_OFFS -->
<!-- END LEARNER RESPONSE: TRADE_OFFS -->

## Revisit conditions

Give measurable triggers such as a changed FP/FN cost ratio, changed operating
capacity, prevalence or score-distribution drift, recall below a declared
floor, calibration degradation, or enough mature outcomes to reverse the
candidate ranking. Assign an owner and review cadence.

<!-- BEGIN LEARNER RESPONSE: REVISIT_CONDITIONS -->
<!-- END LEARNER RESPONSE: REVISIT_CONDITIONS -->

## Status

Choose `Proposed`, `Accepted`, `Superseded` or `Rejected`, with owner and date.
A generated notebook result cannot by itself make the status `Accepted`.

<!-- BEGIN LEARNER RESPONSE: STATUS -->
<!-- END LEARNER RESPONSE: STATUS -->
