# M10 Formal Engineering Review Brief

## Review objective

Defend the V02 metric and threshold policy as a consequential engineering decision. The review is adversarial but constructive; it evaluates reasoning and evidence, not presentation polish.

## Architecture and meaningful artifact

Architecture: fixed risk score → versioned threshold → inspect / do-not-inspect action → observed consequences → monitoring.

Meaningful artifact: `labs/M10_metrics_consequences.ipynb`, the validation/test datasets, the completed threshold ADR prompted by `adr_prompt.md`, and the controlled-failure diagnosis.

## Required walkthrough

1. State the positive event and action before naming a metric.
2. Defend TP, FP, TN and FN orientation in operational language.
3. Show validation prevalence and the all-negative baseline.
4. Reproduce the accuracy-controlled failure and quantify harmed utility.
5. Compare accuracy-, F1-, and cost-selected thresholds.
6. Explain ROC and PR views without treating either curve as a deployed policy.
7. Show the validation-only selection boundary and one-time test result.
8. Present sensitivity to FN cost and the inspection-capacity constraint.
9. Defend the ADR's decision, owner, monitoring and revisit conditions.

## Principal-engineer challenge prompts

- Who owns the 2:18 cost ratio, and what evidence supports it?
- Which omitted consequence is most likely to reverse the decision?
- Why is the score trustworthy enough to threshold? Is it calibrated, or only ranked?
- Why not optimize F1, recall at a precision floor, or cost under a capacity constraint?
- How does prevalence shift change precision, alert volume and realized cost?
- How will delayed outcomes and censored failures affect monitoring?
- Which metric can look healthy while the business or safety outcome worsens?
- What deterministic policy and rollback behavior exists when monitoring is stale?

## Required uncertainty

At least one unresolved uncertainty must be quantified or bounded. Examples include cost estimation error, test-window sampling noise, outcome delay, subgroup error, or future prevalence shift. “More data is needed” without a decision impact is insufficient.

## Review comment disposition

Record every actionable comment with exactly one disposition:

| Comment | Evidence considered | Disposition (Accept / Reject / Defer) | Written reasoning | Owner / due condition |
|---|---|---|---|---|

Never auto-implement review feedback. Accepted comments require a verified change; rejected comments require evidence; deferred comments require a trigger and owner.

## Exit criteria

- The controlled failure is reproduced and correctly diagnosed.
- The threshold traces to stated consequences and validation evidence.
- Test evidence was not used to tune policy.
- The ADR is complete and exposes trade-offs and revisit conditions.
- Every reviewer comment is dispositioned in writing.
- One material uncertainty remains visible rather than being disguised by a scalar metric.
