# V11 integration — open evaluation, do not close the flagship

## M34/M39 → M40 boundary

M34 already grounds generation. M39 already wraps M38 with memory,
routes, and fallbacks. M40 is the first V11 mission and **owns
evaluation**:

```
M34 answer_labeled / verify_support / evaluate_set
M39 run_robust_task / retrieved_ids / route / degraded
  -> versioned eval pack
  -> deterministic graders
  -> calibrated rubric boundary
  -> slices, proxies, traces
  -> release gates
```

The observable V11 surface after M40 is a small eval harness: twelve
frozen cases, four invariant graders, a fail-closed judge adapter,
and gates M41 can cite. It is not a production eval platform.

## What M40 must not change

M40 does not design the integrated architecture (M41). It does not
retune M34 holdout questions or M39 route predicates to raise
averages. It does not import OpenAI, Anthropic, LangGraph, or paid
eval SDKs. It does not edit M34, M37, M38, or M39. It does not mark
the mission repository-executable.

## Phase-end honesty

P7 lists M40 as `phase_end: true` because evaluation is the last
Agents & Evaluation skill. **V11 does not close because this package
exists.** Flagship V11 also includes M41. Learner evidence, the ADR,
and formal review remain unfilled. Implementation status is not
learner completion.

## M40 → M41 / M42 handoff

M41 may design architecture only after the learner can defend:

- eval cases are versioned and not tuned against
- objective invariants use deterministic graders
- critical failures have explicit gates
- traces support diagnosis after a field is missing
- optional judges are calibrated and not sole graders
- V11 evaluation is reusable, not finished

Reusable artifacts: `load_eval_pack` / `run_suite`, the four
`grade_*` functions, `decide_release_gate`, `pipeline_with_defect` /
`repair_run`, and `handoff_contract()`.
