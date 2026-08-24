# Code reading — pack load, invoke, graders, slices, gate

Read `load_eval_pack`, `invoke_case`, `grade_tool_schema`,
`grade_citation_support`, `grade_state_termination`,
`grade_idempotency`, `aggregate_report`, `decide_release_gate`,
`ablate_trace`, and `repair_run` in
`missions/M40/evaluation_harness.py`. M40's code-reading target is the
**eval harness** around M34 and M39:

1. `load_eval_pack` versions the suite and can refuse a contaminated freeze
2. `invoke_case` dispatches to `missions.M34.rag_pipeline` or
   `missions.M39.robust_agent` (or M37 for schema/idempotency probes)
3. Deterministic graders localize issue kind and object id
4. `aggregate_report` separates outcome success from grader slices
5. `decide_release_gate` can fail the ship on critical rate even when
   the average is high
6. `ablate_trace` names diagnoses that a missing field blocks
7. `repair_run` recomputes governance from the broken object

Before running the code-reading cell, predict:

- what `load_eval_pack(..., require_canonical=True)` does with
  `m40.eval.tuned-dev`
- the fail-reason strings from `decide_release_gate` on a report with
  high outcome success and one critical citation fail
- which diagnosis `ablate_trace(..., "used_memory_ids")` blocks
- whether `optional_llm_judge` can be the sole required grader

Do **not** look for a Braintrust/Promptfoo SDK, an M41 architecture
diagram, or a retune of M34 holdout ids. If a failure can be diagnosed
from `case_ids`, `fail_reasons`, or slice rates, stay at that boundary.

Do not print substring membership of a later helper. Probe the live
objects: case ids, gate fail reasons, slice rates.
