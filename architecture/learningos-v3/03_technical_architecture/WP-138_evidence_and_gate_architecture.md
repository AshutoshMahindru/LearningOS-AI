# WP-138: Evidence and Gate Architecture

## 1. Evidence Provenance & Cryptographic Graph
Every piece of accepted evidence forms a tamper-evident node linking 5 distinct dimensions:
1. **Learner Identity**: `learner_id`
2. **Attempt Hash**: `stage_attempt_id`
3. **Execution Artifact**: SHA-256 hash of output payload / file
4. **Test Suite / Runner**: SHA-256 hash of execution test harness
5. **Curriculum Commit**: Git SHA of the mission specification

## 2. Gate Evaluation Engine
The Gate Evaluator tests the submitted evidence against the mission's `gate_contract`:
- If `passed_criteria >= pass_threshold`: Returns `status: "PASSED"`, awards competency level increments, and unlocks the next mission.
- If `passed_criteria < pass_threshold`: Returns `status: "REPAIR_REQUIRED"` accompanied by a precision `repair_plan` detailing the failed knowledge node and providing an immediate targeted drill.
