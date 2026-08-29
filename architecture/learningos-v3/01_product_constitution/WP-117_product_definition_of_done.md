# WP-117: Product Definition of Done (DoD)

## End-State Acceptance Criteria for LearningOS V3
A mission, work package, or release is considered **DONE** only when all of the following conditions are met:

1. **Schema Compliance**:
   - 100% of mission and stage definitions pass `learningos.mission.v1.json` schema validation without exceptions or custom schema extensions.
2. **Generic Rendering**:
   - Mission renders completely using only the standard 11 stage components from the generic stage registry. Zero custom React pages or custom API endpoints.
3. **Execution Safety & Structured Results**:
   - All code executions run inside the isolated subprocess worker with CPU/memory limits, producing valid Structured Result payloads (`table`, `chart`, `trace`, `diff`, `state`, `artifact`).
4. **Full Pedagogical Flow**:
   - Includes whole-system tracing, predict-commit-run-observe-explain cycle, code decomposition, controlled failure injection, and No-AI transfer assessment.
5. **Verifiable Gate Contract**:
   - Mission-specific competency gate passes only when required evidence artifacts meet rubric thresholds and are recorded with full provenance in the SQLite database.
6. **Data Isolation**:
   - Zero learner state or session logs are written to the Git worktree. All mutable state resides under `~/.learningos/`.
7. **Automated CI Validation**:
   - Unit tests, contract tests, and end-to-end mission verification pass with 0 warnings or errors.
