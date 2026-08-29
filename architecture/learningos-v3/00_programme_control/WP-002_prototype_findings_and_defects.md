# WP-002: V2 Prototype Findings, Defects & Architectural Lessons

## Metadata
- **Work Package**: WP-002
- **Status**: COMPLETED
- **Gate**: G0
- **Inputs**: Branch `feature/learningos-app-v1`, PR #61, M01 Prototype Experience

## Executive Summary
The V2 prototype successfully validated key pedagogical concepts (predict &rarr; commit &rarr; run &rarr; observe &rarr; explain, Socratic questioning, and whole-system tracing). However, it revealed severe structural limits that preclude it from scaling across M01–M42.

## Key Discovery Findings (What Worked)
1. **Whole-System First Pedagogy**: Learners gain immediate mental scaffolding by tracing and mapping the full architecture before drilling into components.
2. **Prediction Gating**: Forcing learners to commit predictions to an immutable record before running code dramatically increases focus and retention.
3. **Structured Observability**: Graphing traces, state diffs, and output tables provides much better insight than raw stdout.
4. **Targeted Failure Injection**: Injecting controlled faults and requiring diagnosis tests real architectural understanding.

## Architectural Defects & Structural Limits (What Failed)
1. **Mission-Specific Coupling**:
   - `m01_experience.py` and `web/m01.html` hardcoded M01-specific logic and UI routes (`/api/m01/...`). Scaling this to 42 missions would require 42 distinct backend modules and 42 custom HTML/JS frontends.
2. **Learner State in Git Worktree**:
   - Tracking files (`tracking/progress.json`, `tracking/evidence_ledger.jsonl`) were committed inside the curriculum repository, causing git dirty states, merge conflicts, and privacy leakage.
3. **Notebook-as-UX Anti-Pattern**:
   - Relying on raw Jupyter notebook rendering exposed learners to fragile execution kernels, unordered execution bugs, and clutter.
4. **Generic Four-Flag Gate Contract**:
   - Gates were treated as superficial boolean checkmarks rather than cryptographically linked, rubric-validated evidence models with automated provenance.
5. **Client-Side Assistance Bypass**:
   - No-AI restrictions could be bypassed in the browser because assistance policy was not strictly enforced at the backend/runtime execution boundary.

## V3 Architectural Mandates Arising from V2
- **Schema-Driven Rendering**: 100% of missions render via reusable stage components defined by JSON/YAML schemas.
- **External Local-First Storage**: All learner data, session states, and evidence ledgers live outside the Git repository in `~/.learningos/`.
- **Runtime-Enforced Gate & No-AI Policies**: The API server and execution worker enforce stage locks, prediction seals, and assistance rules.
