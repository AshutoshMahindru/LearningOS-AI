# WP-116: No-AI Certification Policy & Provenance Protocol

## 1. Objective of No-AI Certification
No-AI Certification records that LearningOS assistance was disabled during the recorded attempt. It supports an inference about independent performance from the captured evidence; it does not prove the absence of assistance from external devices, services, people, or unobserved channels. See [ADR-006](../00_programme_control/ADR/ADR-006_no_ai_assistance_boundary_erratum.md).

## 2. Enforcement Protocol
1. **Runtime Isolation**:
   - The frontend disables the Tutor sidebar, LLM keyboard shortcuts, and code-completion plugins.
   - The backend API blocks `/api/v1/tutor/*` and `/api/v1/assistance/*` for the active session.
2. **Fresh Unseen Cases**:
   - Transfer challenges utilize randomized datasets or novel architectural variants not present in prior guided stages.
3. **Evidence Tagging & Sign-off**:
   - Accepted transfer artifacts are stamped with:
     - `assistance_level: "NO_AI_CERTIFIED"`
     - `session_id`, `learner_id`, `stage_id`, `timestamp`
     - `code_hash`: SHA-256 hash of submitted code
     - `runner_hash`: SHA-256 hash of the execution test suite
     - `curriculum_sha`: Git commit SHA of the mission specification
4. **Invalidation within the LearningOS boundary**:
   - If LearningOS assistance is enabled, invoked, or its lock fails during the recorded attempt, the attempt is marked `INVALIDATED` and requires a new fresh-case challenge.
   - Network telemetry may support diagnostics or disclose a limitation, but LearningOS must not claim that it detects or blocks every external assistance channel.
