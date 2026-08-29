# WP-116: No-AI Certification Policy & Provenance Protocol

## 1. Objective of No-AI Certification
No-AI Certification proves that the learner possesses internal conceptual understanding and implementation ability without relying on external generative models, AI completions, or interactive hints.

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
4. **Decertification on Breach**:
   - If network telemetry indicates an attempt to access external LLM APIs during a locked test, the stage attempt is marked `INVALIDATED` and requires a new fresh-case challenge.
