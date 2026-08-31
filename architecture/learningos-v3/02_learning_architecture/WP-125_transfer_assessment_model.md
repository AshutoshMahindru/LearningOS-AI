# WP-125: Transfer-Assessment Model (Fresh-Case & Isolation Protocol)

## 1. Objective of Transfer Assessment
Transfer assessment tests whether a learner has generalized the core concept or merely memorized the specific numbers, variable names, and dataset features of the guided lab.

## 2. Requirements for Transfer Tasks
1. **Structural Isomorphism**: The mathematical or algorithmic mechanism remains identical, but the domain, feature schema, or API interface is completely fresh.
2. **Strict Answer Isolation**: Test fixtures and verification code are not readable by the learner in the sandbox environment before submission.
3. **No-AI Lock**: LearningOS tutor, hints, completions, prior-answer surfaces, and other in-product assistance are disabled for the recorded attempt. External assistance outside the product boundary is not technically certified; see [ADR-006](../00_programme_control/ADR/ADR-006_no_ai_assistance_boundary_erratum.md).
4. **Deterministic Rubrics**: Automated evaluation asserts correct shapes, invariants, error bounds, and algorithmic complexity.
