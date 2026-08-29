# WP-126: Targeted-Repair Model

## 1. Targeted Repair Principle
When a learner fails a competency gate or transfer assessment, forcing them to repeat an entire 90-minute mission causes frustration and disengagement. Instead, the gate evaluator generates a **Bounded Targeted Repair Task** targeting precisely the failed sub-competency.

## 2. Repair Engine Architecture
1. **Diagnosis**: Gate evaluator maps failed test assertions to specific knowledge graph nodes (e.g. `kn.m25.loss_backward` or `kn.m04.missing_imputation`).
2. **Task Generation**: The system dynamically synthesizes or selects an isolated micro-lab focusing exclusively on the missing invariant.
3. **Execution & Return**: The learner completes the 10–15 minute repair drill. Upon passing, they return directly to the competency gate evaluation without repeating unblocked stages.
