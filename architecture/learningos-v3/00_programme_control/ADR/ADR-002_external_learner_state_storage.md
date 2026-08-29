# ADR-002: External Local-First Storage Architecture for Learner State

## Status
ACCEPTED (Controlling baseline for LearningOS V3)

## Context
LearningOS V2 committed learner progress, evidence files, and experiment logs directly inside the Git repository worktree under `tracking/`. This coupled user data with curriculum source code, caused merge conflicts, and violated privacy.

## Decision
1. **Isolated Data Directory**: All learner state, database instances, session recordings, execution logs, and generated artifacts will reside strictly outside the curriculum repository, defaulting to `~/.learningos/` (configurable via `LEARNINGOS_HOME`).
2. **Relational Database**: A local SQLite database (`~/.learningos/learningos.db`) manages learners, sessions, stage attempts, evidence claims, and competency graphs with WAL mode and foreign key constraints enabled.
3. **Artifact Repository**: Binary artifacts (model weights, charts, export bundles) reside in `~/.learningos/artifacts/` keyed by SHA-256 content hashes.

## Consequences
- Positive: Clean separation of immutable curriculum source code from mutable learner data.
- Positive: Curriculum updates can be pulled/rebased cleanly without wiping learner history.
- Positive: Learner profiles can be exported, backed up, or migrated independently.
