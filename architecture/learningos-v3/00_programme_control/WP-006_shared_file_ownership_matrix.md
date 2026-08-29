# WP-006: Shared-File Ownership & Parallel Safety Matrix

## Overview
To prevent merge conflicts and inconsistent architecture changes during parallel development across multiple agents and workstreams, every subsystem, contract, and directory has a strictly designated owner.

| Subsystem / Path | Designated Owner | Secondary Reviewer | Rule |
|---|---|---|---|
| `architecture/` | Lead Architect | Programme Lead | Shared platform contracts require ADR for any change |
| `schemas/` (MDL & Results) | Platform Core Lead | Lead Architect | Single source of truth; breaking changes require version bump |
| `learning_os/core/` (API & DB) | Backend Tech Lead | Platform Core Lead | Core backend shared runtime |
| `learning_os/worker/` | Runtime Worker Lead | Backend Tech Lead | Execution isolation and sandboxing |
| `web/` (React Frontend Shell) | Frontend Tech Lead | UI/UX Lead | Zero mission-specific code allowed |
| `missions/` (Curriculum YAMLs) | Curriculum Migration Lead | Pedagogy Lead | Mission-local modifications only; cannot alter platform schemas |
| `tests/contract/` | QA & Contract Lead | Tech Lead | Contract tests run on every PR |
| `tests/missions/` | Curriculum Migration Lead | QA Lead | Mission-specific validation suites |

## Parallel Safety Invariant
No mission agent or author may modify shared platform files (`schemas/`, `learning_os/core/`, `web/src/components/stages/`) to accommodate a single mission. Any capability gap must be filed as a Platform Change Request (CR) and reviewed by the Lead Architect.
