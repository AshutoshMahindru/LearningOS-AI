# WP-006: Shared-File Ownership & Parallel Safety Matrix

## Controlling G3 ownership

This matrix supersedes legacy path references for the G3 implementation tranche. Every lane starts from the same authorized integration SHA and works in an isolated worktree.

| Lane / owner | Allowed paths | Prohibited or integrator-owned paths |
|---|---|---|
| 11A — Tooling / CI / Launcher | `.github/workflows/platform-ci.yml`; `start.sh` or its replacement; `tools/platform/**`; `tests/platform/tooling/**`; root V3 tooling manifests and lock files | Product runtime implementation; frozen architecture; curriculum and learner state |
| 11B — Frontend Foundation | `platform/frontend/**` and frontend-local tests | Backend, storage, worker, curriculum, root launcher/workflows |
| 11C — API / Security | `platform/backend/app/main.py`; `platform/backend/app/api/**`; `platform/backend/app/models/schemas.py`; `platform/backend/app/core/config.py`; `errors.py`; `security.py`; `version.py`; `platform/backend/tests/api/**` | Database/migrations, artifact/ledger services, worker/curriculum internals, frontend |
| 11D — Storage / Ledger / Artifacts | `platform/backend/app/db/**`; storage migrations/services; `platform/backend/app/core/artifact_store.py`; ledger/backup models; `platform/backend/tests/storage/**` | API router/main, frontend, worker/curriculum, frozen architecture DDL |
| 11E — Worker / Curriculum | `platform/worker/**`; consolidation/removal of `platform/backend/worker_daemon.py`; `platform/backend/app/core/worker_client.py`; `mission_loader.py`; `registry.py`; worker/curriculum tests and fixture package | API router/main, storage, frontend, real M01–M42 migration, frozen schemas |
| G3 Integrator | Router/application assembly; final API-client contract reconciliation; launcher service wiring; `tests/platform/integration/**` | Redesigning frozen architecture or hiding lane defects |

## Frozen and prohibited paths

Implementation lanes must not modify `architecture/**`, `missions/**`, `labs/**`, `datasets/**`, `tracking/**`, `data/lab_status.json`, legacy `learning_os/**`, or legacy `web/**`. A required shared-core change must be raised to Master Control; a mission-specific workaround is never permitted.

## Collision rules

- Backend dependency changes are coordinated through 11A/Integrator when two lanes need the same manifest.
- Implementers do not merge their own work.
- Any file outside a lane's allowed set requires written Master Control disposition before editing.
- Any commit added after independent review invalidates that review.
