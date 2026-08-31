# G3 Execution Delta Register

## Authorization

- Repository: `AshutoshMahindru/LearningOS-AI`
- Audited source baseline: `1fe547048a3937429ebfb9c2c3c8148620772f6d`
- Formal position: G0–G2 PASSED; WP-200 authorized but not accepted; G3 PENDING.
- Integration branch: `integration/v3-g3-platform`.
- Lane base: the exact reviewed governance-bootstrap commit merged into the integration branch.
- Unaccepted reference: `feature/wp-200-platform-foundation@b2d0dd41164528e68c8668f92901cdc31c00e108`; inspect only, never merge or cherry-pick wholesale.

## R0 decision

The repository is ready for five bounded G3 implementation lanes after this control bootstrap is reviewed and applied to the integration branch. No current WP-200 item is gate-accepted.

## Retain, complete, replace and quarantine

| Area | R0 disposition |
|---|---|
| Frozen `architecture/learningos-v3/**` contracts | Retain; changes require approved ADR/erratum. |
| Platform/curriculum/test directory separation | Retain and complete as WP-211. |
| Generic React shell | Retain and complete; current client/design/a11y/error handling are incomplete. |
| FastAPI application seed | Retain as a seed; replace placeholder health/errors/auth/config/secrets behavior. |
| External SQLite/WAL/foreign-key seed | Retain; add `LEARNINGOS_HOME`, migrations, ledger, artifacts and backup/restore. |
| Worker IPC skeleton | Retain concept; consolidate two daemons and implement lifecycle/recovery. |
| Stage, execution, evidence, gate, tutor and workbench spikes | Quarantine to WP-300/WP-400/WP-600; do not count toward G3. |
| Dummy hashes, hard-coded provenance, direct provider coupling | Remove from the accepted G3 path or quarantine behind later interfaces. |

## Exact WP-200 delta

| Lane | Work packages | Current position | Required G3 outcome |
|---|---|---|---|
| 11A | WP-211, 214, 216 | Structure partial; launcher spike; V3 CI missing | Coherent tooling, preflight, one-command launch/shutdown, backend/frontend/contracts/smoke CI, no learner state in Git. |
| 11B | WP-221, 222, 224, 226 | Shell partial; design/client/a11y incomplete | Generic shell, reusable components, typed authenticated client, global states and accessibility tests; zero mission-specific logic. |
| 11C | WP-231, 233, 234, 236 | API skeleton; stable errors/auth/config/secrets missing | Live health/version/config, typed errors, loopback session protection, server-side secrets and comprehensive tests. |
| 11D | WP-241, 242, 244, 245, 248 | SQLite partial; remaining capabilities missing | External data home, transactional migrations, append-only ledger, atomic checksum store, restart durability and exact backup/restore. |
| 11E | WP-251, 252, 254, 255, 257 | Loader/worker spikes | Versioned fixture package discovery/integrity plus canonical worker health, cancel, stop, restart and crash recovery. |

The path matrix in WP-006 is controlling. Cross-lane wiring is Integrator-owned.

## G3 acceptance scenario

Clean start -> create external LearningOS home -> load fixture curriculum -> create learner/session -> store artifact -> restart with state intact -> kill/restart worker without database corruption -> backup -> restore into a clean home -> confirm offline core operation and frontend secret isolation.

G3 does not pass because code exists. The complete scenario, lane tests, integrated tests and independent review must pass at one recorded integration SHA.

## Active risks and constraints

1. Frontend/API authentication bootstrap must not expose provider secrets or an unauthenticated bearer token.
2. API/storage/worker lanes consume shared interfaces; they must not create parallel schemas or daemons.
3. Stage runtime, runner security, evidence/gates and tutor behavior remain later work.
4. Offline operation must not depend on remote fonts or services.
5. Version drift must converge on one platform version source.
6. Root clone is read-only; every lane uses an isolated worktree and draft PR to integration.

## Governance exceptions

- The V2 freeze head is preserved at `feature/learningos-app-v1@35293455ff769855014a49fa56315b22829e74b1`; remote tag publication remains open because the current environment lacks tag-ref write capability.
- GitHub reported no branch protection or rulesets. Manual safeguards in WP-007 apply.

These exceptions do not authorize hidden workarounds. They must be closed or explicitly dispositioned by the independent G3 review before merge to `main`.
