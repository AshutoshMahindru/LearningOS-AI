# WP-007: Branch and Pull Request Policy

## Gate-integration strategy

No implementation is committed directly to `main`. Each gate uses bounded implementation branches that fan into one gate integration branch, followed by independent review and one authorized gate merge.

```text
main @ accepted gate baseline
  └── integration/v3-g3-platform
        ├── v3/g3/tooling-ci
        ├── v3/g3/frontend-foundation
        ├── v3/g3/api-security
        ├── v3/g3/storage-ledger
        └── v3/g3/worker-curriculum
```

## Pull-request lifecycle

1. Master Control records the exact authorized base SHA and path ownership before a lane starts.
2. Each lane uses an isolated worktree and a dedicated branch from that same SHA.
3. Each lane opens a draft PR to the integration branch; lane PRs never target `main`.
4. Implementers test and report but do not merge.
5. The integrator verifies merge base, changed paths, acceptance evidence and CI before combining lanes.
6. An independent reviewer binds the gate decision to the exact integration commit and tree.
7. Any subsequent change requires a new review.
8. Only Master Control may authorize the integration-to-`main` PR and merge.

## Required checks

- backend static/unit/integration tests;
- frontend lint, type-check and production build;
- frozen contract and cross-mission schema validation;
- gate-specific lifecycle smoke tests;
- repository consistency validation where applicable.

## Temporary manual safeguards

At R0, GitHub reported `main` and all branches as unprotected and no rulesets existed. Until server-side protection is configured, the root clone remains read-only, worktrees are isolated, PR targets and exact SHAs are verified manually, force-pushes and direct-main updates are prohibited, and Master Control is the sole merge authorizer. This exception blocks G3 merge to `main` unless closed or explicitly re-authorized at gate review.
