# WP-009: Programme Dashboard & Gate Status

## Gate Progression Summary

```
[G0: Controls Active] ──► [G1: Constitution] ──► [G2: Arch Freeze] ──► [G3: Foundation] ──► [G4: Runtime] ──► [G5: M01-M05] ──► [G6: M06-M42] ──► [G7: RC] ──► [G8: Release]
      (COMPLETED)               (COMPLETED)           (COMPLETED)           (BLOCKED)          (BLOCKED)         (BLOCKED)          (BLOCKED)        (BLOCKED)       (BLOCKED)
```

| Gate | Meaning | Status | Target Date | Sign-off Owner |
|---|---|---|---|---|
| **G0** | Prototype frozen & programme controls active | **PASSED** | 2026-08-29 | Lead Architect |
| **G1** | Product constitution approved | **PASSED** | 2026-08-29 | Product Lead |
| **G2** | Technical & product architecture frozen | **PASSED** | 2026-08-29 | Lead Architect & Tech Lead |
| **G3** | Platform foundation accepted | PENDING (Blocked on WP-200) | TBD | Tech Lead |
| **G4** | Generic mission runtime accepted | PENDING (Blocked on WP-300) | TBD | Core Lead |
| **G5** | Reference missions M01–M05 accepted | PENDING (Blocked on WP-500) | TBD | QA Lead |
| **G6** | M06–M42 curriculum migration complete | PENDING (Blocked on WP-700) | TBD | Migration Lead |
| **G7** | Release candidate accepted | PENDING (Blocked on WP-800) | TBD | Release Lead |
| **G8** | Stable release | PENDING (Blocked on WP-800) | TBD | Programme Lead |

## R0 reconciliation and active tranche

- Authorized source baseline: `main@1fe547048a3937429ebfb9c2c3c8148620772f6d`.
- WP-200 is authorized and active but not accepted; G3 remains PENDING.
- Existing `platform/**` code and `feature/wp-200-platform-foundation@b2d0dd4` are acceleration material only.
- Active integration branch: `integration/v3-g3-platform`.
- Active lanes: 11A Tooling/CI, 11B Frontend, 11C API/Security, 11D Storage/Ledger, 11E Worker/Curriculum.
- PR #61 was retired as superseded V2 product-discovery evidence; its branch/head is retained.

## Open governance exceptions

1. The local annotated `v2-prototype-freeze` tag points to `35293455ff769855014a49fa56315b22829e74b1`, but remote tag publication was unavailable in the execution environment. The V2 branch remains pinned to that SHA. Close before G3 merges to `main`.
2. GitHub branch protection/rulesets were unavailable through the connected interface. Manual controls in WP-007 apply. Close or explicitly re-authorize at G3 review.
