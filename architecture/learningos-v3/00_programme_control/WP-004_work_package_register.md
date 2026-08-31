# WP-004: Master Work Package Register

## Overview
This register tracks all parent and child work packages across the LearningOS V3 programme lifecycle, mapping each to its gate, ownership, dependencies, and execution status.

| WP ID | Name | Parent WP | Gate | Owner | Status | Dependencies |
|---|---|---|---|---|---|---|
| **WP-001** | Freeze and tag V2 prototype | WP-000 | G0 | Lead Arch | **DONE** | None |
| **WP-002** | Record prototype findings & defects | WP-000 | G0 | Lead Arch | **DONE** | WP-001 |
| **WP-003** | Create V3 branch & repo structure | WP-000 | G0 | Lead Arch | **DONE** | WP-001 |
| **WP-004** | Master work-package register | WP-000 | G0 | Lead Arch | **DONE** | WP-003 |
| **WP-005** | Architecture decision register (ADRs) | WP-000 | G0 | Lead Arch | **DONE** | WP-003 |
| **WP-006** | Shared-file ownership matrix | WP-000 | G0 | Lead Arch | **DONE** | WP-003 |
| **WP-007** | Branch and PR policy | WP-000 | G0 | Lead Arch | **DONE** | WP-003 |
| **WP-008** | Risk, issue, and change registers | WP-000 | G0 | Lead Arch | **DONE** | WP-003 |
| **WP-009** | Programme dashboard | WP-000 | G0 | Lead Arch | **DONE** | WP-004 |
| **WP-010** | Release & versioning convention | WP-000 | G0 | Lead Arch | **DONE** | WP-003 |
| **WP-111** | Product constitution | WP-100 | G1 | Product Lead | **DONE** | G0 |
| **WP-112** | User and usage model | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-113** | Target learner journey | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-114** | Product surface architecture | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-115** | Assistance policy | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-116** | No-AI certification policy | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-117** | Product definition of done | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-118** | Out-of-scope register | WP-100 | G1 | Product Lead | **DONE** | WP-111 |
| **WP-121** | Canonical stage vocabulary | WP-100 | G2 | Pedagogy Lead | **DONE** | G1 |
| **WP-122** | Mission runtime state model | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-123** | Experiment pedagogy contract | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-124** | Controlled-failure model | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-125** | Transfer-assessment model | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-126** | Targeted-repair model | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-127** | Retention and decay model | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-128** | Progressive-autonomy model | WP-100 | G2 | Pedagogy Lead | **DONE** | WP-121 |
| **WP-131** | System context architecture | WP-100 | G2 | Tech Lead | **DONE** | G1 |
| **WP-132** | Container & service architecture | WP-100 | G2 | Tech Lead | **DONE** | WP-131 |
| **WP-133** | Local-first storage architecture | WP-100 | G2 | Tech Lead | **DONE** | WP-131 |
| **WP-134** | Relational data model & migrations | WP-100 | G2 | Tech Lead | **DONE** | WP-133 |
| **WP-135** | Generic OpenAPI API contract | WP-100 | G2 | Tech Lead | **DONE** | WP-132 |
| **WP-136** | Mission definition schema (MDL v1) | WP-100 | G2 | Tech Lead | **DONE** | WP-121 |
| **WP-137** | Structured result schema | WP-100 | G2 | Tech Lead | **DONE** | WP-123 |
| **WP-138** | Evidence and gate architecture | WP-100 | G2 | Tech Lead | **DONE** | WP-134 |
| **WP-139** | Tutor and provider architecture | WP-100 | G2 | Tech Lead | **DONE** | WP-115 |
| **WP-140** | Execution worker protocol | WP-100 | G2 | Tech Lead | **DONE** | WP-132 |
| **WP-141** | Security architecture | WP-100 | G2 | Tech Lead | **DONE** | WP-140 |
| **WP-142** | Backup, migration & recovery | WP-100 | G2 | Tech Lead | **DONE** | WP-133 |
| **WP-143** | Desktop packaging strategy | WP-100 | G2 | Tech Lead | **DONE** | WP-132 |
| **WP-150** | Cross-mission architecture proof | WP-100 | G2 | Lead Arch | **DONE** | WP-136, WP-137 |
| **WP-160** | G2 architecture freeze review | WP-100 | G2 | Programme Lead | **DONE** | WP-150 |
| **WP-200** | Platform foundation | Post-G2 | G3 | G3 Integration Lead | **ACTIVE — AUTHORIZED, NOT ACCEPTED** | G2 PASSED |
| **WP-300** | Schema-driven mission runtime | Post-G2 | G4 | Platform Core Lead | **BLOCKED** | G3 PASS |
| **WP-400** | Execution, workbench, evidence and assessment | Post-G2 | WP400 acceptance | Platform Core Lead | **BLOCKED** | G4 PASS |
| **WP-500** | Reference missions M01–M05 | Post-G2 | G5 | Curriculum Lead | **BLOCKED** | WP400 accepted |
| **WP-600** | Tutor, learner model and flagship integration | Post-G2 | G6 | Adaptive Learning Lead | **BLOCKED** | G5 PASS |
| **WP-700** | Curriculum migration M06–M42 | Post-G2 | G6 | Curriculum Migration Lead | **BLOCKED** | G5 PASS |
| **WP-800** | Productisation, migration and release | Post-G2 | G7–G8 | Release Lead | **BLOCKED** | G6 PASS |
