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
| **WP-111** | Product constitution | WP-100 | G1 | Product Lead | **ACTIVE** | G0 |
| **WP-112** | User and usage model | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-113** | Target learner journey | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-114** | Product surface architecture | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-115** | Assistance policy | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-116** | No-AI certification policy | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-117** | Product definition of done | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-118** | Out-of-scope register | WP-100 | G1 | Product Lead | **ACTIVE** | WP-111 |
| **WP-121** | Canonical stage vocabulary | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | G1 |
| **WP-122** | Mission runtime state model | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-123** | Experiment pedagogy contract | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-124** | Controlled-failure model | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-125** | Transfer-assessment model | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-126** | Targeted-repair model | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-127** | Retention and decay model | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-128** | Progressive-autonomy model | WP-100 | G2 | Pedagogy Lead | **ACTIVE** | WP-121 |
| **WP-131** | System context architecture | WP-100 | G2 | Tech Lead | **ACTIVE** | G1 |
| **WP-132** | Container & service architecture | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-131 |
| **WP-133** | Local-first storage architecture | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-131 |
| **WP-134** | Relational data model & migrations | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-133 |
| **WP-135** | Generic OpenAPI API contract | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-132 |
| **WP-136** | Mission definition schema (MDL v1) | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-121 |
| **WP-137** | Structured result schema | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-123 |
| **WP-138** | Evidence and gate architecture | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-134 |
| **WP-139** | Tutor and provider architecture | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-115 |
| **WP-140** | Execution worker protocol | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-132 |
| **WP-141** | Security architecture | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-140 |
| **WP-142** | Backup, migration & recovery | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-133 |
| **WP-143** | Desktop packaging strategy | WP-100 | G2 | Tech Lead | **ACTIVE** | WP-132 |
| **WP-150** | Cross-mission architecture proof | WP-100 | G2 | Lead Arch | **ACTIVE** | WP-136, WP-137 |
| **WP-160** | G2 architecture freeze review | WP-100 | G2 | Programme Lead | **ACTIVE** | WP-150 |
| **WP-200..800**| Platform construction & release | Post-G2 | G3–G8 | Various | **BLOCKED** | G2 Sign-off |
