# WP-008: Risk, Issue, and Change Registers

## 1. Integrated Risk Register (Top Architectural Risks)

| Risk ID | Description | Severity | Likelihood | Mitigation Strategy | Owner |
|---|---|---|---|---|---|
| **RSK-01** | Mission authors bypass generic stage system for bespoke UI | High | Med | CI linter strictly rejects non-whitelisted components in mission YAMLs. | Core Lead |
| **RSK-02** | Local execution worker gets stuck in infinite student loops | High | High | Enforce mandatory subprocess timeout and CPU resource limits in WP-140. | Worker Lead |
| **RSK-03** | Heavy ML models (PyTorch/Transformers) slow local UI | Med | High | Stream status/metrics over websocket/IPC; execute in background thread. | Tech Lead |
| **RSK-04** | Accidental leakage of protected answers by Socratic tutor | High | Med | Strict system prompt isolation and automated leakage test suite (WP-645). | AI Lead |

## 2. Issue Register (Active Programme Blockers)

| Issue ID | Summary | Impact | Resolution | Status |
|---|---|---|---|---|
| **ISS-01** | V2 Prototype M01 used custom Python/HTML routes | High | Replaced by generic Mission Definition Language in WP-136. | **RESOLVED** |
| **ISS-02** | Learner state polluted curriculum git tree | High | Decoupled into `~/.learningos/` SQLite in WP-133/WP-134. | **RESOLVED** |

## 3. Change Request Register (Formal Scope Changes)

| CR ID | Title | Scope Impact | Disposition | Gate Effect |
|---|---|---|---|---|
| **CR-01** | Adopt SQLite WAL mode + JSON1 extension for local store | Platform storage | **APPROVED (ADR-002)** | G2 Baseline |
| **CR-02** | Universal 8-type Structured Result Contract | API & Worker | **APPROVED (ADR-004)** | G2 Baseline |
