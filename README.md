# AI Learning OS

**A top-down, mission-driven ML/AI apprenticeship system.**

This repository merges an adaptive ML/AI curriculum engine with the apprenticeship discipline of a technical-builder programme. The aim is not course completion; it is to produce an AI Systems Architect/Builder who can **build, inspect, debug, challenge, evaluate and direct** AI systems — including work produced by AI agents.

## Core invariant

> **Start with the useful whole. Map it. Interrogate it. Descend only to the narrowest blocker. Decompose it. Rebuild it. Break it. Explain it. Transfer without AI. Prove competence. Return to the system.**

## What is inside

- **42 missions / 9 phases** — the ML/AI route.
- **76 observable competencies** — L0 to L5 evidence model.
- **253 knowledge nodes** — prerequisite/enables graph.
- **Canonical content map** — primary/official sources, exact assignments, labs and side quests.
- **Apprenticeship control plane** — no-AI gates, code reading, controlled failure, Chaos Days, engineering review, Git/ADR discipline and progressive autonomy.
- **One flagship system / 12 versions** — an Operations Intelligence System that evolves from data workbench to evaluated agentic AI system.
- **Tutor orchestration** — navigator, Socratic tutor, debugger, examiner, Feynman reviewer, zoom controller, chaos engineer, code-reading coach and principal-engineer reviewer.
- **Evidence infrastructure** — ledgers for experiments, side quests, no-AI work, chaos, review, ADRs and macro maturity.

## Canonical runtime

```text
MISSION + FLAGSHIP VERSION
        ↓
WHOLE FIRST → MAP → INTERROGATE
        ↓
MINIMUM CANONICAL CONTENT
        ↓
MANIPULATE / EXPERIMENT
        ↓
ZOOM IN ONLY IF BLOCKED
        ↓
DECOMPOSE / CODE READ
        ↓
REBUILD → BREAK → EXPLAIN
        ↓
NO-AI TRANSFER
        ↓
GATE
        ↓
GIT EVIDENCE + ADR / REVIEW
        ↓
ADVANCE / TARGETED REPAIR
```

## Start here

1. Read [`docs/SYSTEM_BLUEPRINT.md`](docs/SYSTEM_BLUEPRINT.md).
2. Read [`docs/LEARNING_RUNTIME.md`](docs/LEARNING_RUNTIME.md).
3. Open [`dashboard.html`](dashboard.html) locally.
4. Start **M01** in [`docs/MISSION_PLAYBOOK.md`](docs/MISSION_PLAYBOOK.md).
5. Use the exact content route in [`docs/CONTENT_MAP_42_MISSIONS.md`](docs/CONTENT_MAP_42_MISSIONS.md).
6. Record learner evidence under `tracking/`; repository implementation status is tracked separately in `data/lab_status.json` and mission-local status files.
7. Run `python tools/validate_repo.py` before committing structural changes.

## Flagship release spine

| Version | Missions | Release |
|---|---|---|
| V01 | M03-M07 | Structured Data Workbench |
| V02 | M08-M10 | Predictive Decision System |
| V03 | M11-M14 | Model Comparison & Diagnostics |
| V04 | M15-M20 | Mathematical Instrumentation Layer |
| V05 | M21-M26 | Neural Learning Component |
| V06 | M27-M30 | Language Representation Layer |
| V07 | M31-M32 | LLM Inference & Adaptation Layer |
| V08 | M33 | Semantic Search Service |
| V09 | M34-M36 | Grounded Knowledge System |
| V10 | M37-M39 | Tool-Using Agent System |
| V11 | M40-M41 | Evaluated & Observable AI System |
| V12 | M42 | Integrated AI Systems Capstone |

## Apprenticeship defaults

- 30–60 min no-AI competency block every mission.
- 10–15 min unfamiliar code/artifact reading in coding sessions.
- Controlled failure every mission.
- Hidden-fault Chaos Day every 2 missions.
- Engineering review every 4 missions or phase end.
- ADRs for consequential decisions from M07 onward.
- Git engineering record for substantive changes.
- AI implementation autonomy expands only with independent evidence.
- Do not study a concept for >2 hours without using it.

## Implementation status

The architecture, 42-mission route, canonical 253-node graph, content routing and apprenticeship overlays are instantiated.

**M01-M39 are implemented and repository-executable.** Their mission packages, tests and 39 source notebooks have passed combined minimal-runtime and full-dependency validation, including clean notebook execution. **M40-M42 are not executable** and must not be advertised as executable until they pass the same implementation and validation gates.

Learner progress is intentionally separate from repository implementation progress: merging a mission does not mark a learner as having completed it.

## Repository map

```text
ai-learning-os/
├── README.md
├── dashboard.html
├── docs/
├── data/
├── labs/
├── datasets/
├── missions/
├── prompts/
├── templates/
├── tracking/
├── schemas/
├── tools/
├── requirements/
└── .github/
```

## Validation

The default CI keeps a dependency-light runtime job and a full M01-M39 mission-validation job. The full job installs the union of mission requirements, runs repository and mission tests, validates source notebook invariants, and executes all repository-executable notebooks in fresh kernels.

## Source policy

Prefer primary/official sources. The AI tutor supplements rather than replaces canonical material. Fast-moving implementation resources must be rechecked at the cadence recorded in `data/resource_library.csv`.

## Status

**M01-M41 integrated and verified; M42 pending implementation.**
