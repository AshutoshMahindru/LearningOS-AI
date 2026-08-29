# WP-131: System Context & Boundary Architecture

## 1. System Context Diagram (C4 Level 1)

```
                       ┌───────────────────────────────────────────────┐
                       │               THE AI LEARNER                  │
                       └───────┬───────────────────────────────▲───────┘
                               │                               │
                       Interact│                               │ View Stages,
                       & Code  │                               │ Workbench,
                               ▼                               │ Evidence
                 ┌───────────────────────────────────────────────────────────┐
                 │                  LEARNINGOS V3 PLATFORM                   │
                 │                                                           │
                 │  ┌───────────────────────┐     ┌───────────────────────┐  │
                 │  │  React + TS Frontend  ├────►│    Typed Python API   │  │
                 │  └───────────────────────┘     └───────────┬───────────┘  │
                 │                                            │              │
                 │                                            ▼              │
                 │                                ┌───────────────────────┐  │
                 │                                │   Execution Worker    │  │
                 │                                └───────────────────────┘  │
                 └───────────────┬────────────────────────────┬──────────────┘
                                 │ Reads                      │ Reads & Writes
                                 ▼                            ▼
                 ┌───────────────────────────┐  ┌────────────────────────────┐
                 │    CURRICULUM PACKAGE     │  │     LOCAL LEARNER STORE    │
                 │  (Read-Only Git Repo)     │  │   (~/.learningos/ Store)   │
                 │  - Missions (M01-M42)     │  │  - learningos.db (SQLite)  │
                 │  - Datasets & Fixtures    │  │  - Artifacts & Weights     │
                 │  - Rubrics & Tests        │  │  - Session Logs & Evidence │
                 └───────────────────────────┘  └────────────────────────────┘
```

## 2. Boundary Isolation Invariants
1. **Curriculum Package**: Completely read-only at runtime. No mutations, session files, or learner tracking written inside this directory.
2. **Local Learner Store**: Completely outside the curriculum git tree (`~/.learningos/`). Survives curriculum branch switches and git clean operations.
3. **External Model Providers**: Optional and pluggable (OpenAI, Anthropic, Gemini, local Ollama). System works 100% offline without them.
