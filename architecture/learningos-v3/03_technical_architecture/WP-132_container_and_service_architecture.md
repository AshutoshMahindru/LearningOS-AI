# WP-132: Container and Service Architecture

## 1. Subsystem Decomposition

```
LearningOS V3
├── Frontend Web Client (React 18 + TypeScript + Vite + Tailwind/Vanilla CSS)
│   ├── App Shell & Router (8 Surfaces)
│   ├── Generic Stage Component Registry (11 Canonical Stages)
│   ├── Workbench UI (Monaco Editor, Table Viewer, Chart.js/Plotly, Trace Canvas)
│   └── Typed API Client
├── API & Orchestration Backend (FastAPI / Pydantic v2 / Python 3.11+)
│   ├── Mission Loader & Schema Validator (MDL v1)
│   ├── Session & Stage State Machine
│   ├── Gate Evaluator & Evidence Engine
│   ├── Socratic Tutor Router (Provider-Agnostic)
│   └── Local SQLite Data Access Layer (WAL mode)
└── Isolated Execution Worker (Subprocess Daemon)
    ├── IPC Server (Unix Domain Socket / Local TCP)
    ├── Task Dispatcher & Resource Watchdog (Timeout / Memory caps)
    ├── Python Function & Notebook Adapter
    └── Structured Result Serializer (Table/Chart/Trace/StateDiff/Artifact)
```

## 2. Process Lifecycle & IPC
- The API backend starts the Execution Worker as a child subprocess.
- Health heartbeat runs every 2 seconds. If the worker encounters a segfault or out-of-memory error from student code, the API backend automatically restarts the worker and reports a structured error to the UI without dropping the HTTP session.
