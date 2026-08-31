# LearningOS V3

LearningOS V3 is a schema-driven, local-first pedagogy engine for software engineering. Built on the core principle of **"Transfer without AI"**, this system aims to force the learner into the driver's seat by stripping away instant answers and replacing them with deep, Socratic interrogation and rigorous code execution bounds.

## Core Philosophy

> "Start with the useful whole. Map it. Interrogate it. Descend only to the narrowest blocker. Decompose it. Rebuild it. Break it. Explain it. Transfer without AI. Prove competence. Return to the system."

**Key Principles:**
- **Zero Mission-Specific Logic:** The platform has no hardcoded knowledge of the "missions" it runs. Everything is driven by a unified `MissionSchema`.
- **Local-First Verification:** No internet is required to run the primary loop. Learner state lives under `LEARNINGOS_HOME` (default `~/.learningos`), never inside the Git worktree.
- **Socratic Friction:** The tutor surface is a later-gate capability. In the G3 platform foundation, `POST /api/v1/tutor/chat` returns `501 TUTOR_NOT_AVAILABLE` and does not proxy provider APIs or read `OPENAI_API_KEY`.

## Architecture

LearningOS V3 consists of a three-tier architecture:

1. **Frontend (`platform/frontend`)**
   - React + Vite SPA with a generic application shell (catalog, artifacts, settings).
   - Talks to the local API at `/api/v1` (Vite proxies to `http://127.0.0.1:8765` in development).
   - No mission-specific pages or provider secrets in the browser.

2. **Backend Server (`platform/backend`)**
   - FastAPI + uvicorn on `http://127.0.0.1:8765`.
   - SQLite at `$LEARNINGOS_HOME/learningos.db` (WAL), checksummed artifacts, append-only ledger, backup/restore.
   - Local loopback bearer token in `$LEARNINGOS_HOME/.auth_token`. Provider keys, if present in the process environment, are never returned on `/system/config`.

3. **Execution Worker (`platform/worker/daemon.py`)**
   - Canonical JSON-RPC 2.0 daemon started by `./start.sh`.
   - Socket: `$LEARNINGOS_WORKER_SOCKET`, else `$LEARNINGOS_HOME/run/worker.sock`.
   - Isolated from the API process. G3 does not `exec()` untrusted mission code (WP400 sandbox is later). There is no `platform/backend/worker_daemon.py`.

## Setup & Running

Python 3.11+ and Node.js 20+ are required.

```bash
python3 -m pip install -r platform/backend/requirements.txt
npm ci --prefix platform/frontend
./start.sh
```

Learner data defaults to `~/.learningos`. Override with an **external** directory:

```bash
LEARNINGOS_HOME=/tmp/learningos-dev ./start.sh
```

**Services started:**
- Frontend UI: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8765/api/v1`
- Worker socket: `$LEARNINGOS_HOME/run/worker.sock`

Diagnostics:

```bash
./start.sh --check
./start.sh --smoke --timeout 60
python3 tools/platform/state_guard.py
```

`Ctrl+C` stops every service. See `tools/platform/README.md` for preflight and environment overrides.

Backup restore unpacks into a **clean** `dest_home` (Settings and `POST /api/v1/system/restore`). The live data home is not a valid restore target because it already contains the database.
