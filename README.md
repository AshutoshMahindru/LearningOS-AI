# LearningOS

LearningOS is a local-first pedagogy engine for software and AI systems work. The core loop is **offline**: catalog, mission player, workbench, evidence, and gates run on your machine. The tutor is optional and talks only to the local API.

> Start with the useful whole. Map it. Interrogate it. Descend only to the narrowest blocker. Decompose it. Rebuild it. Break it. Explain it. Transfer without AI. Prove competence. Return to the system.

Learner state never lives in the Git worktree.

## Launch (learners)

One-click install and launch from a checkout:

```bash
python3 tools/desktop/launch.py
# or: python3 tools/platform/install.py --launch
```

That command bootstraps a managed runtime under `$LEARNINGOS_HOME/runtime/` (outside the Git worktree) and starts the local UI, API, and isolated execution worker. **Jupyter is not required.** `python3 tools/platform/install.py` without `--launch` only prepares the runtime. Desktop helpers live under `tools/desktop/`.

Then open the URL printed at launch (typically `http://127.0.0.1:5173`). Identify a local learner, pick a mission from the catalog, and work through stages in the player.

See [docs/LEARNER.md](docs/LEARNER.md) for data location, backup/restore, and the offline loop.

## Where data lives

All mutable learner data is under `LEARNINGOS_HOME` (default `~/.learningos`): SQLite (`learningos.db`), artifacts, backups, and the worker socket. Override with an **external** directory — never a path inside this repository:

```bash
LEARNINGOS_HOME=/Volumes/private/learningos
```

Backup and restore are on the Settings surface. Restore unpacks into a **clean empty `dest_home`**. The live data home is not a valid restore target because it already contains the database.

## Tutor

`POST /api/v1/tutor/chat` returns **501** unless a provider is configured on the **local API process**. There is no in-browser model client. Do not put provider credentials in the frontend or in Vite environment variables.

## Architecture

1. **Frontend** (`platform/frontend`) — generic shell: catalog, player, workbench, tutor, artifacts, settings.
2. **Backend** (`platform/backend`) — FastAPI on `http://127.0.0.1:8765`, SQLite under `LEARNINGOS_HOME`.
3. **Execution worker** (`platform/worker/daemon.py`) — JSON-RPC daemon on `$LEARNINGOS_WORKER_SOCKET` or `$LEARNINGOS_HOME/run/worker.sock`. Isolated from the API. Sandboxed execution is part of this worker (WP400).

The primary loop does not need the internet.

## Developer inner loop

From a source checkout, after host prerequisites are installed:

```bash
./start.sh
```

`Ctrl+C` stops every service. Diagnostics:

```bash
./start.sh --check
./start.sh --smoke --timeout 60
python3 tools/platform/state_guard.py
```

See `tools/platform/README.md` for preflight and environment overrides.
