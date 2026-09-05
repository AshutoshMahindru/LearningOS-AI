# LearningOS desktop runtime

One command prepares a managed runtime and starts LearningOS. From a clone of this repository:

```bash
python3 tools/desktop/launch.py
```

That is the learner launch path. It works on a clean checkout: you do not create an environment, install packages, or start extra tools yourself.

## What you need on the machine

- Python 3.11 or newer
- Node.js 20 or newer (the installer uses it to prepare the UI)

If Python 3.11+ is missing, launch stops with a clear preflight error. Second launch is the same command; bootstrap is skipped when the runtime is already current.

Override the data directory with an **external** path (never inside the Git worktree):

```bash
LEARNINGOS_HOME=/tmp/learningos-dev python3 tools/desktop/launch.py
```

Default `LEARNINGOS_HOME` is `~/.learningos`.

## What the installer places outside the repo

All managed files live under `$LEARNINGOS_HOME/runtime/`:

| Path | Role |
|---|---|
| `runtime/python` | Managed Python environment with API dependencies |
| `runtime/frontend` | Managed Node install used by the UI |
| `runtime/bootstrap.json` | Idempotency stamp (requirement and lockfile hashes) |

The Git worktree is not used for the Python environment. The UI runtime is the managed Node install; `platform/frontend/node_modules` is only a symlink to that install so `./start.sh` can find Vite. A checkout that already has a local `node_modules` directory is left unchanged.

After bootstrap, launch sets `LEARNINGOS_PYTHON` (and `PATH`) and execs `./start.sh`, which supervises API + worker + UI.

- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8765/api/v1`

`Ctrl+C` stops every service.

## Other commands

```bash
python3 tools/desktop/launch.py --check
python3 tools/platform/install.py
python3 tools/platform/install.py --launch
python3 tools/desktop/launch.py -- --smoke
```

`--check` only verifies the host (Python 3.11+, Node.js 20+, external data home). `install.py` without `--launch` prepares the runtime and returns. Packaging wrappers in `packaging/` call this same launch path.
