# LearningOS V3 developer runtime

From the repository root, the single development command is:

```bash
./start.sh
```

It preflights the host, starts the execution worker, API on
`http://127.0.0.1:8765`, and frontend on `http://127.0.0.1:5173`, then owns their
complete lifecycle. `Ctrl+C` sends a graceful stop to every service and force-stops
only a service that exceeds the shutdown grace period.

Install prerequisites once if preflight reports them missing:

```bash
python3 -m pip install -r platform/backend/requirements.txt
npm ci --prefix platform/frontend
```

Useful diagnostics:

```bash
./start.sh --check
./start.sh --smoke --timeout 60
python3 tools/platform/state_guard.py
```

Mutable learner data defaults to `~/.learningos`. Override it with an absolute
external location such as `LEARNINGOS_HOME=/Volumes/private/learningos`; a path
inside the repository is rejected before any service starts. Advanced local
integrations may override individual commands with
`LEARNINGOS_WORKER_COMMAND`, `LEARNINGOS_BACKEND_COMMAND`, and
`LEARNINGOS_FRONTEND_COMMAND`.
