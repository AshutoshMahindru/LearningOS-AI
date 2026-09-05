# Runtime quickstart

This is the product a learner launches: the local V3 shell. Jupyter is not required. The core loop is offline.

## Launch

```bash
python3 tools/desktop/launch.py
```

Equivalent launch: `python3 tools/platform/install.py --launch`. Runtime-only bootstrap: `python3 tools/platform/install.py`. Desktop helpers: `tools/desktop/`. Open the URL printed at launch (typically `http://127.0.0.1:5173`).

Developers on a source checkout may use `./start.sh` instead. See `tools/platform/README.md`.

## Data

Learner state lives under `LEARNINGOS_HOME` (default `~/.learningos`), never in the Git worktree.

Restore requires a clean empty `dest_home`. The live data home is not a valid restore target.

## Tutor

Default chat response is HTTP 501 until a provider is configured on the local API. There is no in-browser model client.

Full learner notes: [LEARNER.md](LEARNER.md).
