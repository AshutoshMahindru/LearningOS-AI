# Learner guide

LearningOS is a local app. The catalog, mission player, workbench, evidence, and gates run on your machine. **Jupyter is not required.** The core loop is **offline**.

## Launch

From a source checkout, the intended one-click path is:

```bash
python3 tools/platform/install.py
```

Desktop packaging and shortcuts live under `tools/desktop/`. The installer is meant to install host prerequisites if needed, place learner data outside the Git worktree, and start:

- the UI (typically `http://127.0.0.1:5173`)
- the local API (`http://127.0.0.1:8765/api/v1`)
- the isolated execution worker (`platform/worker/daemon.py`)

You do not need to create a virtualenv, run `pip`/`npm` by hand, or start a notebook server to learn.

Developers iterating on the source tree may still use `./start.sh` (see `tools/platform/README.md`). That is the inner loop, not the learner path.

## First session

1. Open the UI URL the installer prints.
2. Identify a local learner (username). This identity stays on the local API; it is not a cloud account.
3. The **catalog** lists missions the local API has loaded.
4. Start a session. The **player** renders generic stages from the mission specification (orientation, experiment, transfer, gate, and so on).
5. Use **Settings** for health, non-secret config, curriculum package load, and backup/restore.
6. The **tutor** surface talks only to the local API. If no provider is configured on the server, chat returns HTTP 501. Provider credentials never enter the browser.

Keyboard: every primary control is a native button, link, or labelled field. Use `Tab` / `Shift+Tab`, `Enter` / `Space` to activate, and `Escape` to close Diagnostics. The player stage list also accepts `ArrowUp` / `ArrowDown` / `Home` / `End`.

## Where data lives

Learner state is **never** stored in the Git worktree.

| Location | Role |
| --- | --- |
| `LEARNINGOS_HOME` (default `~/.learningos`) | Mutable home: database, artifacts, backups, worker socket |
| `$LEARNINGOS_HOME/learningos.db` | SQLite (WAL) |
| `$LEARNINGOS_HOME/run/worker.sock` | Worker socket unless `LEARNINGOS_WORKER_SOCKET` is set |
| `$LEARNINGOS_HOME/.auth_token` | Loopback bearer token for the local API |

Override the home with an **external** directory:

```bash
LEARNINGOS_HOME=/Volumes/private/learningos
```

A path inside the repository is rejected. `python3 tools/platform/state_guard.py` confirms no mutable V3 state leaked into the checkout.

## Backup and restore

On **Settings → Backup and restore**:

1. **Create backup** writes a snapshot under the data home.
2. **Restore** needs a backup id or path **and** a destination home (`dest_home`).

Restore unpacks into a **clean empty `dest_home`**. The live `LEARNINGOS_HOME` cannot be the target: it already contains the database. After a restore, point `LEARNINGOS_HOME` at that destination (or copy it into place while the app is stopped) to use the restored state.

The same dest_home rule applies to `POST /api/v1/system/restore`.

## Offline core loop

You do not need the internet to:

- browse the catalog
- run a mission in the player
- execute sandboxed jobs through the local worker
- store artifacts
- evaluate gates
- create and restore backups

A tutor provider, if you configure one later, is optional and server-side only. The product does not call a model from the browser.

## Accessibility

The shell exposes skip-to-content, header / navigation / main landmarks, labelled forms, and visible focus. Contrast uses a dark theme with light text and a high-contrast focus ring. Reduced-motion preferences disable non-essential animation.
