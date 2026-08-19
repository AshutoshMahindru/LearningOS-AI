# Dashboard Runtime

The dashboard is a read-only human interface over the same closed-loop runtime used by the CLI. It does not maintain a second progress database and it does not mutate learning state through browser requests.

## Snapshot

```bash
learning-os dashboard
```

returns JSON containing:

- current mission and next action,
- passed mission count,
- learner-model competencies and confidence,
- autonomy level,
- retention events due,
- open side quests,
- recent evidence,
- current mission context and routed resources,
- explicit lab implementation status.

## Local dashboard server

```bash
learning-os dashboard --serve
```

serves the UI at `http://127.0.0.1:8765` by default. Override with `--host` and `--port`.

The server exposes read-only endpoints:

- `/api/dashboard` — complete human-interface projection,
- `/api/step` — current routing decision,
- `/api/context?mission=M33` — assembled mission context,
- `/healthz` — process health.

State-changing operations remain in the CLI/runtime so a browser refresh cannot accidentally change evidence, gates, retention or autonomy.

## Design constraint

The dashboard must reflect repository lab availability rather than planned curriculum state. M01-M21 are now implemented and validated as repository-executable labs. M22-M42 remain non-executable until their mission packages are implemented and pass the same repository and notebook validation gates.
