# V2 learner-state fixtures

V2 (freeze `35293455ff769855014a49fa56315b22829e74b1`, tag `v2-prototype-freeze`)
has no portable export. Learner state lived in the Git worktree as JSON:

```
<v2-root>/tracking/learner_state.json   # dict: learner_id, mission_status, ...
<v2-root>/tracking/learner_model.json
<v2-root>/tracking/evidence.json        # list
<v2-root>/tracking/sessions.json        # list
<v2-root>/tracking/competencies.json    # dict
<v2-root>/tracking/artifacts/           # optional files
```

The freeze commit's `tracking/` is an empty default learner. These fixtures
mirror that layout with populated data so the V3 importer can be proven
offline without checking out the freeze branch.

| Path | What it represents |
|---|---|
| `populated/` | One V2 learner with sessions, mission_status, evidence, and an artifact file |
| `multi/` | Two V2 learner bundles (`alice/`, `bob/`) |
| `garbage/learner_state.json` | Truncated JSON (must fail dry-run) |

Point `tools/platform/v2_migrate.py --source` at `populated/`, `multi/`, or a
`git archive` of the freeze `tracking/` tree. Import writes only to
`LEARNINGOS_HOME` (never the Git worktree).
