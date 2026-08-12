# Canonical Knowledge Graph Runtime

The Learning OS now uses two separate graphs.

## Mission graph

Answers:

> What should happen next?

It manages M01-M42 sequencing, mission prerequisites and advancement.

## Knowledge graph

Answers:

> What concepts enable this?

It contains the canonical 253-node concept dependency graph.

Runtime flow:

```text
Canonical YAML / CSV
        |
        v
KnowledgeGraph adapter
        |
        +--> concept lookup
        +--> prerequisite traversal
        +--> enables traversal
        +--> mission concept discovery
        |
        v
Mission Context
        |
        v
Tutor / Router / Zoom Controller
```

## Design rules

- The canonical graph source is not rewritten by runtime code.
- Relationships authored as concept names are resolved to stable node IDs at load time.
- Mission sequencing and concept prerequisites remain separate systems.
- Bootstrap graph files are historical references only; runtime uses the canonical graph.
