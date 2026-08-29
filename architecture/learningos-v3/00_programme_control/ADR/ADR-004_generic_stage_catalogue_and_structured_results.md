# ADR-004: Reusable Stage Component Catalogue & Structured Result Contract

## Status
ACCEPTED (Controlling baseline for LearningOS V3)

## Context
Mission execution in V2 relied on unstructured console outputs and ad-hoc visual widgets. This made automated evaluation fragile and restricted UI fidelity.

## Decision
1. **Controlled Stage Catalogue**: Standardize 11 canonical stage types (`orientation`, `trace_map`, `interrogate`, `experiment`, `code_reading`, `rebuild_debug`, `controlled_failure`, `transfer_assessment`, `competency_gate`, `reflection_adr`, `flagship_integration`).
2. **Structured Result Contract**: Worker execution produces typed JSON blocks (`table`, `chart`, `trace`, `state_diff`, `diagram`, `markdown`, `metric`, `artifact`) conforming to `WP-137_structured_result_schema.json`. Raw stdout/stderr are routed solely to a learner-toggleable diagnostics drawer.

## Consequences
- Positive: Safe, rich, interactive UI rendering across all data structures and ML outputs.
- Positive: Standardized rubric evaluation on structured state rather than string parsing logs.
