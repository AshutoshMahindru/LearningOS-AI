# ADR-001: Schema-Driven Generic Architecture for Missions M01–M42

## Status
ACCEPTED (Controlling baseline for LearningOS V3)

## Context
In LearningOS V2, mission M01 was implemented as a bespoke HTML and Python module (`web/m01.html`, `learning_os/m01_experience.py`). This cannot scale across 42 missions without creating 42 distinct frontend templates, 42 sets of bespoke API routes, and massive duplication.

## Decision
1. **Generic Schema Definition**: All missions (M01–M42) will be authored exclusively as declarative schema files (YAML/JSON) conforming to the Mission Definition Language (MDL v1).
2. **Zero Mission-Specific Frontend/API Code**: The frontend will contain zero mission-specific JSX/HTML files. All UI renders dynamically from a reusable stage-component registry.
3. **Generic API Endpoints**: All platform interactions use generic endpoints (`/api/v1/sessions`, `/api/v1/stages/{id}/actions`, `/api/v1/gates/evaluate`).

## Consequences
- Positive: New missions can be authored and added without touching platform code or frontend repositories.
- Positive: Upgrades to stage components immediately improve all missions.
- Negative: Custom, non-standard interactions must be generalized into reusable stage primitives.
