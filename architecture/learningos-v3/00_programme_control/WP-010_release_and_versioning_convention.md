# WP-010: Release & Versioning Convention

## Multi-Domain Semantic Versioning
LearningOS V3 operates across four decoupled domains, each with independent semantic versioning:

1. **Application Platform Version (`vX.Y.Z`)**:
   - Manages the React UI shell, FastAPI backend runtime, and execution worker.
   - Example: `v3.0.0-alpha.1`
2. **Curriculum Package Version (`vX.Y.Z`)**:
   - Manages the 42 missions, datasets, rubrics, and knowledge nodes.
   - Example: `curriculum-v3.0.0`
3. **Mission Definition Language (MDL) Schema Version (`schema/vX`)**:
   - Controls backwards-compatibility of mission YAML definitions.
   - Example: `learningos.mission.v1.json`
4. **Database Migration Version (`migrations/XXXX`)**:
   - Sequential integer migrations for SQLite schema evolution.
   - Example: `0001_initial_v3_schema.sql`

## Compatibility Rule
The application platform must declare supported MDL schema versions in its engine specification. A V3.x platform runtime must execute any valid `v1` mission schema without modification.
