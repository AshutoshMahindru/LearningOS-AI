# WP-007: Branch and Pull Request Policy

## Branching Strategy
All work on LearningOS V3 proceeds on dedicated, task-bounded branches. No direct commits to `main` are permitted.

```
main (Protected, stable production baseline)
  └── feature/learningos-v3-architecture (Active WP-000 and WP-100 work)
        ├── wp/WP-111-product-constitution
        ├── wp/WP-134-sqlite-datamodel
        ├── wp/WP-136-mission-schema
        └── ...
```

## Pull Request Lifecycle & Gates
1. **Draft PR Creation**: Created at work package inception with title format `[WP-XXX] Short Title`.
2. **Automated CI Checks Required**:
   - Linting & typechecks (`ruff`, `mypy`, `tsc --noEmit`).
   - Contract validation (`pytest tests/contract/`).
   - Repository consistency check (`python3 tools/validate_repo.py`).
3. **Review & Gate Sign-off**: Requires approval from the designated path owner defined in [WP-006](WP-006_shared_file_ownership_matrix.md).
4. **Merge Requirement**: Squash-and-merge with descriptive commit message linking to the WP ID and ADR references.
