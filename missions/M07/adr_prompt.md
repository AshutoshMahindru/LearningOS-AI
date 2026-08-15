# M07 Architecture Decision Record Prompt

## Consequential decision

Decide whether numerical preprocessing, categorical preprocessing and the
estimator should be fitted, cross-validated, versioned and deployed as one
pipeline artifact, or managed as separately fitted components.

The reference implementation selects **one pipeline artifact**. Your ADR must
independently accept, amend or reject that choice; do not treat the existing
code as proof that the decision is correct.

## Context to evaluate

- Training and inference must accept the same raw feature schema.
- Imputation, scaling and category vocabulary learn state from data.
- Validation estimates are invalid if a transformer sees held-out fold data.
- Inference can contain categories absent from training.
- The artifact must be reloadable without rebuilding transformation state.
- The system is currently CPU-only and small, but V01 will reuse the boundary.

## Required alternatives

Compare at least:

1. duplicated manual preprocessing in training and inference code;
2. one separately persisted transformer plus one estimator;
3. one scikit-learn `Pipeline` containing `ColumnTransformer` and estimator;
4. another viable equivalent if you propose one.

## ADR deliverable

Use `templates/ADR.md` and include:

- status and date;
- decision owners and reviewers;
- context and decision drivers;
- options with advantages, risks and rejected reasons;
- chosen boundary and public input/output contract;
- fit, transform, CV and inference lifecycle;
- feature availability and leakage assumptions;
- unknown-category behavior;
- serialization, trusted-loading and version-compatibility policy;
- positive and negative consequences;
- tests, monitoring and rollback triggers;
- evidence from the controlled failure and reload experiment;
- conditions that would cause the decision to be revisited.

## Review questions

1. Which pieces of state are learned, and from exactly which rows?
2. Can any caller bypass the intended transformation at inference?
3. What failure becomes more likely if artifacts are versioned separately?
4. Does ignoring unknown categories preserve correctness or only shape?
5. What evidence distinguishes reproducibility from a one-time successful run?

Do not prefill learner conclusions. The ADR is assessed as engineering
reasoning supported by observed evidence, not agreement with the reference.
