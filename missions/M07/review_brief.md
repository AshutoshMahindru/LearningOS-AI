# M07 Formal Engineering Review Brief

## Review claim

The reference implementation provides one reproducible boundary from raw,
schema-compatible rows to renewal predictions. It supports mixed numerical and
categorical data, keeps learned transformations inside fitting boundaries, and
can be serialized and reloaded without changing transformations or predictions.

## Architecture under review

```text
raw rows
  ├─ explicit numerical allow-list → median imputer → scaler ─┐
  └─ explicit categorical allow-list → mode imputer → OHE ────┤
                                                              ↓
                                                   logistic regression
                                                              ↓
                                                        prediction
```

The train/test split occurs before the pipeline is fitted. Cross-validation
receives an unfitted complete pipeline, so every fold learns imputations,
scales, categories and coefficients only from that fold's training rows.

## Interface and invariants

- `build_pipeline` returns an unfitted `Pipeline` with named `preprocess` and
  `model` steps.
- Both `fit` and `predict` receive raw frames with `MODEL_FEATURES`.
- `split_features_target` drops identifiers, target and undeclared columns by
  explicit selection.
- Unknown categories do not change transformed width.
- `transform` and `predict` do not update fitted preprocessing or model state.
- Persisted and reloaded artifacts produce identical predictions and
  numerically identical transformed matrices for the same raw rows.

## Alternatives considered

1. Manual notebook transformations: transparent for first inspection, but
   duplicates ordering and learned state at inference.
2. A pre-fitted transformer plus separate estimator: reusable, but easier to
   cross-validate incorrectly and easier to deploy as mismatched versions.
3. One pipeline artifact (selected): gives cloning, CV, persistence and raw-row
   inference one shared boundary at modest abstraction cost.

## Failure modes and controls

| Failure mode | Control | Verification |
| --- | --- | --- |
| Preprocessing fitted before split | Split raw rows first | Test fitted attributes are absent before pipeline fit |
| Preprocessing fitted outside CV | Pass full pipeline to `cross_validate` | Original estimator remains unfitted because CV clones it |
| Target or post-outcome leakage | Explicit model feature allow-list | Injected outcome-derived column is excluded |
| Unseen categorical level | `handle_unknown="ignore"` | Unseen-category raw row retains transformed width and predicts |
| Training/inference code drift | One pipeline public interface | Controlled independent-encoding failure plus repaired prediction |
| Serialization drift | Persist one pipeline | Transforms and predictions compared before/after reload |

## Validation requested from reviewer

1. Inspect the column allow-list against when each feature becomes available.
2. Confirm no learned transformer is fitted before split or outside CV.
3. Challenge the unknown-category policy and its monitoring implications.
4. Review the persistence warning: joblib artifacts must only be loaded from a
   trusted source and should be accompanied by dependency/version metadata.
5. Re-run mission unittest, mission pytest, repository unittest, repository
   validator, notebook Restart + Run All and `git diff --check`.

## Residual uncertainty

The teaching fixture is small and synthetic; its accuracy is not evidence of
business utility or production generalization. `handle_unknown="ignore"`
preserves interface parity but can conceal category drift without monitoring.
Joblib does not promise portability across arbitrary scikit-learn versions and
must not load untrusted artifacts. Production work needs schema enforcement,
drift monitoring, dependency locking and representative data review.

## Review decision requested

Approve the boundary only if the reviewer can explain where every learned
state is fitted, reproduce the controlled failure, and accept or amend the ADR
choice in `adr_prompt.md`. Record concerns as actionable failure cases rather
than score-only objections.
