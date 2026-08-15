# M07 — Build a Reusable Data-to-Model Pipeline

## Mission

Turn a collection of preprocessing and modeling steps into one reproducible
object that accepts raw rows at both training and inference time.

The reference implementation predicts customer renewal from three numerical
features and three categorical features. Numerical values are imputed and
scaled; categorical values are imputed and one-hot encoded. A
`ColumnTransformer` applies those treatments and a scikit-learn `Pipeline`
binds them to logistic regression.

## Core contract

1. Split raw rows before fitting learned preprocessing state.
2. Fit imputers, encoders, scalers and the estimator only on training rows.
3. Cross-validate the complete pipeline, never a preprocessed matrix produced
   from all folds.
4. Pass raw, schema-compatible rows to both `fit` and `predict`.
5. Serialize the fitted pipeline as one artifact.
6. Reload and verify identical transformed features and predictions.
7. Keep the target, identifiers and post-outcome signals outside the explicit
   model feature allow-list.

## Run the mission

Install `requirements/m07.txt`, open
`labs/M07_reusable_pipeline.ipynb`, and use Restart + Run All from the
repository root. The lab is CPU-only, deterministic, secret-free, and uses
only the local fixture at `datasets/M07/customer_renewals.csv`.

For command-line validation:

```bash
python -m unittest tests.missions.test_m07 -v
python -m pytest tests/missions/test_m07.py -q
python tools/validate_repo.py
```

## Leakage boundary

The dataset is split while still raw. The `Pipeline` is then fitted on the
training partition. During cross-validation, each cloned pipeline learns its
own preprocessing state from that fold's training rows. `MODEL_FEATURES` is an
allow-list, so `customer_id`, `renewed`, and undeclared columns cannot silently
enter the model.

## Controlled failure

The notebook independently one-hot encodes training and inference frames. An
unseen inference category creates a different feature matrix, producing a
deliberate train/inference mismatch. The repair is to send raw inference rows
through the fitted pipeline, whose encoder uses `handle_unknown="ignore"`.

## Review and ADR

`review_brief.md` supplies the formal engineering review packet. Complete the
decision exercise in `adr_prompt.md` before calling the mission complete; a
working score alone does not establish a reproducible boundary.
