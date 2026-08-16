# M14 — Discover Structure Without Labels

M14 closes V03 by treating clustering as an engineering investigation rather
than a way to manufacture classes. The working loop is:

`unlabelled observations → feature contract → scaling → candidate k → internal diagnostics → cluster profiles → cautious interpretation`

The lab uses a small synthetic dataset of learning sessions. It deliberately
contains a high-magnitude instrumentation feature so that an unscaled K-means
result can look persuasive while expressing mostly measurement scale.

## Mission flow

1. Audit the dataset and explicitly exclude identifiers from the feature set.
2. Predict which feature will dominate Euclidean distance without scaling.
3. Fit the controlled-failure baseline on raw values.
4. Standardize the selected features and compare candidate values of `k`.
5. Inspect centers, sizes, sample silhouettes, stability, and far-from-center
   observations—not only a two-dimensional plot.
6. Stress the result with an arbitrary `k` and an injected outlier.
7. Write an ADR for the feature, scaling, algorithm, and `k` decisions.
8. State what the clusters support and what they cannot establish without an
   external validation question.

## Run locally

From the repository root, install `requirements/m14.txt`, then run:

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute labs/M14_discover_structure_without_labels.ipynb \
  --output M14_executed.ipynb \
  --output-dir /tmp \
  --ExecutePreprocessor.timeout=300
```

The notebook is CPU-only, deterministic, secret-free, and needs no network
access at runtime. Its source intentionally contains no execution output and no
learner response.

## Completion standard

A strong submission explains why every selected feature participates in the
distance calculation, uses more than one internal diagnostic to choose `k`,
interprets inverse-transformed centers as profiles rather than true classes,
and diagnoses the controlled failures before consulting the discussion prompts.
