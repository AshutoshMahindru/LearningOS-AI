# M17 model-output fixture

`model_predictions.csv` is a deterministic synthetic dataset for reasoning about probability outputs and calibration. It is not a benchmark and contains no personal or production data.

## Schema

- `case_id`: unique synthetic case identifier;
- `cohort`: stable operation or deliberate deployment shift;
- `model_probability`: model-reported probability for the defined event;
- `outcome`: observed event indicator (`1` occurred, `0` did not).

## Designed behavior

The 0.10, 0.40, and 0.80 groups have observed frequencies equal to their reported probabilities. The 0.90 deployment-shift group has a lower observed frequency. This makes calibration visible from counts without training a model or requiring an external package.

Individual rows do not prove or disprove calibration. Calibration is a repeated-outcome, group-level relationship. The fixture is deliberately small enough to inspect by hand and should not be used to claim production quality.

## Provenance and runtime

The rows are authored synthetic fixtures for M17. They are static, CPU-only, deterministic, secret-free, and network-free.
