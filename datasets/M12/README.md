# M12 deterministic ensemble fixture

`ensemble_fixture.csv` is a synthetic binary-classification fixture generated with `generate_fixture.py` using seed `12012`. It contains 480 rows, a non-linear two-moons signal (`x1`, `x2`), two derived weak signal features (`linear_mix`, `periodic`), two independent nuisance features (`noise1`, `noise2`), and the binary `target`.

The CSV is committed so notebook execution has no network dependency. Regeneration requires only NumPy and scikit-learn from `requirements/m12.txt`:

```bash
python datasets/M12/generate_fixture.py
```

The data is for deterministic pedagogy, not for production model selection. The notebook uses a seeded stratified split and injects label corruption only into an in-memory copy of the training labels during the controlled failure.
