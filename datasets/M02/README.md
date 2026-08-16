# M02 dataset

`wine.csv` is a local, header-normalized copy of the 178-row Wine recognition dataset packaged by scikit-learn and originally contributed from the UCI Machine Learning Repository. It contains 13 continuous chemical measurements and the integer target classes `0`, `1`, and `2`.

The committed fixture makes notebook execution independent of runtime network access and makes the raw-data boundary inspectable. The target is educational benchmark data, not a production outcome or a causal label. The mission tests verify row count, schema, finite numeric features, and class coverage.

Columns: `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280_od315_of_diluted_wines`, `proline`, `target`.
