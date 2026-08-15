# M13 KNN scale cases

`knn_scale_cases.csv` is a deterministic synthetic classification fixture for studying KNN geometry.

## Schema

| Column | Role | Meaning |
| --- | --- | --- |
| `case_id` | identifier | Stable row identifier `M13-001` through `M13-096`. |
| `practice_hours` | informative feature | Synthetic weekly practice hours, approximately 2–11. |
| `assessment_score` | informative feature | Synthetic score, approximately 45–81. |
| `interface_event_count` | weak feature | Independent random instrumentation count, approximately 1,000–10,000. |
| `learning_route` | target | Balanced `guided` or `independent` class. |

The two informative coordinates are a noisy, rescaled two-moons pattern. This produces curved local neighborhoods that KNN can model without learning an explicit parametric boundary.

`interface_event_count` is generated independently from the class with a separate fixed seed. Its intentionally incompatible numeric scale lets it dominate raw Euclidean distance despite carrying no designed target information.

## Provenance and reproducibility

The dataset is generated locally by `generate_dataset.py` using scikit-learn's packaged `make_moons` generator:

- samples: 96;
- moon noise: 0.20;
- geometry seed: 13;
- weak-feature seed: 1301;
- target balance: 48 `guided`, 48 `independent`.

Run from the repository root:

```bash
python datasets/M13/generate_dataset.py
```

No network, external data, secrets or paid API are used. The data is synthetic and contains no personal information.
