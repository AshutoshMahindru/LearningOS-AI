# M15 fixtures

`vector_fixtures.json` is a small, synthetic, deterministic dataset authored for M15. It contains no personal data and requires no download.

- `numeric_features` represents learner-session measurements with named dimensions.
- `geometry` represents a position, displacement and expected endpoint.
- `metric_disagreement` is a controlled case where cosine and Euclidean rankings have different winners.
- `semantic_embeddings` contains small precomputed illustrative embeddings. They are hand-authored teaching vectors, not outputs from a production embedding model, and must not be used as a quality benchmark.

All vectors are finite one-dimensional lists. Semantic vectors share one ordered dimension list and are nonzero so cosine similarity is defined.
