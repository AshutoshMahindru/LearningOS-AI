# Code reading — encode, provenance, score, rank, ties

Read `TeachingEncoder.encode`, `assert_compatible`, `rank_neighbors`,
and `operational_score` in `missions/M28/embedding_core.py`. M28's
code-reading target is the **text-to-neighbor contract**:

1. content tokens (stopwords dropped only for the teaching encoder)
2. token-table lookup, numeric prior, residual axis
3. mean pool, then L2
4. stored width, model, version, metric, normalization
5. query vector of the same provenance
6. cosine (always the cosine formula) or inner product (L2 applied only
   when `normalization=="l2"`)
7. sort by `(-score, id)`
8. refuse mixed model/version/normalization

Before running the code-reading cell, predict:

- the shape and L2 norm of `encode("reset my password")`
- whether extra spaces change the vector after `lexical_tokens`
- whether `rank_neighbors` on a v06.1 query and v06.2 corpus raises
  before it returns a ranking

Do **not** look for query/key/value projections, softmax, or an index
service. Those are M29 and M33. If a failure can be diagnosed from
mismatched metadata, stay at that level.
