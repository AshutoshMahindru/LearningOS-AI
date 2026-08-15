# M16 Formal Engineering Review Brief

## Review target

Mission-local contract, deterministic dataset, executable notebook and substantive invariant tests for “Use Matrices as Transformations.”

## Architecture and conventions

- Single mathematical vector: column convention `y = A @ x`.
- NumPy dataset and batch: one sample per row, shaped `(batch, features)`.
- Equivalent batch transform: `Y = X @ A.T`.
- Composition: in `B @ A`, `A` acts first.
- ML bridge: `X @ W + b`, with `W = A.T` when mapped to the single-vector convention.

These choices are explicit because square matrices otherwise allow semantically wrong orientation code to execute and return plausible output.

## Meaningful artifact

`labs/M16_matrix_transformations.ipynb` begins with geometry, then derives matrix operations from observed coordinate movement. It loads a versioned local polygon fixture, uses no secrets or network, and runs on CPU. Stable cell IDs and empty source outputs support reproducible review.

## Failure model

Two isolated controlled cases cover distinct risks:

1. reverse composition order while retaining valid shapes;
2. omit the transpose at a row-batch boundary while retaining valid shapes.

Both diagnoses require a non-symmetric or non-commuting counterexample, first-divergence trace, repair and post-repair invariant. A separate caught dimension mismatch distinguishes loud shape failure from silent semantic failure.

## Verification strategy

- static artifact and contract assertions;
- stable unique notebook cell IDs and no prefilled output;
- compile and sequentially execute every code cell;
- numeric dataset invariants and asymmetric geometry;
- sample-wise/batch equivalence assertions;
- repository validator and repository unittest discovery;
- notebook Restart + Run All with a bounded timeout.

## Review questions

1. Are vector and batch conventions visible before the first ambiguous multiplication?
2. Do the failure examples isolate one root cause each?
3. Are landmark and sample-wise invariants strong enough to reject plausible wrong output?
4. Does the ML-layer bridge clarify the weight orientation without implying all layers are two-dimensional geometry?
5. Does the ADR prompt force a real representation decision rather than asking for a summary?

## Residual risks

- Two-dimensional visuals can make higher-dimensional transformations seem easier to inspect than they are.
- Rotation sign depends on coordinate-system orientation; the notebook explicitly uses Cartesian axes.
- Learners may memorize `.T`; assessment therefore requires derivation from shapes and semantic conventions.
