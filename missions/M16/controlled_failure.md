# Controlled failures

The notebook contains two separate failures. Each has exactly one seeded root cause and each returns plausible numeric output. Diagnose them independently.

## Case A — wrong composition order

Intent: shear a point, then rotate the sheared result.

Seeded implementation: composes the rotation and shear in the reverse application order.

Required diagnostic sequence:

1. write the intended operations in words;
2. predict the intermediate and final position of `[1, 0]`;
3. run the smallest one-vector counterexample;
4. identify the first intermediate value that diverges;
5. repair the matrix order;
6. assert equivalence between the composite and the explicit two-step calculation.

## Case B — wrong batch orientation

Intent: apply the column-vector transformation `y = A @ x` to every row of `X`.

Seeded implementation: computes `X @ A`. Because `A` is square, it runs and preserves `(n, 2)`, but it corresponds to `A.T @ x` for each row when interpreted under the mission convention.

Required diagnostic sequence:

1. write the shape and semantic role of every axis;
2. use a non-symmetric matrix and one landmark row;
3. compare batch output with `A @ landmark`;
4. identify the semantic mismatch despite matching outer shapes;
5. repair the operation as `X @ A.T` with the justified transpose;
6. assert sample-wise equivalence for the entire batch.

For both cases, “the plot looks reasonable” and “the code ran” are symptoms, not verification. Preserve the trace, hypothesis, counterexample, repair and invariant as evidence.
