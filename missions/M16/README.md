# M16 — Use Matrices as Transformations

## Mission

Start with a visible shape. Transform it, compare the picture with a prediction, and only then name the matrix operation that produced the result.

The core loop is:

1. inspect the input geometry;
2. predict the output and its shape;
3. apply one matrix;
4. compare prediction with observation;
5. compose matrices and predict which operation happens first;
6. transform a whole batch;
7. diagnose plausible output that came from the wrong convention.

## Working convention

For one vector, the mission uses a column-vector equation:

`y = A @ x`

For a NumPy batch with one sample per row, `X.shape == (batch, features)`. Applying that same transformation to every row is:

`Y = X @ A.T`

The transpose is not decoration. It reconciles the mathematical column-vector convention with row-major batch storage. Shapes are written before multiplication so a merely executable expression is not mistaken for the intended computation.

## Transformations covered

- matrix × vector multiplication as weighted coordinate construction;
- non-uniform scaling;
- rotation in two dimensions;
- horizontal and vertical shearing;
- composition and non-commutative order;
- transpose at the single-vector/batch boundary;
- batched computation and ML layer shapes.

## Diagnostic discipline

The lab contains two independently seeded plausible failures:

- matrices are composed in the wrong order;
- a square row-batch is multiplied by `A` instead of `A.T`.

Both expressions run and return the expected outer shape, so “no exception” is weak evidence. Diagnose with a landmark vector, an explicit shape/convention table, and the first intermediate value that differs from the intended pipeline.

Use:

**intent → shape prediction → landmark prediction → operation → observation → smallest counterexample → repair → invariant**

## Sources

`numpy-quickstart` and `3b1b-linear-algebra` are registered in `data/source_registry.json`. The lab itself has no runtime network dependency.

## V04 connection

M16 contributes the transformation and batched-computation layer of the V04 Mathematical Instrumentation Layer. A dense ML layer also maps a batch through a matrix, then adds a bias and often a nonlinearity. The same shape and orientation discipline prevents silent feature mixing in real model code.

## Completion evidence

Completion requires predictions, visual comparisons, hand calculations, composition traces, both controlled-failure diagnoses, a batch/transpose explanation, an ML-layer shape trace, the no-AI transfer, and the representation ADR.
