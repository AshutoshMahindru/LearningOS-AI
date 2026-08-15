# M05 → V01 Flagship Integration

M05 supplies the array-computation layer for the **V01 Structured Data Workbench**. M03 establishes Python fluency and M04 prepares structured data; M05 turns rectangular numeric values into explicit, testable array operations that later V01 pipeline stages can reuse.

## What M05 contributes to V01

- an `(orders, products)` representation for structured numeric features;
- explicit shape and dtype inspection at computational boundaries;
- vectorized arithmetic over many records without Python-level element loops;
- per-row and per-column aggregation with named axis semantics;
- broadcasting of shared product parameters and per-record adjustments;
- correctness checks that precede performance claims;
- failure diagnosis for incompatible shapes and semantically wrong axes.

## Hand-off to M06 and M07

M06 can visualize row- and column-level aggregates whose shapes are already understood. M07 can encapsulate the validated array transformations in a reproducible data-to-model pipeline, including boundary checks for incoming shapes and dtypes.

## V01 integration check

Before the V01 review, the learner should be able to take a cleaned rectangular table, state which rows and columns become each array axis, vectorize a useful transformation, verify it against a small loop baseline, and explain any timing result without treating speed as proof of correctness.
