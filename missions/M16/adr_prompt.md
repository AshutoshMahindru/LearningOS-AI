# ADR prompt — Vector and batch representation contract

Write an Architecture Decision Record using `templates/ADR.md` for a small transformation library that serves both geometry code and an ML preprocessing pipeline.

## Decision to make

Choose and document:

- the public representation of a single vector;
- whether batches store samples in rows or columns;
- the declared shape of a transformation from `in_features` to `out_features`;
- the formula used to apply it to one vector and to a batch;
- the convention for composing “first A, then B”;
- where transpose operations are allowed and why;
- validation behavior for incompatible dimensions;
- numeric invariants that detect orientation or order mistakes.

## Options that must be compared

1. column-vector mathematics with row-major batch storage;
2. column-vector mathematics with column-major batch storage;
3. row-vector convention throughout.

## Evidence required before deciding

- shape tables for one non-square transform and one batch;
- a non-symmetric square transform showing how wrong orientation can remain plausible;
- two non-commuting transforms showing order sensitivity;
- one dense-layer example mapped into each viable convention;
- API ergonomics, interoperability and testability trade-offs.

## Acceptance criteria

The ADR must choose one option, reject the alternatives with concrete consequences, specify executable invariants, state migration costs, and identify at least one condition that would justify revisiting the decision. Do not prefill learner evidence or copy the mission convention without evaluating it.
