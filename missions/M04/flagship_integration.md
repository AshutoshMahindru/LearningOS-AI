# V01 — Structured Data Workbench integration

M04 defines the ingestion and data-quality boundary for the V01 Structured Data
Workbench.

V01 should reuse these architecture rules:

- load source fields losslessly before interpretation;
- retain source-row provenance and `raw_*` values;
- separate exact-record duplication from entity/ID conflict;
- make normalization maps and accepted date formats inspectable;
- convert parse failures into evidence-bearing issue codes;
- distinguish missing from malformed values;
- derive only when the inputs and formula are declared;
- keep blocking review rows visible outside the analysis-ready table;
- treat statistical outliers as investigation prompts, not deletion commands;
- reconcile every raw row to an explicit disposition;
- gate downstream analysis with executable invariants.

The workbench view should expose raw state, canonical state, issue evidence,
decision, uncertainty and readiness side by side. M04 evidence can therefore be
reused as the first V01 vertical slice: ingest a CSV, explain its quality state
and publish a table whose analytical fitness is testable.
