# M04 — Turn a Messy CSV into Usable Data

## Mission

Clean a realistic customer-orders CSV without erasing the evidence needed to
explain what happened. The learner moves through this trace:

**raw → inspect → predict defects → declare expectations → analyze duplicates →
normalize → parse → decide missingness → validate → investigate outliers → run a
reproducible pipeline → compare → assert invariants**

## Data-quality discipline

Every finding must be classified explicitly:

- **Observed defect:** a fact visible in the raw data or a failed rule.
- **Hypothesis:** a possible explanation to test, not a fact.
- **Decision:** the declared treatment and its scope.
- **Evidence:** the raw value, rule, comparison or domain note supporting a decision.
- **Lost information / uncertainty:** what cannot be recovered or still needs an owner.

The pipeline keeps `raw_*` columns and source-row provenance. It removes only
exact duplicate rows into an audit log. Conflicting IDs, malformed values and
outliers remain visible for review; they are never silently discarded.

## Runtime

From the repository root, install `requirements/m04.txt`, then run
`labs/M04_messy_csv.ipynb`. The lab is CPU-only, deterministic, secret-free,
paid-API-free and network-free at runtime.

## Completion evidence

Completion requires prediction notes, a declared data contract, duplicate and
parsing evidence, missingness decisions, outlier judgments, controlled-failure
diagnosis, raw-versus-clean reconciliation, invariant assertions and a fresh
no-AI transfer task.

## V01 connection

M04 supplies the auditable ingestion and quality layer for the V01 Structured
Data Workbench. V01 must show why each row is usable, quarantined or retained
with uncertainty—not merely display a clean-looking table.
