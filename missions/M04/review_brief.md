# M04 Formal Engineering Review Brief

## Raw state

`datasets/M04/customer_orders_dirty.csv` is deliberately loaded as strings with
blank strings preserved. It contains interacting missingness, exact duplication,
normalized-ID conflicts, whitespace/case/category variation, currency text,
malformed numerics, mixed and malformed dates, impossible values, arithmetic
violations and outliers that require judgment. Source CSV row numbers are added
before transformation.

## Data contract

- `order_id` matches `ORD-####` and is unique in analysis-ready data.
- region, category and status belong to declared canonical domains.
- units are positive whole numbers; prices and totals are positive.
- accepted date formats are explicit and day-first is declared for slash dates.
- order dates fall between 2025-01-01 and 2026-12-31.
- email has a minimal structural form.
- order total reconciles to units × unit price within ₹0.01.
- required analysis fields are non-missing.

## Architecture

`load_raw` provides a lossless boundary. `clean_orders` performs exact-duplicate
logging, raw-field preservation, normalization, explicit parsing, missing-total
derivation, constraints, outlier flagging and readiness classification. It
returns raw, cleaned, analysis-ready, duplicate-log, issue-summary and row-count
products. `assert_analysis_ready` is the downstream gate. No global registry or
tracking state is changed.

## Evidence

Every cleaned row contains `source_row`, `raw_*`, `observed_defects`,
`decision_log`, `evidence`, `uncertainty`, `information_loss`, outlier fields and
`analysis_ready`. Exact duplicates survive in a separate audit log. The notebook
reconciles counts and compares representative raw/canonical values.

## Controlled failure

The lab measures the loss caused by blanket dropna, aggressive ID dedupe, silent
numeric coercion and automatic outlier deletion. Failures are caught so Restart
+ Run All succeeds. Repair consists of issue logging, conflict preservation,
targeted derivation, explicit quarantine and retained outlier judgment.

## Validation

Formal review requires notebook Restart + Run All, substantive mission unittest
and pytest runs, repository unittest discovery, repository validation, clean
diff checks and confirmation that prohibited shared files were not modified.
Analysis-ready assertions cover non-missingness, identity, domains, whole
positive units, positive money, date range, total reconciliation and absence of
blocking issues.

## Uncertainty and information loss

An ambiguous slash date is interpreted by the declared day-first policy and
flagged. Conflicting IDs remain unresolved pending source-owner evidence.
Statistical outliers remain in cleaned data; an approved wholesale note supports
retention while an unexplained total requires review. Raw values remain beside
canonical values. The only row removal is an exact duplicate, retained in the
audit log with its source row and original ID.

## V01 integration

M04 supplies V01 Structured Data Workbench's auditable ingestion slice. The
workbench can expose raw-to-canonical transformations, issue evidence, review
queues and an invariant-gated analysis table instead of treating cleaning as an
opaque preprocessing step.

## Reviewer prompts

1. Does the data contract match the intended order semantics?
2. Is each irreversible action justified and row-accounted?
3. Can a reviewer recover the original evidence for every canonical value?
4. Are conflicts, parse failures and outliers visible without contaminating analysis?
5. Do the tests prove data invariants rather than only artifact presence?
