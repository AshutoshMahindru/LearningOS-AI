# Controlled failure — unsafe cleaning

The notebook deliberately demonstrates four plausible but unsafe shortcuts:

1. blanket `dropna` after converting blanks to missing values;
2. aggressive `drop_duplicates(subset="order_id")` after ID normalization;
3. silent numeric coercion via `to_numeric(errors="coerce")` without an issue log;
4. automatic deletion of every statistically flagged outlier.

These operations may produce a clean-looking table while destroying evidence.

Required diagnostic sequence:

1. predict which records or facts each shortcut will lose;
2. record raw and output row counts;
3. identify the first operation that makes the loss irreversible;
4. inspect the specific lost source rows;
5. distinguish an observed defect from a hypothesis about its cause;
6. state the decision the shortcut made implicitly;
7. name the missing evidence or uncertainty it erased;
8. replace the shortcut with the smallest auditable treatment;
9. rerun row reconciliation and invariant assertions;
10. explain why the safe repair addresses the cause rather than the appearance.

The controlled failure must remain caught and observable so notebook Restart +
Run All succeeds. A passing diagnosis does not merely say “do not use dropna”;
it quantifies what was lost and shows the repaired evidence trail.
