# M04 dataset

`customer_orders_dirty.csv` is a synthetic, deterministic training fixture. It
contains no real customer data.

The fixture deliberately combines:

- blank required values;
- one exact duplicate row;
- conflicting and normalization-colliding order IDs;
- whitespace and case variation;
- category, region and status aliases plus unmapped values;
- numeric strings and Indian-rupee punctuation;
- malformed numeric text;
- mixed, ambiguous, missing, malformed and out-of-range dates;
- non-positive values, an invalid ID/email and total reconciliation failures;
- statistical outliers with different evidence: one approved wholesale order
  and one unexplained suspicious total.

Do not repair this source file. The mission contract requires lossless raw
loading and reproducible cleaning in `missions/M04/cleaning.py`.

Expected row accounting is asserted in `tests/missions/test_m04.py`; the exact
duplicate is logged, while conflicts and all outliers remain in cleaned output.
