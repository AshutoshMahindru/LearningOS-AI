# M06 Review Brief

## System map

`local CSV fixtures → dataframe validation → question-specific transformation → chart/table → observation/inference/limitation record → next question`

Field timing is a boundary: information recorded after case closure must not cross into a future new-ticket feature set.

## Meaningful artifact

`labs/M06_see_the_data.ipynb` is a CPU-only, network-free visual interrogation lab. It covers distributions, missingness, outliers, record-level relationships, group comparisons with denominators, class imbalance, and possible leakage. A second dataset is reserved for independent transfer.

## Failure diagnosis

The controlled failure feeds identical channel-level mean tenure values into two bar charts but truncates one vertical scale. The code succeeds and the values are correct, yet the first encoding exaggerates the apparent operational difference. The repair restores a zero baseline and the sample's full 0–50 month range while keeping exact values visible.

## Decision record

No ADR is required before M07. The mission-local decision is to use transparent synthetic fixtures so the notebook is deterministic, distributable, and free of privacy or network dependencies. The trade-off is that patterns are pedagogically designed and cannot support population claims.

## Tests and evaluation

The mission suite validates artifact coverage, notebook cell IDs and clean source state, network-free code, dataset quality conditions, a measurable outlier, class imbalance, perfect association in the deliberately leaky post-outcome field, fresh-gate separation, and explicit reasoning language.

## Unresolved uncertainty

The guided fixture cannot show whether patterns reproduce in a real support operation or whether response time causes escalation. A real deployment would need documented collection timing, sampling coverage, subgroup review, and prospective validation.
