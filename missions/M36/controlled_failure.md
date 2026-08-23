# Controlled failure — incompatible scores or late filters

## Failure A: hybrid silently mixes raw score scales

Run exact dense cosine and BM25 sparse on `rag-password-procedure`
with the same store. Then change **one** thing:

- `mix_raw_scores(dense, sparse)`

Cosine (~0–1) is added to BM25 (different units). The original
`ChannelResult` objects are not mutated. Labels stay frozen.

Predict, before running:

- whether `doc-refund-policy::c1` can outrank `doc-account-access::c1`
- whether the mixed fusion field equals `raw-sum`
- whether repairing with RRF from the **same** dense/sparse objects
  changes the broken mix object

## Failure B: filter after a small top-k

On query `Please reset` with `topic=account` and k=1, take unfiltered
exact top-1 and drop non-matching payloads. Eligible gold
`doc-account-access::c1` never entered the small list.

The notebook uses **Failure A** as the primary repair drill and keeps
Failure B as the filter-placement regression. Both remain observable
from candidate ids.

## Repair rule

The smallest repair is **declared rank fusion** (RRF) or **pre-filter
then retrieve**. Call `repair_fusion(broken=..., dense=..., sparse=...,
store=...)` or `repair_filter_placement(...)` and inspect repaired
ids. Do not "fix" metrics by editing M34 labels, opening a live
Qdrant client, or adding cosine to BM25 with a new weight.

Submit prediction, named defect, preserved invariants, observed id
lists, root cause, smallest repair, verification, and the regression
that the broken object still fails.

A repair is rejected if it opens M32/M37 mechanisms, relabels, or
changes several variables at once.
