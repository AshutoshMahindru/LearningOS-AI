# Controlled failure — mixed embedding provenance

## Failure: the ranking looks like search, the stores do not match

Query `q-password` (`v06-teaching-meanpool` `v06.1`, L2, cosine) against
the corpus in `datasets/M28/mismatch.json` (`v06-teaching-meanpool-alt`
`v06.2`, normalization `none`, account and print axes swapped).

Predict, before running:

- whether the unchecked ranking will still return named document ids
  and scores that look like similarities;
- which class of document a password query might retrieve after an
  account/print axis swap;
- which fingerprint fields must disagree (`model`, `version`,
  `normalization`).

The defective path uses one named change:

- `enforce_provenance=False`, with the declared cosine metric

The texts stay the same. Only the store identity is wrong (model,
version, normalization, and swapped account/print axes). Diagnosis
comes from metadata and from a password query retrieving a printer
document.

## Repair rule

The smallest repair calls `assert_compatible` (or
`rank_neighbors(..., enforce_provenance=True)`) and raises
`ProvenanceError` listing the mismatched fields. Do not "fix" the mix
by swapping axes back in the scorer, by changing the query text, or by
opening an index service.

Submit prediction, named defect, preserved invariants, observed mixed
ranking, root cause, smallest repair, verification, and the regression
that silent mixing is rejected.

A repair is rejected if it opens M29-M33 mechanisms or changes several
variables at once.
