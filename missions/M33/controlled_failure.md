# Controlled failure — stale index after a corpus change

## Failure: the ranking looks current, the source is not

Build the canonical exact index over `datasets/M33/corpus.json`. Then
change **one** live chunk (`doc-account-access::c0`) to

`Please reset the printer firmware.`

without rebuilding. Query `q-password` stays fixed.

Predict, before running:

- whether unchecked search still returns the **indexed** password text
  and a plausible cosine ranking;
- whether `source_hash` of the live corpus disagrees with the index;
- what `search(..., enforce_freshness=True)` does with that live corpus.

The defective path uses one named change:

- `enforce_freshness=False` (`search_unchecked`)

The query, metric, and stored vectors stay the same. Only the live
source text changed. Diagnosis comes from index metadata (hashes) and
from served text that no longer matches the live chunk.

A related incompatible-index case (M28 `v06.2` swapped-axis store)
must also fail closed when `enforce_provenance=True`. Do not treat that
as a second simultaneous repair.

## Repair rule

The smallest repair is **rebuild or reject**. Call
`rebuild_index(index, live_corpus)` so records, texts, vectors, and
`source_hash` come from the live source, or refuse to search. Do not
"fix" staleness by editing the query, relabeling relevant ids, opening
Qdrant, or generating an answer.

Submit prediction, named defect, preserved invariants, observed stale
hits, root cause, smallest repair, verification, and the regression
that silent stale serving is rejected.

A repair is rejected if it opens M29/M34–M36 mechanisms or changes
several variables at once.
