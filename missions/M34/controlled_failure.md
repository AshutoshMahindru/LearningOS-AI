# Controlled failure — fluent answer, unsupported citation

## Failure: the sentence looks right, the citation does not

Run the canonical pipeline on `rag-reset-login`
(`How do I reset my login credentials?`) with `top_k=3`. The pack
contains the gold procedure span and two account neighbors.

Then change **one** mapping:

- `defect="unsupported_citation"`

Query, index, retriever, pack, and answer text stay fixed. The
defective path cites a packed neighbor that does **not** contain the
claim stems.

Predict, before running:

- whether the answer text still reads as the login-reset procedure;
- whether `verify_support` is ok;
- whether evaluation `primary` is citation rather than retrieval.

A related invented-support case (ticket `4412` at `top_k=1`, claim
about 4412 citing 4413) must also fail support. Diagnose retrieval
and citation as separate layers. Give that case its own predict/act
step. Do not treat it as a second simultaneous repair of the
login-reset citation.

## Repair rule

The smallest repair is **rebind or abstain**. Call
`repair_grounding(answer, pack)` so the citation becomes a packed
span that actually supports the claim, or the system abstains. Do
not "fix" grounding by editing the query, relabeling support ids,
opening a live model, or adding a reranker.

Submit prediction, named defect, preserved invariants, observed
unsupported citation, root cause, smallest repair, verification, and
the regression that unsupported citations fail evaluation.

A repair is rejected if it opens M32/M35/M36 mechanisms or changes
several variables at once.
