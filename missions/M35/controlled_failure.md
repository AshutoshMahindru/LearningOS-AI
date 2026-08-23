# Controlled failure — evaluation leakage / metric gaming

## Failure: the average improved because the eval set leaked

Run the canonical ranking pipeline on the frozen M34 labels with
`candidate_k=3` and identity rerank. Then change **one** thing:

- `leaked=True` / `leak_eval_phrasing`

Query texts from the eval set are copied into gold support chunks.
The original corpus object is not mutated; a new leaked corpus is
returned. Labels stay frozen.

Predict, before running:

- whether aggregate nDCG/MRR rises
- whether `source_hash` differs from the clean canonical hash
- whether the leaked object still contains a query string after repair
  of a separate clean corpus

A related metric-gaming case (relabel top-1 as relevant after seeing
ranks) must also inflate metrics while ranked ids stay identical.
Give that case its own predict/act step. Do not treat it as a second
simultaneous repair of the leak.

## Repair rule

The smallest repair is **restore the clean eval boundary**. Call
`repair_eval_boundary(broken_corpus=..., source_corpus=..., frozen_labels=...)`
and rerun the baseline config on the restored objects. Do not "fix"
metrics by editing M34 labels, opening a model reranker, or adding
ANN.

Submit prediction, named defect, preserved invariants, observed metric
jump, root cause, smallest repair, verification, and the regression
that a leaked corpus stays leaked.

A repair is rejected if it opens M32/M36 mechanisms, relabels in the
same step, or changes several variables at once.
