# Controlled failure — word/character budget, token truncation

## Failure: the heuristic says it fits, the suffix is gone

Before running, count whitespace **words** in:

`please inspect ticket 4412 then approve_refund`

Predict whether that word count fits a 12-token window if you treat
one word as one token. Then predict the **BPE** length, including
`[BOS]` and `[EOS]`.

The defective path uses one named change:

- `budget_unit="words"` (or `"characters"` with a 4-characters-per-token rule)

and then encodes with `max_tokens=12` and right-truncation.

The tokenizer, the text, and the token limit stay fixed. Only the
budget unit is wrong. The pipeline can still emit IDs. That is the
point. Diagnosis comes from measuring actual token length and from
seeing that `approve_refund` is missing from the decoded prefix.

## Repair rule

The smallest repair counts **tokens** with the same tokenizer that will
consume the text. `budget_unit="tokens"` makes overflow visible:
`silent` is false, `dropped_text` records the lost suffix, and
`on_overflow="raise"` refuses to ship a silently clipped instruction.

Do not keep the suffix by switching schemes, by deleting special
tokens, or by inventing embeddings. To **keep** `approve_refund`, the
token budget must be at least the measured BPE length.

Submit prediction, named defect, preserved invariants, observed
decoded prefix, root cause, smallest repair, and the repaired rerun.

A repair is rejected if it opens M28-M30 mechanisms or changes several
variables at once.
