# Controlled failure — target shift or held-out leak

## Failure: loss falls, evidence is still wrong

Use the teaching corpus, seed `3101`, and the `(V, V)` bigram table.
Predict, before running, which constructed field would disagree with
`tokens[i+1]` if the collator is off by one position, and which lineage
field would become non-empty if a protected eval document entered train.

Then run one named defect.

The defective path uses one named change:

- `target_shift_wrong`: train targets equal `window[:-1]` (predict the
  current token) while evaluation still uses `window[1:]`, or
- `held_out_leak`: copy authored eval id `e02` into `used_lineage.train_ids`
  while the eval id list and NLL formula stay unchanged.

Tokenizer, seed, steps, learning rate, and context length stay fixed.
Only alignment or train membership changes.

The defect can still emit a falling **training objective** curve. That
is the point. Diagnosis comes from:

1. constructed `(input, target)` pairs versus `window[1:]`,
2. split overlap `train_ids ∩ eval_ids`,
3. true next-token NLL on authored eval (always correct shift),
4. the hand shift `(10, 20, 30, 40) → inputs (10, 20, 30), targets (20, 30, 40)`.

## Discriminators

Wrong target shift: `train_doc_ids` still match; `inputs` still equal
`window[:-1]`; `targets` equal `window[:-1]` instead of `window[1:]`.
Training objective loss can fall while true next-token NLL on the same
texts stays high.

Held-out leak: `targets` remain causal; `train_doc_ids` gain `e02`;
`leaked_ids` is `("e02",)`; authored eval ids are unchanged. Held-out
NLL can fall because the example is no longer held out.

Do not start with learning-rate or step-count hypotheses. Read the
pairs and the lineage first.

## Repair rule

The smallest repair calls `repair_run` on the **broken trace** (same
documents, seed, steps, context length, learning rate) so alignment is
correct and train ids are the authored split. Do not change the corpus
texts, do not add a decoder, and do not open M32 sampling.

Submit prediction, named defect, preserved fields, first divergence,
root cause, smallest repair, verification, and the regression that the
broken path still diverges.

A repair is rejected if it opens M32-M34 mechanisms, if it is two
unrelated `defect="none"` runs, or if it changes several variables at
once.
