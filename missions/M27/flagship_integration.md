# V06 integration — freeze the token boundary

## M03 → M27 boundary

M03 hands over Python fluency: strings, lists, functions, and tracing.
M27 is the first V06 mission. It does not assume M15 vectors, and it
does not assume any transformer vocabulary.

The observable V06 surface after M27 is:

`text → pieces → ids` with a named tokenizer (`v06-teaching-word` or
`v06-teaching-bpe`), version `v06.1`, specials `[PAD] [UNK] [BOS] [EOS]`,
right-padding, right-truncation, and a padding mask.

## What M27 must not change

M27 does not invent meaning for IDs, does not retrieve nearest
neighbors, and does not score token-to-token relevance. Those numbers,
if they appear later, are M28/M29 measurements on top of this
boundary.

## M27 → M28 / M29 handoff

M28 may attach vectors to these IDs only after the learner can defend:

- tokens versus words
- tokenizer identity and version
- specials counting toward length
- padding versus truncation
- context budgets in tokens
- a repaired word/character-budget defect
- the frozen fixtures in `datasets/M27/`

M29 may use the padding mask as a keep/drop vector. It must not
relabel that mask as an explanation of model intent. M30 still owns
the block.
