# V06 integration — freeze the embedding contract

## M15 / M27 → M28 boundary

M15 already ranks small vectors with an explicit metric. M27 already
turns text into token IDs. M28 is the first V06 mission that **attaches
sentence vectors** and retrieves with them.

The observable V06 surface after M28 is:

`query text → 12-D unit vector (v06-teaching-meanpool v06.1, mean, L2)
→ cosine against a corpus of the same fingerprint → ranked ids`

plus a lexical Jaccard baseline that is allowed to disagree.

## What M28 must not change

M28 does not score token-to-token relevance, does not introduce Q/K/V,
and does not expose an index or HTTP retriever. Those are M29 and M33
measurements on top of this contract.

## M28 → M29 / M33 handoff

M29 may treat these vectors as a starting intuition for *context
changes the representation*. It must not relabel a cosine score as
attention.

M33 may index the fixtures only after the learner can defend:

- embeddings as operational similarity, not truth
- L2 + cosine as a declared pair
- lexical versus semantic disagreement
- negation, numeric, entity, and domain misses
- refused mixed provenance
- the frozen files in `datasets/M28/`

Reusable artifacts: `embeddings.json` provenance, `catalog.json` query
labels, `mismatch.json` as the negative compatibility case.
