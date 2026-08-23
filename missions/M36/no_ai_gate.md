# No-AI gate — choose infrastructure without a generated explanation

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Use `datasets/M36/transfer.json` only. Do not reuse notebook traces.

## Part A: exact versus approximate

Using `effort_example`, name the exact top-3 ids and the low-effort
ids. State why neighbor-recall can drop even though the embeddings
did not change. State when a simple exact in-memory index is still
preferable (`when_exact_is_enough`).

## Part B: late-filter candidate loss

Using `late_filter_example`, list the unfiltered top-1 id, the
filter, and the eligible relevant id. Say whether that relevant id
can appear in the late-filtered list, and why.

## Part C: fuse two ranked lists

Using `rrf_example` and `rrf_k=60`, compute RRF by hand. Show each
`1/(60 + rank)` term. Do **not** add the raw scores in
`raw_scores_trap`. Name the fused order.

## Part D: choose a channel

Using `query_choice`, pick dense, sparse, or hybrid for the lexical
ticket query and for the password paraphrase. One sentence each
naming the invariant you would hold fixed.

## Part E: incompatible scores

Using `raw_scores_trap`, say which id wins if you add cosine to BM25,
and which method would make the scores comparable.

Pass requires independent RRF arithmetic, a late-filter diagnosis,
an exact-versus-effort explanation, two channel choices, and an oral
defense. Leave all learner responses unfilled in the repository.
