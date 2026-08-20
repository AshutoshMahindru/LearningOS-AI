# No-AI gate — defend a retriever from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Use `datasets/M33/transfer.json` only. Do not reuse notebook rankings.

## Part A: exact top-k

The query vector is already unit length. For each corpus vector:

1. compute cosine similarity by hand (dot product; they are L2);
2. rank with ties broken by `chunk_id`;
3. write top-3 ids and scores.

## Part B: one query trace

Trace `t-q-down` end to end: query text → vector → eligible rows →
scores → top-k → evidence fields you would hand to M34 (id, score,
text, span). Do not generate an answer.

## Part C: labeled success

Using the frozen relevant ids (do not edit them):

1. compute hit@1 and recall@2;
2. name the high-scoring item that is **not** labeled relevant.

## Part D: stale index

The stale probe changes live text for `t-offline` without rebuilding.
State what unchecked search still returns, which hashes must disagree,
and the smallest repair (rebuild or reject).

## Part E: what a score is not

State, in one or two sentences, why a high similarity score is not
answer correctness.

Pass requires independent arithmetic, an honest hard case, a stale-index
diagnosis, and an oral defense. Leave all learner responses unfilled
in the repository.
