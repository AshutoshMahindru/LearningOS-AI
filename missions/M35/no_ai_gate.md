# No-AI gate — score ranking without a generated explanation

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Use `datasets/M35/transfer.json` only. Do not reuse notebook traces.

## Part A: rank-sensitive metrics

For the listed 3-hit result, compute by hand:

1. Recall@3
2. MRR
3. nDCG@3 using the listed grades (support=2, relevant=1) and
   `gain / log2(rank + 1)`

Show the DCG and IDCG terms, not only the final ratio.

## Part B: diagnose two failures

1. Using `low_recall`, say whether the miss is candidate generation or
   ranking, and why a reranker cannot recover it.
2. Using `poor_ranking` / the main candidate list, name the top-1 id
   and say whether candidate recall is already 1.0.

## Part C: choose a lever

Given those two diagnoses, choose **one** of chunking, candidate
generation (k), or reranking and write one sentence that names the
invariant you would hold fixed.

## Part D: spot leakage

`leakage_example` copies the eval query into an irrelevant cafe chunk.
State why an improved cosine on that chunk is not evidence of a better
retriever.

## Part E: candidate recall versus final ranking

In one or two sentences, explain the difference between "gold was in
the candidate set" and "gold is ranked first."

Pass requires independent arithmetic, two diagnoses, a single-lever
choice, a leakage rejection, and an oral defense. Leave all learner responses unfilled in the repository.
