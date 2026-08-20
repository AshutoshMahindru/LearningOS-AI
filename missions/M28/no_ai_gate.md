# No-AI gate — defend embeddings from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Use `datasets/M28/transfer.json` only. Do not reuse notebook rankings.

## Part A: rank by cosine

The query vector is already unit length. For each corpus vector:

1. compute cosine similarity by hand (dot product; they are L2);
2. rank with ties broken by id;
3. name the nearest item.

## Part B: one success and one failure

1. Explain one semantic success (low lexical overlap, high cosine, or
   the reverse if that is what you observe).
2. Explain one failure (negation, a lexical trap, or a score that does
   not prove equivalence).

## Part C: provenance

Identify the mismatch probe. State which fingerprint fields disagree
with the transfer corpus and why scoring it would be invalid.

## Part D: what a score is not

State, in one or two sentences, what an embedding score does **not**
prove.

Pass requires independent arithmetic, an honest hard case, and an oral
defense. Leave all learner responses unfilled in the repository.
