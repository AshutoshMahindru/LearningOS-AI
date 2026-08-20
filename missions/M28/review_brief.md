# M28 Formal Engineering Review Brief

## Review decision requested

Approve M28 as the V06 embedding-retrieval package: an offline,
deterministic nearest-neighbor mission that inherits M15 metric
discipline and M27's text boundary, and that explicitly defers
attention to M29 and a search service to M33.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M28.

## System under review

- bundled `v06-teaching-meanpool` v06.1 (mean pool, L2, cosine, 12-D)
- paraphrase, lexical/semantic, hard-case, and normalization experiments
- mixed `v06.2` axis-swap store and `ProvenanceError` repair
- NumPy cosine/ranking; no secrets; no network; no encoder download

## Required reviewer challenges

- verify M28 is blocked by M15 and M27 and hands fixtures to M29/M33;
- independently cosine-rank `q-password` on the frozen unit vectors;
- confirm a paraphrase keeps account documents above printer traps;
- confirm printer Jaccard can rank `d-login-reset` high while cosine
  ranks printer documents first;
- confirm deny-refund beats approve-refund with approve still > 0.85;
- confirm fifty vs thousand and 4412 vs 4413 as hard neighbors;
- reproduce inner-product length bias on sum-pooled printer documents;
- reproduce silent mix retrieving a printer doc for a password query
  and the `ProvenanceError` repair;
- search code cells for model-hub downloads, softmax, Q/K/V, and
  index/service APIs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd
  import works;
- confirm learner evidence, ADR decisions, and completion remain
  unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
