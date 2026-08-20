# M27 — Turn Text into Tokens

Models do not read English. They read **token pieces** mapped to
**integer IDs**, wrapped with special tokens, then padded or truncated
to a sequence length. The useful whole is:

`text → normalize → pieces → IDs → specials → pad/truncate → padding mask`

This mission inspects that boundary with a **bundled** teaching
tokenizer (`v06-teaching-tokenizer`, version `v06.1`). Two local
schemes run on the same text:

- **word** — whitespace and punctuation split, then vocabulary lookup
- **bpe** — the same pretokens, then a tiny frozen byte-pair merge list

The comparison is real. Common teaching words are often one piece in
both schemes. Rare identifiers, numbers, and URL-like strings explode
under BPE and become `[UNK]` under word lookup.

This mission does **not** turn IDs into vectors (M28), does **not**
weight those vectors by context (M29), and does **not** stack a
transformer block (M30). A padding mask is metadata: which positions
are real tokens versus `[PAD]`. How a later layer uses that mask is
deferred.

Canonical sources: `hf-llm-course` and `karpathy-zero-to-hero` via
`data/source_registry.json`. Use the bundled fixture; do not download
a production tokenizer.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
