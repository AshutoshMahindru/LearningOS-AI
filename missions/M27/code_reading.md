# Code reading — encode, specials, pad/truncate, identity

Read `TeachingTokenizer.encode`, `pack_for_context`, and `apply_bpe` in
`missions/M27/tokenization_core.py`. M27's code-reading target is the
**text-to-ID contract**:

1. normalization (lowercase, strip, collapse whitespace)
2. pretokens (words and punctuation)
3. scheme-specific pieces (lookup vs frozen BPE merges)
4. optional `[BOS]` / `[EOS]`
5. right-truncation of **content** so specials still fit `max_length`
6. right-padding with `[PAD]` and a padding mask
7. decode back to normalized text
8. tokenizer name, version, and `downloaded=false`

Before running the code-reading cell, predict:

- what happens if `max_length=4` with special tokens added
- whether extra spaces in the source text survive normalization
- whether `pack_for_context(..., budget_unit="words")` can report
  `heuristic_fit=True` while still dropping a suffix after BPE

Do **not** look for embedding tables, attention scores, or a transformer
block. Those are M28-M30. If a failure can be diagnosed from token
counts and a dropped suffix, stay at that level.
