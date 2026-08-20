# No-AI gate — defend tokens from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

## Part A: a fresh teaching-tokenizer example

Using `v06-teaching-bpe` rules (normalize; pretokens; `▁` prefix;
frozen merges; `[BOS]`/`[EOS]`):

1. predict the pieces for a fresh in-vocab sentence that is **not**
   `the cat sat on the mat`;
2. predict the pieces for one rare identifier that is not in the word
   vocabulary;
3. explain token IDs versus token text.

## Part B: effective length

Given a short sentence and `add_special_tokens=True`:

1. compute content length and effective length including specials;
2. state what a `max_length` equal to content length (no extra room
   for specials) must drop if truncation is on.

## Part C: truncation diagnosis

You are given encoded output whose decoded prefix is missing a
trailing instruction. Diagnose:

1. whether the drop happened in a word/character budget or in token
   truncation;
2. why tokenization can change downstream behavior without itself
   encoding truth.

Pass requires independent traces, token-budget arithmetic, and an
oral defense. Leave all learner responses unfilled in the repository.
