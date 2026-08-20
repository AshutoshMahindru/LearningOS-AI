# M27 Implementation Review Brief

## Review decision requested

Approve M27 as the V06 tokenization package: an offline, deterministic
text-to-token mission that inherits Python fluency from M03 and
explicitly defers embeddings, attention, and transformer blocks.

This is an implementation review, not learner sign-off.

## System under review

- bundled `v06-teaching-tokenizer` v06.1 (word + tiny BPE)
- special tokens, padding, truncation, padding masks
- two-scheme comparison on a fixed corpus
- word/character-budget controlled failure and token-budget repair
- stdlib tokenizer core; no secrets; no network; no model download

## Required reviewer challenges

- verify M27 is blocked by M03 and hands off fixtures to M28/M29;
- encode `the cat sat on the mat` by independent tracing on both schemes;
- confirm casing/whitespace collapse and punctuation as a named change;
- confirm a rare string UNKs on word and fragments on BPE;
- confirm batch padding preserves non-pad prefixes;
- confirm truncation drops a named suffix;
- reproduce the silent `approve_refund` drop and the token-budget repair;
- search code cells for embedding, attention, transformer, tiktoken, and downloads;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass (core is stdlib; no skipUnless required);
- bare repository unittest discovery stays green;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
