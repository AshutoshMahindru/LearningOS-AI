# M27 fixtures

Offline teaching tokenizer and texts for **Turn Text into Tokens**.

These files are synthetic, deterministic, and authored for M27. They are
not a Hugging Face model, not a tiktoken encoding, and not a quality
benchmark. They require no download and no network.

- `teaching_tokenizer.json` — special tokens, word vocabulary, BPE
  alphabet, frozen merge list, and tokenizer identity (`v06.1`).
- `texts.json` — canonical sentence, surface variants, rare strings,
  padding batch, truncation example, comparison corpus, controlled-failure
  text, and the BPE training corpus used to freeze the merges.
- `generate_tokenizer.py` — regenerates the JSON from the bundled corpus.
  Canonical tests load the frozen files; they do not retrain.

M28 and M29 may reuse these token/text fixtures and must budget context
in **tokens**, not characters or words.
