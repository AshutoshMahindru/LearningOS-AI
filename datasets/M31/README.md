# M31 fixtures

Offline teaching corpus for **Understand LLM Training**.

These files are synthetic, deterministic, and authored for M31. They are
not a production pretraining corpus, not a Hugging Face dataset, and not
a quality benchmark. They require no download and no network.

- `corpus.json` — dataset version `v07-teaching-corpus-1`, train/eval
  document ids, texts, tokenizer identity (M27 word scheme `v06.1`),
  and the named leak target `e02`.

Checksums and split lineage are computed from this file at load time.
Do not treat falling training loss on this corpus as evidence of a
general language model.
