# M31 Formal Engineering Review Brief

## Review decision requested

Approve M31 as the V07 LLM-training package: an offline, deterministic
NumPy teaching simulation of the causal next-token objective, with
explicit pair construction, split lineage, contamination, and a
training-time versus inference-time boundary for M32.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M31.

## System under review

- eight-document synthetic corpus `v07-teaching-corpus-1`
- M27 word tokenizer `v06.1`, special tokens included
- correct shift `inputs = window[:-1]`, `targets = window[1:]`
- named defects `target_shift_wrong` and `held_out_leak`
- evaluation always uses correct shift on authored eval ids
- repair from the broken trace via `repair_run`
- independent next-token shift arithmetic locked in tests
- NumPy SGD on a (V, V) table; no secrets; no network; no pretrained download

## Required reviewer challenges

- verify M31 is blocked by M30 and hands a stage-aware artifact to M32;
- independently slice `SHIFT_TOKENS[:-1]` / `[1:]` and match;
- independently compute softmax NLL of `(0.0, 0.0)` at target 1 and match `log(2)`;
- confirm `target_shift_wrong` first diverges at `targets` and train ids still match;
- confirm `held_out_leak` first diverges at `train_doc_ids` with overlap `{e02}`;
- confirm `repair_run` uses the broken trace's documents, seed, and steps;
- search code cells for torch, model-hub downloads, temperature, top-k/p, and RAG;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `python3 tools/validate_repo.py` still reports M01-M26 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
