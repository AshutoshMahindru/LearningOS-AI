# M30 Formal Engineering Review Brief

## Review decision requested

Approve M30 as the V06 transformer-block package: an offline,
deterministic NumPy block that composes M29 single-head attention,
declares a pre-norm teaching convention, and explicitly defers LLM
training to M31 and inference/adaptation to M32.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M30.

## System under review

- 4-D cash/water teaching sequences; `d_model=4`, `n_heads=2`, `d_ff=8`
- identity-split heads wrapping `scaled_dot_product_attention`
- declared pre-norm; labeled post-norm comparison
- named defects `residual_wrong_branch` and `norm_wrong_boundary`
- repair from the broken trace via `repair_block`
- independent residual arithmetic (`x + sublayer`) locked in tests
- NumPy only; no secrets; no network; no pretrained download

## Required reviewer challenges

- verify M30 is blocked by M29 and hands a block dataflow map to M31;
- independently add `RESIDUAL_STREAM + RESIDUAL_SUBLAYER` and match;
- independently LayerNorm `(3, 1)` (eps-aware) and match;
- confirm raw identity MHA head 0 matches M29 cash weights `1/3`;
- confirm `residual_wrong_branch` first diverges at `attn_add` and
  earlier checkpoints still match;
- confirm `norm_wrong_boundary` first diverges at `attn_norm`;
- confirm `repair_block` uses the broken trace's `x` and parameters;
- search code cells for torch, model-hub downloads, next-token loss,
  temperature, and adaptation policy;
- verify source notebook IDs are unique, outputs empty, and labs-cwd
  import works;
- confirm learner evidence, ADR decisions, and completion remain
  unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
