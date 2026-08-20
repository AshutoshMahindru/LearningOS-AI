# M29 Formal Engineering Review Brief

## Review decision requested

Approve M29 as the V06 single-head attention package: an offline,
deterministic Q/K/V mission that inherits M16 matrix convention and
M28 vector intuition, and that explicitly defers multi-head residual
norm FFN composition to M30.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M29.

## System under review

- hand-computable two-key fixture with scores `(1, 0)`
- cash vs water teaching sequences (`bank` is geometrically ambiguous)
- causal mask, scale, query perturbation, and value-only experiments
- named defects `softmax_over_queries` and `mask_after_softmax`
- repair from the broken trace via `repair_attention`
- NumPy only; no secrets; no network; no pretrained download

## Required reviewer challenges

- verify M29 is blocked by M16 and M28 and hands a single-head trace to M30;
- independently softmax the two-key scores `(1, 0)` and match the core;
- confirm cash-context bank weights are uniform `1/3` and water-context
  bank puts more mass on `water`;
- confirm causal future mass is 0 and bank output becomes `(1.5, 0.5)`;
- confirm a value-only change keeps weights and moves the output;
- reproduce wrong-axis softmax breaking row sums, then `repair_attention`;
- search code cells for multi-head, residual, LayerNorm, FFN, torch,
  and model-hub downloads;
- verify source notebook IDs are unique, outputs empty, and labs-cwd
  import works;
- confirm learner evidence, ADR decisions, and completion remain
  unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
