# M32 Formal Engineering Review Brief

## Review decision requested

Approve M32 as the V07 inference-and-adaptation package: an offline,
deterministic local logits/sampling fixture with greedy, temperature,
top-k/top-p, seed replay, stop budgets, prompt/context effects, an
adaptation decision rubric, and a training-time versus inference-time
consume of M31's checkpoint.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M32 (P5 phase end).

## System under review

- 4-token local score table (`alpha`, `beta`, `gamma`, `stop`)
- independent greedy argmax on `(1.0, 3.0, 2.0, 0.0)` → index `1`
- independent softmax of `(log 3, 0)` at `T=1` → `(0.75, 0.25)` and at
  `T=0.5` → `(0.9, 0.1)`
- filter counts `(10, 6, 3, 1)`; teaching order temperature → top-k → top-p
- named defects `uncontrolled_settings` and `wrong_adaptation`
- repair from the broken trace via `repair_run`
- optional live adapter raises `LiveAdapterUnavailable`
- no secrets; no network; no pretrained download

## Required reviewer challenges

- verify M32 is blocked by M31 and hands InferenceConfig to M33/M34/M37;
- independently argmax `GREEDY_LOGITS` and match;
- independently softmax `TEMP_LOGITS` at `T=1` and `T=0.5` and match;
- confirm greedy index is invariant to temperature for `T > 0`;
- confirm `uncontrolled_settings` first diverges at `temperature` and
  naive compare says `model_changed`;
- confirm `wrong_adaptation` proposes `parameters` on stale hours and
  repair chooses `retrieval` from the broken signals;
- confirm `repair_run` uses the broken trace's reference config or signals;
- search code cells for torch, model-hub downloads, VectorIndex, RAG
  packs, and tool execution;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass; NumPy runtime tests use skipUnless;
- bare repository unittest discovery stays green;
- `python3 tools/validate_repo.py` still reports M01-M31 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
