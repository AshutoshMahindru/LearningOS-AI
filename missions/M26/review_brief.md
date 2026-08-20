# M26 Formal Engineering Review Brief

## Review decision requested

Approve M26 as the V05 deep-learning diagnosis package: an offline,
CPU-canonical failure-injection mission that inherits the M25 training
loop and explicitly defers language-model mechanisms to P5.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M26 (P4 phase end).

## System under review

- Unmodified M25 `train_model` as the useful whole
- Named faults: `label_shuffle`, `feature_scale`, `lr_high`, `lr_low`,
  `frozen_layer`, `tiny_hidden`, `crushing_dropout`, `train_mode_eval`,
  `val_leakage`
- Hidden practice seed `2602` and Chaos Day seed `2625` (category not
  printed in the public symptom report)
- Hypothesis ranking and a cheap diagnostic battery
- Smallest repairs driven from the prepared objects
- Evaluation repairs that keep the checkpoint frozen
- Torch is a mission dependency; tests `skipUnless` so bare CI stays green

## Required reviewer challenges

- verify M26 is blocked by M25 and does not rewrite `training_step`;
- reproduce a healthy M25 trace with falling train loss on CPU float64;
- reproduce train-only label corruption with val/held-out labels honest;
- reproduce a scaled feature column with ratio ≥ 10;
- reproduce high-LR oscillation and low-LR flat loss;
- reproduce `fc1` frozen (`requires_grad` false, weights unmoved) while `fc2` trains;
- reproduce tiny-width underfit versus the known-good width;
- reproduce train-mode evaluation and train-as-val leakage on a frozen checkpoint;
- confirm repair tests edit the broken objects rather than comparing two `defect="none"` runs;
- confirm Chaos Day public symptoms omit defect and category;
- search the notebook for weight downloads, CUDA requirements, secrets, and P5 language-model APIs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass in the M26 environment (torch installed);
- bare repository unittest discovery stays green (`skipUnless` on torch for the runtime path);
- `python tools/validate_repo.py` still reports exactly M01-M22 executable;
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
