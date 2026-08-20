# M25 Formal Engineering Review Brief

## Review decision requested

Approve M25 as the V05 PyTorch training-loop package: an offline,
CPU-canonical autograd mission that inherits M24 reverse-mode numbers
and explicitly defers systematic multi-cause DL debugging to M26.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M25.

## System under review

- M24 teaching module (`3→2→3` affine-ReLU-affine) with matching initial parameters
- Forward parity on named intermediates and mean softmax-NLL
- Autograd versus M24 on W1, b1, W2, b2
- Canonical `zero_grad → forward → loss → backward → step`
- Deterministic 36-row fixture, splits 24/6/6, disjoint indices
- Batched `DataLoader` (`num_workers=0`, CPU generator)
- Train hidden width 8 so the loop is observable; teaching width stays 2
- Eval/no-grad validation; held-out only after freeze
- Checkpoint with model, optimizer, epoch, seeds, architecture, RNG
- Named defects: `stale_grad`, `train_mode_eval`
- Torch is a mission dependency; tests `skipUnless` so bare CI stays green

## Required reviewer challenges

- verify M25 is blocked by M24 and hands off to M26 without opening a diagnosis catalogue;
- reproduce forward logits `[[0,0,0],[1,1.5,-0.25]]` from the teaching module;
- reproduce autograd `W2[0,0] ≈ -0.3296553694702` matching M24;
- confirm `nn.Linear` grads are compared after a transpose to M24 layout;
- confirm one teaching SGD step (momentum 0) lowers loss;
- reproduce stale-grad doubling on two backwards without a step;
- reproduce train-mode Dropout evaluation changing logits without moving parameters;
- confirm held-out indices are disjoint from train and val;
- confirm `evaluate` raises if parameters would move and does not call `step`;
- confirm checkpoint reload replays held-out logits;
- search the notebook for weight downloads, CUDA requirements, secrets, and M26 fault-injection APIs;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass in the M25 environment (torch installed);
- bare repository unittest discovery stays green (`skipUnless` on torch for the runtime path);
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
