# Code reading — register, reset, step, mode, checkpoint, held-out

Read `TwoLayerNet`, `training_step`, `evaluate`, and `save_checkpoint` /
`load_checkpoint` in `missions/M25/pytorch_training.py`. M25's
code-reading target is the **training-loop contract**:

1. `nn.Module` registers `fc1`/`fc2` parameters; they are leaves with
   `requires_grad=True`
2. `nn.Linear` stores `weight` as `(out, in)`; M24's `W` is `(in, out)`
3. `optimizer.zero_grad()` clears `.grad` before a new reverse pass
4. `loss.backward()` writes autograd into `.grad`; it does not change
   parameter values
5. `optimizer.step()` applies the update and may write **optimizer
   state** (momentum buffers) that is not `.grad`
6. skipping `zero_grad` **adds** a second reverse pass into leftover
   `.grad` (the M24 tape `reset=False` hazard, now named)
7. `model.train()` versus `model.eval()` is a mode flag; Dropout uses it
8. `torch.no_grad()` is an inference context; it is not `eval()`
9. `evaluate` never calls `optimizer.step`, so parameters stay put
10. a checkpoint carries model weights, optimizer state, epoch, seeds,
    split policy, and RNG — not only `state_dict` weights
11. held-out indices are disjoint from train and val; the loop must not
    step on them

Before running the code-reading cell, predict:

- what `.grad` contains after `backward` but before `step`
- what happens to `W2[0,0]`'s stored grad if you reverse twice without
  `zero_grad` on the teaching batch
- whether `evaluate(..., defect="none")` can change parameter tensors
- whether train-mode evaluation with Dropout can change **outputs**
  without changing **parameters**
- which checkpoint keys are required to resume versus to run held-out
  inference

Do **not** open a multi-cause diagnosis catalogue (data corruption,
learning-rate search, blocked gradient paths). Those are M26. If a
failure is a missing `zero_grad` or a train-mode eval, stay there.
