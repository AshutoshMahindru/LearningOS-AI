# V05 integration — train with autograd after reverse mode is trusted

## M24 → M25 boundary

M24 hands over reverse-mode gradients, the softmax-NLL logit rule, ReLU
local derivatives, branch addition, and finite-difference check
expectations on the M23 graph. M25 must not invent a different teaching
forward graph without stating the change. Import
`missions.M24.backprop_core` as that reference.

The observable V05 training loop after M25 is:

```
optimizer.zero_grad()
logits = model(batch_x)          # model.train()
loss = mean_softmax_nll(logits, batch_y)
loss.backward()                  # fills .grad
optimizer.step()                 # updates parameters; may write optimizer state

model.eval()
with torch.no_grad():
    val_or_held_out_metrics(...)  # never optimizer.step
```

plus a checkpoint that can reload the accepted state.

## What M25 must not change

M25 does not re-derive reverse mode, does not replace M24 numbers with
unchecked autograd, and does not run a multi-cause debugging catalogue.
The teaching micro-case stays the M24 two-layer fixture. The training
fixture may use a wider hidden layer so the loop is observable; that
change is named.

GPU is optional and educational. CPU is canonical. No silent download of
model weights.

## M25 → M26 handoff

M26 may inject faults **only after** the learner can defend:

- parameter ownership and the Linear layout transpose
- autograd parity on selected M24 grads
- `zero_grad -> forward -> loss -> backward -> step`
- gradient accumulation without reset
- train/eval and no-grad
- evaluation that cannot update parameters
- disjoint train/val/held-out
- checkpoint round-trip, including optimizer state versus `.grad`
- a repaired stale-grad or train-mode-eval defect

M26 receives an instrumented loop whose data, optimization, architecture,
and evaluation controls can be broken independently. M25 does not teach
that catalogue.
