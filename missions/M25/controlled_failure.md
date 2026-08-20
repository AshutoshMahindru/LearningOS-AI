# Controlled failure — stale gradients or train-mode evaluation

## Failure: the loop appears to run, one named control is wrong

Before running, predict whether a second reverse pass **without**
`zero_grad` adds into leftover `.grad`, and predict whether Dropout in
`model.train()` can change evaluation outputs without an optimizer step.

The defective path uses one named change:

- `stale_grad`: skip `optimizer.zero_grad()` on a training step, or
- `train_mode_eval`: score a split while `model.training` is still true
  (Dropout active) instead of `eval()` + `no_grad`.

The module class, teaching or fixture tensors, loss, optimizer family,
and learning rate stay fixed. Only the named loop control changes. The
defect can still emit finite losses and “learning-looking” updates.
That is the point.

Diagnosis comes from traces the loop already records: `zero_grad_called`,
`.grad` before reset, `model.training` during the forward, whether
`optimizer.step` ran, and split membership. Do not open a catalogue of
data, architecture, and learning-rate faults.

## Discriminators

Stale gradient: two reverse passes on the **same** teaching batch, no
step between them. With reset, the second `.grad` matches the first.
Without reset, the second `.grad` is the **sum**. After two full steps,
parameter values diverge from the correctly reset twin.

Train-mode evaluation: Dropout `p=0.5` on the teaching module. Correct
`eval()` matches the M24 forward (Dropout is identity). Train-mode
evaluation changes logits and loss. **Parameters do not move** in either
evaluation path — so a metric change is not evidence that training
happened.

## Repair rule

The smallest repair restores `zero_grad` before each training backward
and `model.eval()` plus `torch.no_grad()` for validation and held-out
inference. Do not change the teaching weights, do not widen the net, and
do not replace SGD.

Submit prediction, named defect, preserved invariants, discriminating
trace, root cause, smallest repair, and the repaired rerun.

A repair is rejected if it opens M26 mechanisms or changes several
variables at once.
