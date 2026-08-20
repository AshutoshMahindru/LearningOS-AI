# No-AI gate — reconstruct the training loop from a skeleton

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

## Part A: fill the skeleton

From memory, fill a stripped loop. Do **not** copy the committed
notebook cells verbatim.

```text
model.____()                    # train or eval?
for batch_x, batch_y in loader:
    optimizer.____()            # what, and why here?
    logits = ____               # forward
    loss = ____                 # which object, which reduction?
    loss.____()                 # what gets written, and where?
    optimizer.____()            # what moves, what does not?
```

Explain each line of `zero_grad -> forward -> loss -> backward -> step`
in your own words: inputs, outputs, mutated state, and the failure if
that line is omitted.

## Part B: one trusted gradient

Load the M24 teaching weights into the module. Compute one autograd
gradient (for example `W2[0,0]` in M24 layout) and compare it to the
M24 reverse-mode number. State the `(out, in)` versus `(in, out)`
transpose explicitly.

## Part C: checkpoint and split identity

1. Load a checkpoint and run **held-out** inference with `eval()` and
   `no_grad`. Show that parameters do not move.
2. Given three unlabeled snippets (a training step, a validation loop,
   and a held-out load), identify which is which and name the
   discriminating signals (`zero_grad`, `step`, `model.eval`, split).

Pass requires independent reconstruction and an oral defense. Leave all learner responses unfilled in the repository.
