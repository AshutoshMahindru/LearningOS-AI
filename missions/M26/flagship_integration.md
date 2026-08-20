# V05 integration — diagnose after the loop is trusted

## M25 → M26 boundary

M25 hands over an instrumented CPU training loop, autograd parity with
M24, protected splits, eval/no-grad, and checkpoints. M26 must not
reimplement that loop. Import `missions.M25.pytorch_training` as the
trusted runner and inject one named control at a time.

The observable V05 debugging loop after M26 is:

```
known-good M25 trace
→ symptoms (no cause yet)
→ ranked hypotheses across data / opt / grad / arch / eval
→ cheapest discriminator
→ root cause
→ smallest repair on the broken object
→ rerun original evidence
→ regression check
```

## What M26 must not change

M26 does not reopen reverse mode, does not replace M25's loop with a
new trainer, and does not teach language-model internals. The teaching
fixture stays the M25 36-row CPU cluster task.

GPU is optional and educational. CPU is canonical. No silent download
of model weights.

## M26 → P5 handoff

P5 may open language-model mechanisms **only after** the learner can
defend this protocol on a non-text network. M26 does not teach those
mechanisms.

The phase-end Chaos Day is a V05 release gate: a hidden defect is not
a pass until ranking, discriminator, repair, and regression exist.
