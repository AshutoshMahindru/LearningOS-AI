# M24 Formal Engineering Review Brief

## Review decision requested

Approve M24 as the V05 reverse-mode package: an offline, deterministic
credit-assignment mission that inherits M19's gradient invariant and
M23's named forward graph, and that explicitly defers framework autograd
and a training loop to M25.

This is an implementation review, not learner sign-off. Formal
engineering review is required at M24.

## System under review

- M19 one-parameter invariant restated on a two-point fixture
- M23 teaching graph plus mean softmax-NLL with targets `(0, 0)`
- scalar affine-ReLU-affine-MSE chain with hand-computable grads
- branch accumulation on a shared hidden value
- ReLU hinge including `relu'(0) = 0`
- central finite-difference checks and an epsilon sweep
- one declared descent step
- named omitted-branch and wrong-ReLU defects
- NumPy-only runtime; no secrets; no network

## Required reviewer challenges

- verify M24 is blocked by M19 and M23 and hands off to M25 without opening autograd;
- reproduce scalar-chain grads `dL/dw = 6`, `dL/dv = -2`, `L = 2` by independent arithmetic;
- reproduce branch contributions `2 + 1 = 3` for `dL/dh`;
- confirm `dL/dlogits = (p - one_hot) / N` on the M23 batch;
- confirm example-1 (smooth) parameter grads match central differences;
- confirm full-batch ReLU-at-zero entries are hinges, not silent passes;
- reproduce omitted-branch as a `dW1`/`db1` mismatch with `dW2` still matching;
- reproduce wrong-ReLU as a dead-unit mismatch (`z < 0`);
- confirm one step lowers loss and is not looped as training;
- search the notebook code cells for `import torch`, `autograd`, `nn.Module`, `DataLoader`, and epoch loops;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass in the M24 environment;
- bare repository unittest discovery stays green (`skipUnless` on NumPy for the network path);
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
