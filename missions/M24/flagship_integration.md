# V05 integration — assign blame after the forward graph is trusted

## M19 → M23 → M24 boundary

M19 hands over the one-parameter invariant: analytic gradient, central
finite difference, and `parameter - learning_rate * gradient`.

M23 hands over the named inference graph and a trusted forward-parity
test. M24 must not invent a different forward graph without stating the
change. Import `missions.M23.forward_core` as that graph.

The observable V05 reverse graph after M24 is:

```
loss
  -- softmax+NLL --> d_logits = (p - y) / N
  -- affine W2  --> d_hidden_activation  (branches add over classes)
  -- relu'      --> d_hidden_preactivation
  -- affine W1  --> d_x, d_W1, d_b1
```

plus the matching `d_W2`, `d_b2`.

## What M24 must not change

M24 does not retrain M21, does not author a PyTorch module, and does not
run an epoch loop. One declared update is local evidence that the sign is
usable. It is not V05 training.

## M24 → M25 handoff

M25 may compare autograd to M24 only after the learner can defend:

- chain rule on a scalar tape
- addition at a branch
- ReLU local derivative, including dead units
- softmax-NLL logit gradient
- central finite-difference checks on smooth parameters
- the ReLU-at-zero hinge as a check caveat
- a repaired omitted-branch or wrong-ReLU defect
- one downhill step without a convergence claim

M25 implements framework autograd and a training loop. It must not
silently replace M24's reverse-mode numbers. M24 does not call
`torch.autograd`.
