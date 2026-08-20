# Code reading — stored values, local rules, reverse order, accumulate, reset

Read `TapeNode`, `build_scalar_chain_tape`, `reverse_accumulate`, and
`two_layer_backward` in `missions/M24/backprop_core.py`. M24's
code-reading target is the **reverse-mode contract**:

1. forward values are stored on a tape or as named intermediates
2. each node records local derivatives to its parents
3. reverse order is the reverse of the M23 graph
   `loss → logits → hidden_activation → hidden_preactivation → x`
4. a parent of several children **adds** incoming contributions
5. `reset=True` zeros stored grads before adding a fresh reverse pass
6. `reset=False` adds another copy of the same reverse pass (stale grads)
7. `relu'(z)` is `1` if `z > 0` else `0`, including `z == 0`
8. softmax + mean NLL uses `dL/dlogits = (p - one_hot) / N`
9. central finite differences check a smooth parameter; a ReLU hinge is
   not a valid check

Before running the code-reading cell, predict:

- what `dL/dh` is when `h` feeds two MSE heads
- what happens to stored `dL/dw` if you reverse twice without reset
- whether example 0's hidden unit at `z = -0.5` receives a ReLU gradient
- whether `dL/dW2` still matches finite differences when `dL/dH` omits a class

Do **not** look for `torch.autograd`, `nn.Module`, an optimizer, or an
epoch loop. Those are M25. If a failure can be diagnosed by comparing
one local analytic gradient with a central difference, stay there.
