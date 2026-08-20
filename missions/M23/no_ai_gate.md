# No-AI gate — defend a two-layer forward pass from array operations

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

## Part A: a fresh two-layer forward pass

Given a small parameter dictionary with `X`, `W1`, `b1`, `W2`, `b2`
(not the committed teaching fixture copied verbatim):

1. implement the forward pass by hand or in a fresh script;
2. annotate every intermediate shape **before** running it;
3. manually verify one example against the loop-level arithmetic.

## Part B: logits versus probabilities

1. State which named array is safe to treat as class probabilities and why.
2. Show that adding the same constant to one example's logits leaves
   those probabilities unchanged.
3. State what `argmax` on logits versus probabilities must agree on.

## Part C: wrong-axis softmax

Given a `(2, 3)` logit matrix, diagnose a result where each **column**
sums to 1 but a singleton run on row 0 disagrees with batch row 0.
Repair only the softmax axis. Do not introduce a loss or a gradient.

Pass requires independent arithmetic, explicit shape contracts, and an
oral defense. Leave all learner responses unfilled in the repository.
