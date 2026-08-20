# No-AI gate — defend a reverse-mode trace from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

## Part A: a fresh scalar graph

Given a small graph that is **not** the committed teaching fixture copied
verbatim (new numbers, still affine → ReLU → affine → squared loss):

1. name every node and write the local derivative at each edge;
2. reverse-accumulate by hand to every parameter;
3. perform one central finite-difference check on one parameter.

## Part B: a tiny neuron / network

1. Compute parameter, activation, and loss gradients for a tiny
   dense-layer-plus-activation example.
2. State which object is `dL/dW`, which is `dL/dh`, and which is `dL/dlogits`.

## Part C: branch, update, and diagnosis

1. Explain why a value that feeds two downstream paths adds contributions.
2. Predict the sign of one parameter update
   (`parameter - learning_rate * gradient`) and the local loss direction.
3. Given a plausible backward pass that fails a finite-difference check
   on `dL/dh` but not on `dL/dW2`, name the omitted-branch hypothesis
   and the smallest repair.

Pass requires independent arithmetic and an oral defense. Leave all learner responses unfilled in the repository.
