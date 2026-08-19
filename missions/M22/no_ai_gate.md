# No-AI gate — defend a neuron and a layer from arithmetic

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

## Part A: a fresh two-input neuron

Given a fresh pair of weights, a bias, and a two-feature input:

1. compute the weighted sum and the pre-activation separately;
2. apply ReLU and sigmoid by hand;
3. state what changes if only the bias is set to zero.

## Part B: a two-neuron layer

Given a fresh 3-feature input and a `(3, 2)` weight matrix:

1. compute both pre-activations;
2. apply the declared activation;
3. name every dimension in `X`, `W`, `b`, and `Y`.

## Part C: representation and repair

1. Explain why two stacked affine maps without a nonlinearity cannot
   represent more than one affine map.
2. Repair a fresh orientation or activation-boundary error from expected
   dimensions, without referring to gradients.

Pass requires independent arithmetic, explicit shape contracts, and an
oral defense. Leave all learner responses unfilled in the repository.
