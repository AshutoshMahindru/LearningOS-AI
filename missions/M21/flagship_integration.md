# V05 integration start — Neural Learning Component

## M20 → M21 boundary

M20 hands over a monitored training discipline: declared optimizer policy, learning-rate
awareness, seed policy, loss/update monitoring, stagnation/divergence triggers, and
rollback behavior. M21 applies that discipline to a complete neural estimator without
claiming to understand its internal neuron mathematics yet.

The observable V05 layer after M21 is:

`dataset/split → preprocessing → declared neural estimator → fit → learning/validation traces → held-out metrics → error profile`

## M21 implementation contract

M21 establishes that a complete neural model can learn useful held-out signal, be replayed
under a fixed seed, show bounded seed sensitivity, fail because of insufficient training
budget or corrupted labels, and be diagnosed from whole-system evidence.

M21 explicitly does not expose weight matrices, activations, manual forward-pass equations,
gradients, backpropagation, or autograd. Implementation completion also does not satisfy
learner competence; evidence and the ADR remain unfilled until produced by the learner.

## M21 → M22 handoff

M22 may open the neuron and layer only after the learner can defend one accepted M21 run:
its split, preprocessing, architecture knobs, optimizer settings, seed, training budget,
learning curve, validation evidence, held-out metrics, confusion-matrix errors, failure
responses, and acceptance policy. M22 explains internals; it must not retroactively alter
M21's held-out evidence to make those internals look successful.
