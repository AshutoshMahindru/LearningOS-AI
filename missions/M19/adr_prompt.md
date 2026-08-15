# ADR prompt — gradient update and verification policy

This choice is consequential for V04 because a hidden sign or derivative error can make every later optimizer experiment misleading.

Using `templates/ADR.md`, decide how the Mathematical Instrumentation Layer will represent and verify gradient updates before multi-parameter training is accepted.

Address at least these options:

1. Analytic gradients only.
2. Finite-difference gradients only.
3. Analytic gradients with a central finite-difference check on deterministic fixtures.

Record the update convention (`parameter - learning_rate * gradient`), tolerance and epsilon policy, failure behavior, runtime cost, and why the rejected alternatives are insufficient. Include the wrong-sign fixture as an acceptance test. Do not pre-select the decision in learner evidence; the reviewer must evaluate the reasoning.
