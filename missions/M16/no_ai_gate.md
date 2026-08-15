# No-AI gate

Complete this transfer without AI-generated code.

You receive a fresh `(n, 2)` set of polygon vertices stored one point per row. The requested pipeline is:

1. scale horizontal coordinates by `1.5` and vertical coordinates by `0.5`;
2. shear horizontally with factor `-0.4`;
3. rotate counter-clockwise by `30°`;
4. apply the pipeline to the entire batch in one expression.

Deliver:

1. a sketch or written landmark prediction before execution;
2. the three elementary matrices;
3. the composite matrix with application order explained;
4. the batch expression and a principled explanation of any transpose;
5. a shape table for all operands and results;
6. an explicit step-by-step calculation for one landmark;
7. an assertion that the composite batch equals the three-step batch;
8. a counterexample showing that one reversed order produces a different answer;
9. one deliberate dimension or orientation mistake, its observed symptom, and its repair;
10. a plain-language connection to a dense ML layer.

Passing requires correct code, predictions, numeric evidence and explanation. A plausible plot alone does not pass.
