# Controlled failure — two valid metrics, two different winners

The fixture contains a query, a far-away candidate pointing in exactly the same direction and a nearby candidate pointing in a different direction.

Cosine similarity should select the direction match. Euclidean distance should select the nearby point. This disagreement is deliberate: the measurements answer different questions.

Required diagnostic sequence:

1. record both predicted winners before calculation;
2. calculate candidate norms, dot products, cosine scores and Euclidean distances;
3. verify that the implementation ranks cosine high-to-low and distance low-to-high;
4. state whether direction or absolute closeness is the intended invariant;
5. inspect whether vectors were normalized before interpreting a dot product as cosine;
6. reproduce the normalization mistake by ranking with raw dot product;
7. repair the measurement pipeline, not the fixture;
8. rerun the semantic ranking as a regression check;
9. record the metric decision in `adr_prompt.md`.

A report that merely says “cosine is better” or “Euclidean is wrong” fails. The diagnosis must connect the winner to representation, magnitude, normalization and the intended invariant.
