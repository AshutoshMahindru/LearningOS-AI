# V04 — Mathematical Instrumentation Layer integration

M15 creates the vector measurement interface for V04. It does not turn the learning runtime into a vector database and does not require an embedding API.

The lab produces inspectable measurements:

- named vector dimensions and shapes;
- vector magnitudes;
- normalized directions;
- dot products;
- cosine similarities;
- Euclidean distances;
- deterministic top-k rankings;
- explicit disagreement between metrics.

Every measurement must retain three pieces of context: the representation, preprocessing and metric. That context prevents a plausible score from being treated as self-explanatory.

M16 can apply the same vectors in matrix transformations. M17–M20 can add uncertainty, statistical, gradient and optimization instruments around the same pattern: **predict → measure → compare → diagnose → decide**.

The M15 ADR prompt makes the boundary consequential. V04 callers must decide whether the application needs direction, absolute closeness or magnitude-sensitive alignment, and must record how zero vectors and normalization are handled.
