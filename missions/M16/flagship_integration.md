# V04 — Mathematical Instrumentation Layer integration

M16 supplies the matrix transformation and batch-shape instrument for V04.

The learner should carry these habits forward:

- view a matrix as a function with an input dimension and output dimension;
- predict basis-vector and landmark movement before plotting;
- treat composition order as executable architecture, not typography;
- name batch, feature and output axes;
- justify transposes from representation conventions;
- validate sample-wise equivalence rather than trusting a plausible batch result;
- use non-symmetric counterexamples to expose orientation mistakes.

The bridge to ML is direct. With a row-major batch `X` shaped `(batch, in_features)` and weights `W` shaped `(in_features, out_features)`, a dense layer computes `X @ W + b`. Under the mission's single-column-vector view, the same weights are represented as `A = W.T`, so each sample is `A @ x` and the row batch is `X @ A.T`. The formulas agree once the representation boundary is explicit.

M16 evidence supports later V04 work on probability, statistics, gradients and optimization because those systems also depend on precise shape, axis and composition reasoning.
