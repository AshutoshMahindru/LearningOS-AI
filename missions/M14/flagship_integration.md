# M14 Flagship Integration — V03 Model Comparison & Diagnostics

M14 extends V03 beyond supervised model comparison. The V03 diagnostic layer
must now support a model whose assignments have no target-label accuracy metric.

The mission contributes:

- an explicit feature-and-distance contract before fitting;
- deterministic K-means candidate comparison;
- internal diagnostics at both aggregate and sample levels;
- original-unit center profiles suitable for cautious downstream discussion;
- stress tests for scale, arbitrary `k`, outliers, and lossy visualization;
- an ADR boundary between mathematical evidence and product interpretation.

V03 should expose cluster assignments and diagnostics as analytical outputs, not
as stable identities, risk tiers, or true classes. Any downstream action requires
a separately reviewed purpose, external validation evidence, and monitoring for
drift or harmful interpretation.
