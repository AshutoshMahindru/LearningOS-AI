# V03 — Model Comparison & Diagnostics integration

M11 supplies the interpretable-tree baseline for V03.

The flagship should retain:

- the deterministic train/test boundary used for comparisons;
- structural diagnostics: depth, node count, leaf count, and smallest leaf;
- per-row decision-path explanations rather than only aggregate metrics;
- experiments that change one tree constraint at a time;
- an explicit distinction between training fit and held-out behavior;
- feature-importance caveats covering correlation, instability, and non-causality;
- the ADR that records why a constrained tree was chosen or rejected.

M12–M14 can compare their models with this baseline, but they must not imply that a tree is globally interpretable merely because a single path can be traced. V03 comparison evidence should preserve both predictive metrics and model-specific diagnostic limits.
