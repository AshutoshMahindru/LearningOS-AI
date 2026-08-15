# V04 — Mathematical Instrumentation Layer integration

M17 adds probability and uncertainty instrumentation to V04.

The Operations Intelligence System should be able to carry these artifacts forward:

- an explicit event and reference population for every reported probability;
- empirical outcome frequencies beside model probabilities;
- cohort-level reliability summaries that can reveal deployment shift;
- a base-rate-aware interpretation of alerts;
- dependence checks before multiplying probabilities;
- thresholds justified by expected consequences;
- an ADR that records the evidence and assumptions behind an operational threshold.

This mission does not add a production model or promise that a probability is certain. It supplies the reasoning layer needed to inspect model scores built in earlier flagship versions and to support later statistical inference, evaluation, and reliability controls.

The V04 boundary is explicit: the notebook computes mathematical instrumentation from a static synthetic fixture and seeded simulations. It does not modify shared runtime registries, call a remote model, or create fabricated operational evidence.
