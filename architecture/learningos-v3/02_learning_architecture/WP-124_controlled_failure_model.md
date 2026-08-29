# WP-124: Controlled-Failure & Fault Injection Model

## 1. Principles of Controlled Failure
True systems architects are defined by their ability to debug under failure. Controlled-failure stages inject specific architectural faults, data corruptions, numerical instabilities, or agent alignment failures to develop diagnostic reflexes.

## 2. Fault Archetypes
1. **Data Pipeline Faults**: Schema skew, missing values, silent type coersions, train/test leakage.
2. **Numerical & Optimization Faults**: Gradient vanishing/explosion, NaN loss, learning rate divergence, rank deficiency in covariance matrices.
3. **Model & Architecture Faults**: Off-by-one autoregressive masking, wrong normalization axes, tokenization truncation.
4. **Agentic & Tool Faults**: Hallucinated tool arguments, infinite tool call loops, context window overflow, fragile regex parsing.

## 3. Repair Protocol
- **Diagnosis Phase**: Learner isolates the fault line and identifies root cause.
- **Hypothesis Phase**: Learner proposes the fix and rationale.
- **Verification Phase**: Regression tests verify the fix without breaking existing invariants.
