# WP-123: Experiment Pedagogy Contract (Predict &rarr; Commit &rarr; Run &rarr; Observe &rarr; Explain)

## 1. The 5-Step Protocol

```
1. PREDICT   ──► Learner articulates expected system behavior, metric, or shape
        ↓
2. COMMIT    ──► Sealed to database with timestamp & hash (Immutable)
        ↓
3. RUN       ──► Code executes in isolated worker; structured result captured
        ↓
4. OBSERVE   ──► Interactive inspection of actual tables, charts, tensors, logs
        ↓
5. EXPLAIN   ──► Learner explains delta between prediction and observation
```

## 2. Enforcement Rules
- **Rule 1 (The Seal)**: Execution button is disabled until the learner selects or writes a prediction payload conforming to the stage's hypothesis schema.
- **Rule 2 (No Retrospective Alteration)**: Once committed, the prediction record is locked. Subsequent executions cannot overwrite the initial hypothesis.
- **Rule 3 (Explanation Requirement)**: Stage completion requires submitting a synthesis explanation reconciling discrepancies between hypothesis and empirical results.
