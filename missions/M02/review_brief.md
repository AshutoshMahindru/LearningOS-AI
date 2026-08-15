# M02 Formal Engineering Review Brief

## Review decision requested

Determine whether the learner can operate and interrogate a complete supervised ML system, preserve evaluation integrity, and transfer that orientation to a fresh no-AI run. Review the evidence, not the apparent sophistication of the classifier.

## System map

```text
datasets/M02/wine.csv
  -> schema and numeric-integrity checks
  -> X (13 measurements) / y (3-class target)
  -> stratified, disjoint train/test indices
  -> StandardScaler + LogisticRegression.fit(training only)
  -> predict(held-out features)
  -> honest metrics against held-out truth
  -> confusion, error, coefficient, and sensitivity interrogation
```

Trust boundaries: test labels must not cross into fit or preprocessing; row alignment must survive prediction and evaluation; training labels must remain attached to their source rows; evaluation must compare truth with predictions for the same examples.

## Evidence to inspect

- Restart + Run All result from the source notebook in a clean environment.
- Baseline split integrity checks, class counts, prediction count, metrics, confusion matrix, and error table.
- Prediction-before-action entries and controlled comparison results.
- Exact identification of `.fit`, `.predict`, and metric boundaries.
- No-AI fresh system artifact and the learner's own explanation.
- Package tests, repository tests, validator output, and diff-safety review.

## Required experiments

| Dimension | Controlled change | Evidence expected | Review question |
|---|---|---|---|
| Split | seed/test proportion | size, balance, held-out metrics | Is sampling variation distinguished from model improvement? |
| Features | 13 versus 2 | named columns, metrics | Is reduced information interpreted rather than guessed? |
| Model | logistic versus shallow tree | train/test metrics | Are inductive-bias and overfit signals noted? |
| Hyperparameter | logistic `C` | train/test sweep | Is one knob changed while other settings remain fixed? |
| Labels | intact versus shuffled training labels | disagreement rate, honest metric | Is row-label integrity recognized as causal to failure? |
| Evaluation | holdout versus stratified CV | fold scores, mean, spread | Are estimates and uncertainty reported without cherry-picking? |

## Failure diagnosis

The corrupted-label experiment should produce evidence materially worse than the intact baseline. The invalid self-comparison should produce a deceptive `1.0`. Accept the diagnosis only if it names the wrong boundary/input relationship, proposes the smallest repair, and verifies the repaired honest metric. Merely noting that a score changed is insufficient.

## Validation criteria

The notebook must be deterministic where practical, CPU-only, secret-free, runtime-network-free, stable-cell-ID complete, and stored without outputs. Restart + Run All must succeed within 300 seconds. Mission tests must exercise dataset, notebook, contract, failure, and no-AI invariants through `unittest.TestCase`; repository validation and unittest discovery must remain green. No prohibited shared registry or tracking file may change.

## Limitations and residual uncertainty

Wine is small, clean, balanced enough for orientation, and not representative of messy or high-stakes production data. A single split can be optimistic or pessimistic; cross-validation reduces dependence on one split but does not prove deployment performance. Coefficient magnitude is model- and scaling-dependent, not causal importance. Label corruption here is obvious and synthetic; real labeling errors may be sparse, systematic, or correlated with groups. The formal review establishes system orientation and experimental reasoning, not mechanism mastery.

## V00 integration decision

Approve integration when the learner can reproduce the vertical slice, explain its boundaries, interrogate errors and sensitivity, reject invalid evaluation, and repeat the workflow unaided. Defer approval if evidence is copied, outputs cannot be reproduced, evaluation uses training information, or limitations are absent.
