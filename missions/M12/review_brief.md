# M12 Formal Engineering Review Brief

## Review decision

Decide whether the proposed ensemble experiment is strong enough to support a V03 model-selection trial. This review does **not** approve production deployment and does not prescribe which model must win.

## Artifacts in scope

- the executed M12 notebook and its prediction log;
- the deterministic fixture and generation notes;
- the learner's controlled-failure diagnosis;
- the learner-authored ADR produced from `adr_prompt.md`.

## Architecture/system map and meaningful artifact

The system under review is:

```text
committed fixture → seeded stratified split → limited tree baseline
                                      ├→ bootstrap samples → parallel trees → averaged vote
                                      ├→ bootstrap + feature sampling → random-forest vote
                                      └→ sequential weak trees → additive boosted prediction
all candidates → held-out balanced accuracy + generalization gap → ADR → review verdict
```

The meaningful artifact is the reproducible notebook experiment plus its mission-local dataset and contract. Training state belongs to each fitted estimator; fixture ownership and label quality remain data responsibilities. The source notebook exposes code and prompts but contains no fitted models, execution output, learner evidence, or production endpoint.

## Required review questions

### Experimental validity

- Is the limited depth-2 tree preserved as the baseline?
- Are data split, random seeds, and balanced-accuracy definition held fixed across comparisons?
- Are bagging, random forest, and boosting compared without silently changing unrelated controls?
- Does the learner separate test observations from a recommendation that would require cross-validation or a validation set?

### Mechanism and uncertainty

- Is bootstrap disagreement described as sample sensitivity rather than a formal decomposition?
- Is random-forest feature randomness distinguished from bagging?
- Is boosting explained as sequential loss correction, with both corrected and newly wrong cases inspected?
- Are estimator-count and depth interactions addressed?

### Operational consequences

- Does the ADR consider training cost, inference latency, memory, parallelism, interpretability, and reproducibility?
- Are the proposed estimator count and depth bounded by evidence and a budget?
- Are label quality and monitoring treated as separate controls rather than outsourced to model capacity?

### Failure analysis

- Does the diagnosis explain why repeated resampling preserves corrupted target signal?
- Does it challenge both excess tree complexity and “more trees always fixes it”?
- Are repair and revalidation steps explicit?

### Current tests/evaluations

- Did the source notebook pass Restart + Run All with no prefilled source outputs?
- Did the mission unittest and pytest runs validate dataset, notebook, review, ADR, and evidence-contract invariants?
- Did repository unittest discovery and `tools/validate_repo.py` pass without changes to shared tracking or lab-status files?
- Is the controlled failure reported as evidence from a synthetic fixture rather than as a guarantee about V03?

## Unresolved uncertainty

The fixture cannot establish which ensemble, capacity, latency, or calibration behavior will transfer to V03 data. The learner must record what validation dataset and uncertainty estimate would change the proposed decision.

## Reviewer challenge prompts

- Why this design, and what can be removed without weakening the claim?
- What fails, where is fitted state, and who owns the data and label-quality controls?
- What evidence supports the recommendation, what is exposed operationally, and what would change the decision?
- For every substantive comment, should the learner accept, reject, or defer it, and why?

## Severity guide

- **Blocker:** leakage, test-set tuning presented as production selection, missing baseline, non-reproducible run, or a conclusion unsupported by reported evidence.
- **Major:** incorrect ensemble mechanism, unsupported bias/variance claim, no capacity or operational trade-off, or no revisit condition.
- **Minor:** presentation or traceability gap that does not change the decision.

## Required review output

The reviewer records a verdict (`approve experiment`, `revise`, or `reject`), findings with evidence locations, unresolved uncertainty, and required follow-ups. The repository intentionally does not prefill that output.
