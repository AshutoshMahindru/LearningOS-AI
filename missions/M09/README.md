# M09 — Make Binary Decisions

M09 builds one complete binary decision system before unpacking its parts:

**baseline → split → classifier → predicted probabilities → default classification → confusion matrix → threshold changes → consequences**

The working case predicts whether a learner will disengage in the next 30 days. `1` means the event occurs; `0` means it does not. The dataset is synthetic, so the exercise is about decision reasoning rather than a claim about real learners.

## Run the mission

From the repository root, install `requirements/m09.txt`, open `labs/M09_binary_classification.ipynb`, and use **Restart Kernel and Run All Cells**. The source notebook intentionally contains no saved outputs.

The notebook uses only Python's standard library at runtime. It reads the local CSV, makes a deterministic stratified split, fits logistic regression with batch gradient descent, emits probabilities, applies several thresholds, and relates TP/TN/FP/FN to accuracy, precision, recall, and an explicit consequence model.

## Learning contract

For every threshold experiment, record a prediction before running the relevant cell. Explain changes by referring to cases that crossed the threshold and the resulting confusion-matrix cells. A threshold is part of the decision policy; it is not learned automatically by the probability model in this mission.

Completion requires the evidence in `evidence_contract.yaml`, the controlled-failure diagnosis, and the fresh decision in `no_ai_gate.md`. Generated notebook output is practice evidence, not proof that a learner completed the mission.

## Threshold ADR handoff

Choosing an operating threshold from FP/FN consequences and capacity is a consequential policy decision. After recording predictions and completing the threshold experiments, use `missions/M09/adr_prompt.md` and `templates/ADR.md` to create a **separate learner-authored ADR**. It must record the selected threshold and comparison rule, cost and capacity assumptions, alternatives, evidence, accepted trade-offs, owner/status/date, monitoring, rollback, and measurable revisit triggers.

The repository prompt is intentionally unfilled and must not be converted into fabricated completion evidence. Notebook output alone does not satisfy the ADR requirement.

## Boundaries

- CPU-only and deterministic where practical.
- No paid API, secret, or runtime network access.
- No real personal data.
- No claim that this toy model is suitable for deployment.
- No global registry, tracking, or lab-status mutation is part of M09.
