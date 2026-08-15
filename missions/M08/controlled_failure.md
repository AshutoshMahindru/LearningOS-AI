# Controlled Failure — Deceptively Strong Regression

The dataset contains `post_sale_assessment_k`. It is a near-copy of the sale price produced after the transaction closes. It is retained only as a controlled failure fixture and must never appear in the safe feature set.

The notebook intentionally fits a model that includes this field. A random train/test split and ordinary cross-validation both report extraordinary results because the forbidden relationship exists in every partition. This is **target leakage across time**, not genuine predictive performance.

The notebook also evaluates an unrestricted decision tree on its training rows. Its near-perfect training score is an **invalid estimate of generalization**, not evidence that the model will perform equally well on unseen cases.

Required diagnostic sequence:

1. predict what each invalid evaluation will report;
2. record the safe model, leaky model and training-only metrics;
3. verify that row overlap is not the leakage mechanism;
4. write the real-world prediction timestamp;
5. audit when each feature becomes available;
6. identify the first contract violation;
7. remove every post-outcome field;
8. refit using training data only;
9. rerun cross-validation and the held-out evaluation;
10. add an allow-list and schema test as prevention;
11. explain why neither a random split nor cross-validation automatically detects time-of-availability leakage.

A weaker repaired score is acceptable. Trustworthy evaluation is the objective.
