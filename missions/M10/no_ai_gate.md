# M10 No-AI Gate — Choose From Unseen Consequences

Complete this gate without an AI assistant, generated formulas, or a copied solution. Use a calculator or plain Python only after writing your own formulas and prediction.

## Fresh scenario

A food-safety system scores lots for an undeclared-allergen audit. Auditing a safe lot is a false positive costing **4 units**. Failing to audit a truly unsafe lot is a false negative costing **60 units**. Correct outcomes have zero marginal cost in this bounded exercise. The audit team can process **at most 20 alerts per 100 lots**.

`datasets/M10/unseen_threshold_evidence.csv` contains aggregate evidence from a validation window. It has no recommended threshold or learner answer.

## Independent task

Before calculating, predict which threshold region will be defensible and which error must dominate the decision.

Then, for every candidate threshold:

1. verify that TP + FP + TN + FN equals 100;
2. compute accuracy, precision, recall, specificity and F1;
3. compute alert volume as TP + FP;
4. mark whether the candidate satisfies the 20-alert capacity limit;
5. compute expected cost as `4 × FP + 60 × FN`;
6. choose a feasible threshold and a primary operating metric;
7. defend the choice from consequences and evidence, not metric popularity;
8. name at least two assumptions that could reverse the decision;
9. define monitoring and a quantitative revisit condition.

## Challenge questions

- Would your answer change if false negatives cost 12 rather than 60 units?
- What information is lost if only ROC AUC is reported?
- What information is lost if only F1 is reported?
- If score calibration is unknown, what may still be concluded from this threshold table?

## Pass standard

Pass only when the arithmetic is correct, infeasible candidates are rejected, the chosen metric and threshold trace to stated consequences, uncertainty is explicit, and the reasoning survives oral challenge without AI assistance.
