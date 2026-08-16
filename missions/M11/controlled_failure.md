# Controlled failure

The notebook deliberately fits an unconstrained decision tree to a small dataset containing label noise. The model can make its training score look excellent by creating extra splits and small leaves.

## Failure A — apparent perfection

Do not accept training accuracy as sufficient evidence. Use this diagnostic sequence:

1. predict whether removing the depth constraint will raise or lower training accuracy;
2. record depth, node count, leaf count, smallest leaf size, train accuracy, and test accuracy;
3. compute the train–test generalization gap;
4. locate evidence that the deep tree is isolating small groups;
5. state one falsifiable overfitting hypothesis;
6. fit one constrained repair using `max_depth` or `min_samples_leaf`;
7. compare the repair on the unchanged test split;
8. explain why a smaller gap is relevant but does not by itself prove deployment fitness.

## Failure B — causal overreach

The deliberately invalid statement is:

> `practice_accuracy` has the largest impurity-based importance, so increasing practice accuracy will cause a learner to become ready.

Reject it. The fitted tree describes predictive associations in this synthetic sample. It does not identify interventions, control confounding, establish temporal order, or rule out correlated substitutes. Rewrite the statement to say what this particular model actually did.

Required evidence includes the observed failure, a repair comparison, and a corrected non-causal interpretation. Quietly deleting the deep tree or replacing the causal sentence without explaining the evidence does not pass.
