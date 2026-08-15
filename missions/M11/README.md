# M11 — Interrogate a Decision Tree

## Mission

Start from a trained shallow classifier, inspect its structure, and account for one prediction at a time. The goal is not merely to fit a decision tree. The goal is to explain how a sequence of learned threshold tests sends a row through branches to a leaf.

The learning loop is:

1. run the shallow baseline;
2. inspect nodes, thresholds, branches, leaves, sample counts, class counts, and impurity;
3. predict an individual decision path before asking the model for it;
4. compare shallow and deep trees on train and test data;
5. vary `max_depth` and `min_samples_leaf` one at a time;
6. perturb one feature and explain any changed path;
7. diagnose an over-deep tree from the generalization gap;
8. report feature importance as model-specific split usage, not causation;
9. refuse an unjustified causal interpretation.

## Interpretation vocabulary

- A **node** holds the rows that have reached one point in the tree.
- A **split** asks whether one feature is less than or equal to a learned threshold.
- The **left branch** is followed when that comparison is true; the **right branch** is followed when it is false.
- A **leaf** stops splitting and predicts from the class distribution of the training rows that reached it.
- **Impurity** summarizes how mixed the classes are at a node. A useful split makes the weighted children less mixed than their parent.
- **Depth** counts how many split decisions separate the root from a node.

## Interpretation boundary

This mission uses a synthetic fixture for learning mechanics. A feature can be selected often because it helps this fitted tree partition this dataset. That does not show that changing the feature will cause an outcome to change. Correlated features can share or substitute for importance, and small data changes can produce a different tree.

## Completion evidence

Completion requires recorded pre-action predictions, path traces, a depth comparison, a minimum-samples comparison, a perturbation analysis, train/test evidence, an overfitting diagnosis, a cautious importance statement, a causal-claim rejection, and a fresh no-AI transfer.
