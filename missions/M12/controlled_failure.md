# M12 Controlled Failure — More Trees Cannot Repair the Wrong Signal

## Seeded fault

The notebook flips a deterministic 28 percent of the training labels while leaving the held-out labels clean. It then grows random forests from 1 to 200 estimators and also exposes the generalization gap of increasingly deep individual trees.

## Prediction before action

Before running the failure cell, write down whether clean-test balanced accuracy will rise monotonically and whether a larger forest can infer which training labels were flipped. Give a mechanism, not only a direction.

## Learner diagnosis

Use the reported curve and gaps to answer:

1. Which quantity is stabilized by adding trees?
2. Which source of error remains present in every bootstrap sample?
3. Does train performance on corrupted labels reveal learning or memorization?
4. At what point are returns negligible or negative?
5. What evidence would justify cleaning labels, limiting depth, changing the loss, or collecting more data?
6. Why does this result refute the unqualified statement “more trees always fixes it”?

## Repair experiment

Restore clean training labels, choose a bounded estimator count and depth from held-out evidence, and rerun the same metric table. Do not tune on the test set in a real project; the mission uses one fixed holdout for a compact demonstration, so the ADR must call for cross-validation or a validation set before production selection.

## Failure boundary

The expected learning outcome is a diagnosis, not a prescribed winner. A run is valid even if the exact score curve changes under a supported library version, provided the learner distinguishes variance reduction from corrupted-target bias and explains the limits of the experiment.
