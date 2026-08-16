# M09 No-AI gate — Fresh probability and threshold decision

Complete this gate without AI-generated analysis or code. Do not reuse the notebook's threshold table.

## Fresh scenario

A quality team assigns these defect probabilities to eight new batches:

| Batch | Predicted defect probability |
|---|---:|
| A | 0.08 |
| B | 0.19 |
| C | 0.27 |
| D | 0.36 |
| E | 0.48 |
| F | 0.59 |
| G | 0.73 |
| H | 0.88 |

Sending a batch to inspection costs **2 units** when it is actually good (FP). Shipping a defective batch without inspection costs **9 units** (FN). The team can inspect at most four batches.

## Part A — Commit before outcomes

Without looking ahead to Part B:

1. Define positive class `1` in plain language.
2. Choose a probability threshold that respects the four-batch capacity.
3. List the batches your threshold classifies as positive.
4. Predict how lowering the threshold would affect FP, FN, precision and recall.
5. Record why `0.50` is or is not appropriate. A bare reference to “the default” earns no credit.

## Part B — Reveal and evaluate

Only after recording Part A, use these outcomes:

| Batch | Actual defect target |
|---|---:|
| A | 0 |
| B | 0 |
| C | 1 |
| D | 0 |
| E | 1 |
| F | 0 |
| G | 1 |
| H | 1 |

By hand:

1. Compute TP, TN, FP and FN for your threshold.
2. Compute accuracy, precision and recall.
3. Compute consequence cost as `2 × FP + 9 × FN`.
4. Repeat for threshold `0.50`.
5. Keep or revise your decision and explain the consequence trade-off.
6. Explain why a predicted probability is not a guarantee for one batch.
7. Describe, without naming a library, what evidence would make you trust or distrust calibration across many batches.

## Pass standard

Pass requires a committed pre-outcome policy, correct confusion-matrix accounting, correct metric denominators, an explicit capacity/consequence argument, and a calibration explanation based on long-run agreement between predicted and observed rates. Arithmetic without a decision rationale is insufficient.
