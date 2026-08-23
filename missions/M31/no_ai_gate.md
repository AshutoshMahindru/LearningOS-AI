# No-AI gate — defend next-token training from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers below.

## Fixture (fresh)

Token ids (not from the notebook corpus):

```
tokens = (4, 8, 15, 16, 23)
```

Wrong collator example: `inputs = (4, 8, 15, 16)` and
`targets = (4, 8, 15, 16)`.

Lineage example: `train_ids = ("d01", "d02", "e02")`,
`eval_ids = ("e01", "e02")`.

NLL micro-case: logits `(0.0, 0.0, 0.0)`, target class `0`. Ignore
the teaching table; compute by hand.

## Part A: next-token pairs

Write correct `inputs` and `targets` for `tokens`. State the number of
prediction targets if `context_length=3`.

## Part B: causal-LM loss

1. Compute softmax NLL for the three-class micro-case.
2. Explain, in one or two sentences, what causal-LM loss is measuring.

## Part C: lifecycle

Draw pretraining → adaptation/post-training → evaluation and mark
where inference (frozen weights) sits. Label training-time versus
inference-time.

## Part D: spot the bug

1. What is wrong with the collator example above?
2. Is the lineage example a valid evaluation boundary? Why or why not?
3. Why is a low training loss alone weak evidence?

Pass requires pair arithmetic, an NLL, a lifecycle map, a shift or
contamination diagnosis, and an oral defense.
Leave all learner responses unfilled in the repository.
