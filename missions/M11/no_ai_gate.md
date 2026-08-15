# No-AI gate

Complete this gate without AI-generated code or prose.

Use this fresh textual tree:

```text
|--- practice_accuracy <= 0.73
|   |--- attendance_pct <= 86.50
|   |   |--- class: 0
|   |--- attendance_pct > 86.50
|   |   |--- class: 1
|--- practice_accuracy > 0.73
|   |--- study_hours_week <= 4.50
|   |   |--- class: 0
|   |--- study_hours_week > 4.50
|   |   |--- class: 1
```

Tasks:

1. Predict the class and write every comparison for `(study_hours_week=6, practice_accuracy=0.70, attendance_pct=91)`.
2. Repeat for `(study_hours_week=4, practice_accuracy=0.80, attendance_pct=95)`.
3. Identify every node, branch, and leaf used by each row.
4. Explain why a leaf prediction represents the training rows that reached it, not a universal rule.
5. Suppose the first tree has train accuracy `0.91` and test accuracy `0.88`, while an unconstrained tree has train accuracy `1.00` and test accuracy `0.70`. Choose one for deployment and justify the choice using more than training fit.
6. State how increasing `min_samples_leaf` is expected to affect tiny leaves before testing it.
7. Critique: “Practice accuracy is the first split, so raising practice accuracy will cause readiness.” State the missing evidence and rewrite it as a defensible model-specific observation.
8. Explain one reason feature importances could change after adding a correlated feature.

Passing requires correct path traces, a train/test argument, a hyperparameter prediction, and a non-causal interpretation. Answers that merely name the final classes do not pass.
