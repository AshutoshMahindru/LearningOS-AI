# No-AI gate

Complete this transfer without AI-generated code.

Use these labeled points:

```text
guided:     (1, 8), (2, 7), (3, 9), (7, 3)
independent:(7, 8), (8, 7), (9, 9), (3, 2)
query:      (6, 6)
```

Tasks:

1. Predict the query label for `k=1`, `k=3` and `k=5` before calculating.
2. Compute Euclidean distances by hand and show the ordered neighbor table.
3. Implement Euclidean distance, neighbor ordering and majority vote without using `KNeighborsClassifier`.
4. Verify the implementation against the hand calculation.
5. Multiply the first coordinate by 100 without changing its meaning; predict and observe any neighbor change.
6. Repair the scale mismatch using statistics computed only from the training points.
7. Explain one tie policy and one reason a very large `k` can erase a useful local pattern.
8. Add one irrelevant high-scale coordinate, diagnose its effect, and decide whether scaling or removal is the better repair.

Passing requires correct code, hand-worked distance evidence, predictions recorded before execution, and a plain-language explanation of every changed neighborhood.
