# Code reading

Read this unfamiliar function without running it:

```python
def apply_layers(points, matrices):
    states = [points.copy()]
    for matrix in matrices:
        points = points @ matrix.T
        states.append(points.copy())
    return states
```

Trace it using a two-row batch and two non-commuting matrices.

Before execution, record:

1. the semantic role and shape of `points`;
2. the semantic role and shape of each `matrix`;
3. why `.T` appears;
4. which matrix acts first;
5. how many arrays `states` contains;
6. whether the caller's input is mutated;
7. the value of one landmark after every loop iteration;
8. how reversing `matrices` changes the result.

Then run the function and compare the first divergent value with your trace. Finally add shape validation that rejects a matrix whose input dimension does not equal the current feature dimension, without changing correct behavior.
