# No-AI gate — manual gradient and faulty-loop repair

Complete this gate without AI-generated code or explanations.

## Part A: tiny manual update

For `x = [-1, 0, 1]`, `y = [-2, 0, 2]`, model `prediction = weight * x`, starting `weight = 0.5`, and mean squared error:

1. Write all three predictions and residuals.
2. Calculate the mean squared loss.
3. Calculate `2/n * sum(x * residual)` by hand.
4. With learning rate `0.25`, calculate the updated weight using `weight - learning_rate * gradient`.
5. Calculate the new loss and state whether the evidence supports the update.

## Part B: repair from a trace

Given this unfamiliar loop, do not replace the loop wholesale:

```python
for step in range(4):
    loss = one_parameter_loss(xs, ys, weight)
    gradient = analytic_weight_gradient(xs, ys, weight)
    weight = weight + learning_rate * gradient
    print(step, weight, loss)
```

Predict the behavior, identify the single faulty operator, make the smallest repair, and produce a four-step trace showing that loss falls. Explain the chain `parameter → prediction → loss → gradient → update` in your own words.

Pass requires correct arithmetic, a minimal repair, a decreasing-loss verification trace, and an explanation without borrowed code or AI assistance.
