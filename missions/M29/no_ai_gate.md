# No-AI gate — defend single-head attention from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers below.

## Fixture (fresh)

Query (one position), `d_k = 2`, `d_v = 3`, scale `1/sqrt(d_k)`:

- `q = (1.0, 2.0)`
- `k0 = (1.0, 0.0)`, `k1 = (0.0, 1.0)`, `k2 = (1.0, 1.0)`
- `v0 = (1.0, 0.0, 0.0)`, `v1 = (0.0, 1.0, 0.0)`, `v2 = (0.0, 0.0, 1.0)`

This query sits at **position 1** in a three-key sequence (0-based).

## Part A: one query, no mask

1. Compute the three raw scores `q · k_j`.
2. Scale them by `1/sqrt(2)`.
3. Softmax over the three keys.
4. Compute the output as the weighted sum of values.

## Part B: causal mask by hand

Position 1 may not attend to key 2. Apply the mask **before** softmax.
Report the three weights and the output. State the invariant you used
to check the forbidden key.

## Part C: wrong-axis diagnosis

A 2×3 score matrix is softmaxed and the **columns** (keys) each sum to
1, but the **rows** (queries) do not. Which axis was reduced, and which
invariant failed?

## Part D: what a weight is not

State, in one or two sentences, why attention weights are not
automatically a causal explanation of intent.

Pass requires independent arithmetic, a mask invariant, an axis
diagnosis, and an oral defense. Leave all learner responses unfilled
in the repository.
