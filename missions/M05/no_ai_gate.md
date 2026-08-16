# M05 No-AI Gate — Predict Shapes and Results by Hand

Complete this gate without an AI assistant or AI-generated code. Use paper or a blank text file for predictions.

## Part A — Manual shape prediction

Given:

```text
quantities shape: (3, 4)
prices shape: (4,)
discount rates shape: (3,)
```

Before implementation, write:

1. the result shape of `quantities * prices` and why;
2. why `line_values * discount_rates` fails;
3. the reshape/indexing operation that makes discount rates broadcast per order;
4. the result shapes of `sum(axis=0)` and `sum(axis=1)`.

## Part B — Manual result prediction

For:

```text
quantities = [[2, 1, 0],
              [0, 3, 2]]
prices = [10.0, 4.0, 1.5]
```

Calculate by hand:

- the complete elementwise product;
- each order subtotal;
- each product's total revenue across orders;
- the boolean mask for order subtotals greater than or equal to 20.

Only after writing the prediction, implement it with NumPy and compare every element.

## Part C — Fresh implementation

Write a fresh function that accepts quantities shaped `(orders, products)`, prices shaped `(products,)`, and tax rates shaped `(orders, 1)`. Return taxed line values and per-order totals. Include shape validation, at least one slice, one boolean selection, and one axis reduction.

## Pass standard

Pass only when the written predictions precede implementation, every shape is justified from axis meaning, manual and computed results match, and the fresh function works on a second input with different numbers of orders and products.
