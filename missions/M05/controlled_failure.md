# M05 Controlled Failure — The Wrong Shape Looks Almost Right

## Failure A — Discount rates on the wrong axis

The working data have these shapes:

```text
line_values: (3, 4)       # orders, products
discount_rates: (3,)      # one rate per order
```

The expression below fails:

```python
line_values * (1.0 - discount_rates)
```

Broadcasting compares trailing dimensions. It tries to align `4` products with `3` discounts, and neither dimension is equal to one. Capture the resulting `ValueError`; do not hide it with an unrelated reshape.

### Repair

Make the intended order axis explicit:

```python
discount_columns = discount_rates[:, np.newaxis]  # (3, 1)
discounted_lines = line_values * (1.0 - discount_columns)
```

The singleton product dimension can expand across four product columns while the three-order dimension stays aligned.

## Failure B — A valid reduction with the wrong meaning

`line_values.sum(axis=0)` is valid but returns one value per product. The pricing rule needs one value per order, so treating that result as order totals is a semantic axis error even when no library exception occurs.

Detect the problem with an explicit expected-shape assertion, inspect the observed `(products,)` shape, then repair it with `sum(axis=1)` and verify the expected `(orders,)` shape.

## Diagnosis record

For each failure, record:

1. the prediction made before execution;
2. operand shapes and axis meanings;
3. the observed exception or incorrect output shape;
4. the smallest repair;
5. a correctness check after the repair;
6. one upstream validation that would prevent recurrence.

The learning target is not “add `newaxis`” or “switch 0 to 1.” It is to derive the repair from the intended dimensions and business meaning.
