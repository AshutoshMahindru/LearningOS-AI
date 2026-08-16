# M05 Code Reading — Read the Shapes Before the Syntax

Read the notebook's pricing computation twice: first as business logic, then as data movement.

## Pass 1 — Trace the Python loop

Find `python_order_totals(...)` and follow one order.

1. Which loop iterates over orders?
2. Which loop combines quantities with prices?
3. When is the discount applied?
4. Which value determines shipping?
5. What is appended to the result list?

Write the operation count that grows with the number of orders and products.

## Pass 2 — Read the vectorized shapes

Find `vectorized_order_components(...)`. For each named array, record shape, dtype, and business meaning:

- `units_array`;
- `prices_array`;
- `discounts_array`;
- `line_values`;
- `discount_columns`;
- `discounted_lines`;
- `discounted_subtotals`;
- `shipping`;
- `totals`.

Annotate the two broadcasts. State which input dimension is logically shared across rows and which is shared across columns.

## Pass 3 — Trace both axes

For `line_values` with shape `(orders, products)`:

- explain why `sum(axis=1)` returns one subtotal per order;
- explain why `sum(axis=0)` returns one revenue value per product;
- state which axis is removed in each case.

Do not describe `axis=0` only as “column-wise.” Tie it to the actual dimension removed and the output meaning.

## Pass 4 — Locate representation changes

Find where integer quantities and floating-point prices become floating-point line values. Explain why a homogeneous dtype is useful and why dtype conversion can matter for correctness, memory use, and speed.

## Pass 5 — Inspect the benchmark honestly

Locate `best_seconds(...)` and the correctness comparison. Explain:

- why the input data are created outside the timed functions;
- why multiple repetitions use the minimum observed duration;
- why the notebook does not require a fixed speedup;
- why timing alone cannot establish that the vectorized code is correct.

## Transfer prompt

An unfamiliar fulfillment system stores `(warehouses, products)` inventory, a `(products,)` reorder threshold, and a `(warehouses, 1)` safety multiplier. Without running code, predict the shapes of the two broadcasts and the result of aggregating shortages with `axis=0` versus `axis=1`.
