# Controlled failure

The notebook contains a version of the order-total calculation that produces a plausible number but is wrong.

The failure has **one seeded root cause**.

Do not begin by rewriting the function.

Required diagnostic sequence:

1. record the expected and observed totals;
2. reproduce the failure with the smallest useful input;
3. trace subtotal, discount and shipping intermediate values;
4. state one hypothesis;
5. design one test that could falsify that hypothesis;
6. identify the first incorrect intermediate value or operation;
7. make the smallest repair;
8. rerun the original and boundary cases as verification;
9. explain why the repair fixes the cause rather than merely the symptom.

Evidence must contain the trace and reasoning, not only the corrected code.
