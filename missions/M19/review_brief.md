# M19 engineering review brief

## Review target

Confirm that the gradient mechanism is mathematically correct, observable, deterministic, and taught in the required order.

## Consequential invariants

- Manual one-parameter changes appear before any derivative.
- Central finite difference and analytic derivative agree on the fixture.
- A correct update lowers loss from the reference start.
- The wrong-sign controlled failure raises loss with only the update operator changed.
- Multiple parameters appear only after repeated scalar updates.
- The source notebook is clean, offline, secret-free, CPU-only, and Restart + Run All succeeds.

## Review evidence

Inspect `gradient_core.py`, the executed notebook trace, the controlled-failure comparison, and the M19 unittest/pytest results. Reviewers should reject a result that merely reaches the expected parameters without preserving the trace or prediction-before-action gates.

## Deferred boundary

Optimizer families, momentum, schedules, and systematic learning-rate experiments are deferred to M20.
