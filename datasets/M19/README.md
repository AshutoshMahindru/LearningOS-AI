# M19 deterministic fixture

`tiny_linear.csv` contains five centered integer inputs and two targets:

- `y_one_parameter = 3x` supports the scalar-weight experiments.
- `y_with_bias = 3x + 2` supports the later weight-and-bias transfer.

The fixture is synthetic, checked into the repository, and intentionally tiny so every prediction, loss, derivative, and update can be reproduced by hand. It has no external license or runtime network dependency.
