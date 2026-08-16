# M16 shape fixture

`shape_points.csv` is a deterministic, hand-authored seven-vertex arrow polygon used to make two-dimensional matrix transformations visible.

Columns:

- `point_id`: stable landmark name;
- `vertex_order`: polygon traversal order;
- `x`, `y`: Cartesian coordinates;
- `landmark`: whether the point is explicitly used for prediction checks.

The asymmetric shape is intentional: rotations, shears, orientation mistakes and reversed composition order are easier to distinguish than they would be with a circle or centered square. The notebook closes the polygon when plotting; the CSV does not duplicate the first row.

No download, API, secret or generated learner evidence is involved.
