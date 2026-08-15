# V04 integration — Mathematical Instrumentation Layer

M19 adds inspectable gradient instrumentation to V04. The flagship should be able to display one training step as:

`parameters → predictions → loss → gradient → updated parameters`

The mission supplies deterministic reference values for a scalar weight and then a `(weight, bias)` pair. Those values can seed a future gradient-check panel without introducing a training framework or network dependency.

The integration boundary is deliberate: M19 explains what a gradient update does and how to verify it. Learning-rate dynamics and broader optimizer comparisons belong to M20.
