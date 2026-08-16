# M20 — Understand Optimization Experimentally

M19 made the gradient update visible. M20 keeps that gradient mechanism fixed and
changes **optimization behavior**: learning rate, exact versus stochastic gradients,
momentum, and Adam. Learners predict a loss curve before every run, execute controlled
experiments from the same objective and initialization, and explain the observed
convergence, stagnation, oscillation, or divergence from the update trace.

The reference objective is a two-parameter anisotropic quadratic. Its unequal
curvatures make a single learning rate behave differently by coordinate while keeping
every gradient and update easy to inspect. SGD adds a committed, zero-mean component
gradient fixture and a fixed seed; it does not change the aggregate objective.

The controlled failures are observable and recoverable: a very small learning rate
makes negligible progress, a too-large rate oscillates outward and diverges, and an
apparently successful final loss can hide momentum overshoot or Adam tuning cost.
No optimizer is presented as universally superior.

The package is deterministic, CPU-only, offline, paid-API-free, and secret-free.
Implementation status is not learner completion. Predictions, evidence, no-AI work,
scores, reviewer sign-off, and the optimizer-policy ADR remain intentionally unfilled.

