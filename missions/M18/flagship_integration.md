# M18 → V04 Mathematical Instrumentation Layer

M18 adds a statistical decision instrument to V04. Upstream probability concepts
from M17 become executable checks on observed model or product differences:

`observations → descriptive distribution → estimand → sampling uncertainty → effect size → decision`

The instrument exposes, rather than hides, the primary metric, unit of analysis,
sample size, comparison family, point estimate, interval, resampling seed, test
assumptions, and causal limits. Those fields let later flagship model comparisons
distinguish a repeatable improvement from metric noise or analysis flexibility.

The mission does not promote every numerical difference into a release decision.
Its handoff to V04 is a reviewable inference policy plus reproducible simulation
code that downstream components can reuse when comparing training runs, evaluation
slices, or operational interventions.
