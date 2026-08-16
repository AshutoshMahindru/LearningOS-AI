# V04 integration and closure — Mathematical Instrumentation Layer

## M19 → M20 boundary

M19 supplies the verified chain `parameters → loss → gradient → update`. M20 does
not re-teach derivative mechanics or change the analytic gradient to improve a run.
It instruments the next layer:

`gradient → optimizer state → applied update → parameter path → loss curve`

V04 can now expose learning rate, optimizer state, update magnitude, stochastic seed,
coordinate sign changes, and loss by step. A reviewer can distinguish a gradient bug
from an optimization-policy failure.

## V04 closure contract

M20 closes V04 only when a learner can predict, observe, and explain convergence,
stagnation, oscillation, and divergence; compare GD, SGD, momentum, and Adam under
declared controls; repair the seeded failures; pass the no-AI transfer gate; and defend
an unfilled-at-source optimizer-policy ADR in formal review.

Implementation completion does not satisfy that learner contract. Evidence and review
fields remain unpopulated until produced by the learner and accepted by a reviewer.

## M20 → M21 handoff

M21 may treat neural-network training as a black box only after inheriting M20's
monitored optimizer regime: recorded hyperparameters, loss/update traces, divergence
and rollback triggers, reproducibility seed policy, and a documented reason for the
chosen optimizer. Neural-network architecture internals remain outside M20.

