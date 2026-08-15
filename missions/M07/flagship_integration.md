# V01 — Structured Data Workbench integration

M07 supplies V01's reusable raw-data-to-prediction boundary.

The workbench can reuse these assets and invariants:

- an explicit feature allow-list separates identity and target fields from
  pre-outcome model inputs;
- numerical and categorical treatments are selected by column role;
- learned preprocessing and model state are fitted within the same split or
  cross-validation boundary;
- raw training and inference rows enter the same public interface;
- one serialized artifact contains the exact preprocessing and estimator state;
- equivalence checks make reload drift visible;
- controlled failure evidence demonstrates why duplicated preprocessing code
  is an operational risk.

V01 may replace the fixture, feature contract or estimator, but it must retain
these boundaries. M07 completion is evidence that the learner can turn a
notebook sequence into a reviewable, testable component rather than a one-off
model fit.
