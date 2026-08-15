# M13 — Learn from Neighbors

## Mission

Run a complete K-nearest-neighbors classifier, then explain every prediction as a vote among nearby training cases.

The learning loop is:

1. visualize the training cases;
2. place a query point and predict its class;
3. inspect the exact neighbors and distances used for the vote;
4. vary `k` and the distance metric;
5. compare raw and standardized feature spaces;
6. seed an incompatible-scale failure;
7. repair the distance design and explain the changed boundary.

## Data contract

`datasets/M13/knn_scale_cases.csv` is deterministic, balanced, synthetic classification data. `practice_hours` and `assessment_score` define a curved local pattern. `interface_event_count` is generated independently of the target and is intentionally much larger in numeric scale.

The weak high-scale feature is a teaching instrument, not a recommended production feature.

## Distance reasoning

For each experiment, use:

**prediction → run → observation → explanation → generalization → transfer**

An explanation must name which feature changed a distance, which cases entered or left the neighborhood, and how their votes changed the prediction. Accuracy alone is insufficient.

## Controlled failure

Add `interface_event_count` without scaling. Its numeric differences overwhelm the informative coordinates, so the model retrieves cases with similar instrumentation counts rather than similar learning behavior. Diagnose this using per-feature distance contributions and neighbor identities before repairing it.

## Source policy

`sklearn-guide` in `data/source_registry.json` is the authoritative just-in-time reference. The lab requires no runtime network access.

## V03 connection

M13 contributes an instance-based model and distance diagnostics to V03 Model Comparison & Diagnostics. Its artifact supports direct comparison with trees and ensembles from M11–M12 and prepares the learner to reason about clustering in M14.

## Completion evidence

Completion requires prediction logs, neighbor traces, `k` and metric comparisons, scaling evidence, failure diagnosis, a boundary explanation, code reading, a no-AI transfer, and an ADR covering the distance design.
