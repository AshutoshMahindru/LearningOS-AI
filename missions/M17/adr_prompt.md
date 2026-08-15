# ADR prompt — choose an operational threshold

Complete this decision record after the expected-loss and calibration experiments. Do not select a threshold from accuracy alone.

## Decision

Which threshold or threshold policy will the Operations Intelligence System use for this event?

## Context

State the event, reference population, model-output cohorts, observed calibration evidence, action cost, missed-event cost, and any deployment-shift concern.

## Alternatives considered

Consider at least:

1. keep one global threshold;
2. change the global threshold;
3. recalibrate before changing the threshold;
4. use a cohort-specific policy with an explicit fairness and operations review;
5. defer automated action and collect more outcomes.

## Evidence

Include reliability groups, confusion counts, expected loss, base-rate assumptions, sample-size limitations, and a prediction of what the chosen policy will change.

## Trade-offs

Explain false-positive burden, missed-event harm, operational capacity, cohort effects, and the risk of acting on unstable calibration evidence.

## Revisit conditions

Name measurable triggers such as a base-rate shift, calibration gap, cost change, cohort drift, or minimum new-outcome count.

## Status

Use proposed, accepted, superseded, or rejected. The mission package intentionally leaves the decision unanswered.
