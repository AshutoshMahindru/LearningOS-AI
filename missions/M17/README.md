# M17 — Quantify Uncertainty

## Mission

Treat probability as an instrument for reasoning about model outputs, not as a list of formulas. Start with a small set of predictions and outcomes, make a prediction about what the numbers mean, run the uncertainty workbench, and only then name the probability concepts that explain the observations.

The mission loop is:

1. inspect model probabilities and outcomes;
2. predict what repeated outcomes should look like;
3. simulate or aggregate counts;
4. compare prediction with observation;
5. name the event, reference group, and condition;
6. diagnose a violated assumption;
7. choose an action using consequences as well as probability;
8. transfer the reasoning to fresh counts without AI assistance.

## What the lab covers

- events and complements;
- empirical frequency across repeated trials;
- conditional probability and the direction of a condition;
- base rates and posterior probability;
- independence versus dependence;
- expected loss for a simple action decision;
- model probabilities as graded outputs rather than guarantees;
- threshold trade-offs;
- calibration as agreement between predicted probability and observed frequency.

The source dataset is a deterministic, synthetic model-output fixture. Three probability groups are deliberately calibrated and one deployment-shift group is deliberately miscalibrated. The notebook also uses seeded, CPU-only simulations and requires no network, secret, or paid API.

## Controlled failure

The failure treats a 90% sensitive screening test as if a positive result implied a 90% chance of the condition. The lab exposes the neglected 1% base rate and the false-positive population, then repairs the reasoning with a count table. See `controlled_failure.md` for the diagnostic contract.

## Prediction discipline

Every experiment begins with a **Prediction checkpoint**. Write the prediction before running the next cell. A correct post-hoc story does not substitute for a recorded prediction.

## V04 connection

M17 adds the probability instrument to the V04 Mathematical Instrumentation Layer. It lets the Operations Intelligence System interpret scores, compare cohorts, reason about uncertain alerts, and make consequence-aware threshold decisions without pretending that a model probability is certainty.

## Completion evidence

The required learner-created evidence is defined in `evidence_contract.yaml`. No learner response, score, experiment result, or completed ADR is pre-populated in this package.
