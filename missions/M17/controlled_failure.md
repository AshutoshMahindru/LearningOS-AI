# Controlled failure — base-rate neglect

## Plausible but incorrect claim

> The test is 90% sensitive, so a person who tests positive has a 90% probability of having the condition.

The claim reverses the condition. Sensitivity is the probability of a positive result **given** the condition. The decision question is the probability of the condition **given** a positive result. A low base rate and false positives can make those values very different.

## One seeded root cause

The seeded root cause is **base-rate neglect caused by using sensitivity as the posterior probability**. Do not introduce a second bug or repair the result by changing the simulated population.

## Required diagnostic sequence

1. state the two conditional probabilities in words before using symbols;
2. record the 1% base rate, 90% sensitivity, and 91% specificity;
3. predict the result for a population of 10,000 before constructing counts;
4. enumerate affected and unaffected people;
5. enumerate true positives, false negatives, false positives, and true negatives;
6. identify the denominator for “condition given positive”;
7. compare the failed 90% claim with the count-derived result;
8. vary only the base rate and verify that sensitivity remains fixed while the posterior changes;
9. state the operational harm the failed claim could cause;
10. add a guardrail that requires the reference population and denominator to be named.

## Evidence bar

A corrected number without the complete count table and condition direction does not pass. Evidence must include the original claim, a prediction, the counts, the repaired reasoning, and a counterfactual base-rate check.
