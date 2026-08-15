# No-AI gate — uncertainty from fresh counts

Complete this gate without AI-generated code or prose. Work from the fresh scenario below rather than copying notebook outputs.

An operations model scores 200 cases:

- 40 cases receive a probability of 0.50; 18 of them later contain the event.
- 160 cases receive a probability of 0.10; 8 of them later contain the event.
- Acting on a case costs 8 units.
- Missing an event costs 50 units.

Tasks:

1. Define the event and reference population in words.
2. Predict the overall base rate before calculating it.
3. Calculate the overall event frequency.
4. Calculate the event frequency conditional on each score group.
5. Calculate the probability of being in the 0.50 group given that the event occurred.
6. Test whether membership in the 0.50 group and the event are independent using marginal and joint rates.
7. Compare predicted probability with observed frequency for both groups and explain calibration without calling any individual prediction “wrong.”
8. Calculate expected loss for acting and not acting in each group; choose an action for each group and state the consequence assumptions.
9. Change one cost assumption and predict which action, if any, changes before recalculating.
10. Write a short guardrail that would prevent base-rate neglect in a review.
11. Complete the threshold ADR prompt in `adr_prompt.md`.

Passing requires calculations, named denominators, consequence-aware reasoning, a prediction-before-change record, and a plain-language explanation. A formula list without interpretation does not pass.
