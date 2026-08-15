# M06 → V01 Structured Data Workbench

M06 adds the visual interrogation surface to the V01 Structured Data Workbench. Before any later pipeline or model consumes a table, the workbench must make the table's shape, missingness, distributions, suspicious values, group structure, target balance, and field timing inspectable.

## Contribution to V01

- a question-first analysis sequence rather than a gallery of chart types;
- reusable checks for row grain, missingness, IQR candidates, group rates, and class balance;
- a reasoning record that separates observation, inference, and limits;
- a decision-time field map that exposes possible target leakage;
- a visual-integrity check for scales, aggregation, binning, omissions, and denominators.

## Hand-off to M07

M07 can turn repeated data preparation into a reusable pipeline only after M06 has identified which fields are trustworthy and available at prediction time. The `post_case_priority` field is deliberately unsuitable for a new-ticket model because it is recorded after case closure; carrying it forward would automate leakage.

## V01 integration check

Given a fresh table, the learner can state the decision question, choose an evidence-bearing view, trace the data into the chart, identify what the display does and does not show, and name the next uncertainty to resolve before modeling.
