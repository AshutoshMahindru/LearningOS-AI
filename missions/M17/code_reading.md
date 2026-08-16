# Code reading — trace a reliability report

Read the notebook function `reliability_report` before running its cell.

Trace these boundaries:

1. the input row schema and data types;
2. the key used to create probability groups;
3. the accumulator state for count, probability sum, and outcome sum;
4. the denominator used for each observed rate;
5. the calculation of the calibration gap;
6. the ordering of returned groups;
7. the behavior for an empty input;
8. whether the function mutates the source rows.

Before execution, manually trace one group from `model_predictions.csv` and predict the returned count, mean probability, observed event rate, and gap.

Then execute the function and compare the first divergent value, if any. Explain whether a difference came from the data, grouping rule, numerator, denominator, or rounding.

Finally, make one targeted extension: add cohort-level grouping or a minimum-group-size warning without rewriting the original calculation.
