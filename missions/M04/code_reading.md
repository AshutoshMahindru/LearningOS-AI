# Code reading

Read `missions/M04/cleaning.py` before changing it.

Trace one row with a currency-formatted price and one row with a conflicting ID.
For each row identify:

1. the raw value and source row;
2. the normalization or parsing function called;
3. the canonical value returned;
4. every issue code added;
5. every decision added;
6. whether the issue is blocking;
7. whether the row enters `analysis_ready`;
8. which raw columns preserve the original evidence.

Before executing, predict the exact `observed_defects`, `decision_log`,
`uncertainty` and `analysis_ready` values for both rows. Then run the pipeline
and identify the first line where any mental trace diverges.

Finally, propose one new alias or constraint. State a failing example and the
invariant it protects before modifying code.
