# M06 Controlled Failure — The Truncated Scale

## Failure

The notebook renders the same mean customer tenure by support channel in two bar charts. The first chart truncates the vertical scale close to the observed means. The second restores a zero baseline and covers the sample's full 0–50 month range.

The underlying records, grouping, aggregation, ordering, and computed means are identical. Only the scale changes.

## Predict before running

Write which view you expect to make the channel differences look larger and why. State what you will compare to verify that the data did not change.

## Diagnostic task

1. Record the visible impression produced by the truncated view.
2. Inspect the plotted values and both `set_ylim` calls.
3. Verify that both charts receive the same group means.
4. Explain what a reader could wrongly infer from the first view.
5. Repair the presentation with an honest scale or a more suitable non-bar encoding.
6. Preserve exact values and denominators near the chart.
7. State what neither view can establish about causation or future performance.

## Prevention

The review checklist must ask whether axis bounds, aggregation, binning, omitted groups, and denominators change the substantive impression. A chart is not trustworthy merely because its code runs or its labels are accurate.
