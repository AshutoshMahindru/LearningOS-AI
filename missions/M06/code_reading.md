# M06 Code Reading — Trace Data Into Marks

Read the notebook's plotting cells before running them. For one chart at a time, trace:

1. the source dataframe and any rows excluded;
2. the columns selected;
3. each transformation, grouping, or aggregation;
4. the values encoded on each axis, colour, or mark;
5. the denominator behind every rate;
6. the axis limits and bin boundaries;
7. any ordering imposed by code;
8. what information is lost before rendering.

For `iqr_outliers`, trace a small sequence manually. Identify the median, quartiles, IQR, fences, and rows returned. An IQR flag is a review signal, not permission to delete a record.

For the group-rate chart, explain why `size` and `sum` are computed alongside `mean`. State how the interpretation would change if one group had only two tickets.

For the controlled failure, identify the single plotting choice that changes the impression while the underlying means remain identical. Explain why code that executes successfully can still produce a misleading artifact.

Transfer question: when reading an unfamiliar charting function, what evidence would let you reconstruct the exact data that reached the plotting call?
