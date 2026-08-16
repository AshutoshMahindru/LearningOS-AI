# Code reading

Read the notebook's scaling-and-KNN pipeline before modifying it.

Trace these boundaries:

1. CSV row to feature matrix and target;
2. full dataset to train/test indices;
3. training features to fitted scaler statistics;
4. raw query to transformed query;
5. transformed query to distance calculation;
6. neighbor indices to target labels;
7. labels to majority vote;
8. prediction to evaluation metric.

Before executing, record:

- the shape and units at each boundary;
- which objects learn state during `fit`;
- why the scaler must learn only from training rows;
- where the same transformation is guaranteed at prediction time;
- how to retrieve neighbor identities from inside the pipeline;
- one failure path caused by passing raw coordinates to a model trained on scaled coordinates.

Then run the trace, compare it with your prediction and identify the first point of divergence.
