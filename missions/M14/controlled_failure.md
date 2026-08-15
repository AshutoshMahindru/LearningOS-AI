# M14 Controlled Failure — The Persuasive Segmentation

The lab seeds four related traps. Diagnose each one before reading any review
notes or changing the pipeline.

## Faulty claim

> “K-means found the three real learner types. The plot has clean colors, the
> silhouette is positive, and the stakeholder asked for three segments.”

## Traps to investigate

### Scale

The raw feature matrix mixes proportions, small counts, minutes, and a much
larger activity-event count. Fit K-means on raw values and determine which
feature supplies most of the squared-distance budget. A clean partition along
that feature is not automatically a meaningful behavioral structure.

### Arbitrary k

Compare the requested `k=3` with `k=2..6`. Look for a falling inertia curve,
silhouette behavior, assignment stability, and small-cluster fragmentation.
None of these diagnostics is a truth oracle; disagreements must be surfaced.

### Outlier

Inject one extreme instrumentation value, refit both the scaler and K-means,
and compare assignments for the unchanged rows. StandardScaler and squared
Euclidean objectives can both respond strongly to an extreme observation.

### Visualization

Inspect a two-feature view and a two-component PCA projection. Record how much
variance the PCA view retains and compare the picture with diagnostics computed
in the full selected feature space. Color and separation in a projection do not
prove natural categories.

## Diagnosis contract

A defensible diagnosis identifies the mechanism of each failure, names the
affected decision or claim, quantifies at least one effect, and proposes a next
check. It must explicitly distinguish cluster assignments from externally
validated classes or learner identities.
