# No-AI transfer gate

Complete this gate without AI-generated code.

You receive a fresh local table with:

- two numerical columns containing missing values;
- two categorical columns, including a category seen only at inference;
- a binary target;
- one identifier and one post-outcome field that must not become features.

Build a CPU-only pipeline that:

1. declares an explicit pre-outcome feature allow-list;
2. splits raw rows before any learned preprocessing;
3. imputes and scales numerical features;
4. imputes and one-hot encodes categorical features;
5. cross-validates the entire pipeline on training rows;
6. fits once on all training rows;
7. predicts raw inference rows with the unseen category;
8. serializes and reloads one fitted artifact;
9. proves transformed matrices and predictions are identical after reload;
10. explains which calls learn state and which only apply it.

Passing requires runnable code, the leakage argument, a prediction recorded
before the unseen-category test, and evidence from a fresh dataset. Copying the
M07 fixture or its fitted artifact does not demonstrate transfer.
