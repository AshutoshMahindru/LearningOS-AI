# Code reading

Read `missions/M07/pipeline.py` before running the notebook.

Trace these paths in order:

1. `load_dataset` validates the fixture contract.
2. `split_features_target` selects only declared pre-outcome features.
3. `train_test_frames` splits raw rows before any learned transformation.
4. `build_pipeline` creates two preprocessing branches and one estimator.
5. `fit` causes imputers, scaler, encoder and model to learn state.
6. `transform_features` reuses fitted preprocessing without updating it.
7. `save_pipeline` and `load_pipeline` preserve that state as one artifact.

Before execution, draw the raw column flow through the `ColumnTransformer`.
Identify which methods learn state and which only apply existing state. Predict
what happens when inference contains a category absent from training. Then run
the lab and compare the observed behavior with your trace.

Modification task: add a new pre-outcome numerical feature to a copy of the
fixture and list every contract location that must change. Do not make the
change until you can explain how omission at any one location could create
silent dropping or train/inference drift.
