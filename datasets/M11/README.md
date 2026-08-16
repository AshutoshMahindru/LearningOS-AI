# M11 dataset fixture

`learner_readiness.csv` is a deterministic **synthetic** classification fixture authored for M11. It contains 96 invented rows and no personal or real learner data.

Columns:

- `learner_id`: synthetic row identifier, excluded from model features;
- `study_hours_week`: invented weekly study hours;
- `practice_accuracy`: invented fraction of practice items answered correctly;
- `attendance_pct`: invented attendance percentage;
- `sleep_hours`: invented nightly sleep estimate;
- `ready_for_assessment`: synthetic binary target.

The labels come from a simple multi-feature rule with seven deterministic flips. Those flips create realistic-enough noise for an unconstrained tree to overfit. The generation is intentionally transparent so the fixture tests tree mechanics rather than claims about a real population.

Use restrictions:

- do not infer anything about real learners;
- do not use the fixture for ranking, access, diagnosis, or intervention;
- do not interpret a split or feature importance as a causal effect;
- keep `learner_id` out of the feature matrix;
- preserve the fixed split when comparing mission experiments.
