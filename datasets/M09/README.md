# M09 synthetic learner-disengagement fixture

`learner_disengagement.csv` contains 180 deterministic, synthetic rows for the M09 binary-classification lab. It contains no real learner or personal data.

The target `disengaged_next_30_days` is binary:

- `1`: the synthetic learner disengages in the following 30 days;
- `0`: the synthetic learner does not.

Features are `account_age_days`, `weekly_sessions`, `overdue_tasks`, `assessment_score`, and `help_requests`. The label generator intentionally creates a minority positive class and injects noise so the exercise exposes baseline, error, probability, and threshold reasoning rather than perfect separation.

Run `python datasets/M09/generate_dataset.py` from any directory to reproduce the committed CSV byte-for-byte. The generator requires only the Python standard library and a fixed random seed. This fixture is for education only and must not be interpreted as a validated model of real learner behavior.
