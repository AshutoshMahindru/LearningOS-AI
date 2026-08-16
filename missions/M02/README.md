# M02 — Run and Interrogate Your First ML System

M02 is a whole-first mission. You operate one complete supervised classification system before studying each mechanism in depth. The system uses a local copy of the Wine recognition dataset, a deterministic CPU-only workflow, and no network, paid API, or secret.

## System map

```text
raw CSV
  -> features (13 measurements) + target (wine class)
  -> stratified train/test split
  -> preprocessing + classifier fit on training rows only
  -> predictions for held-out rows
  -> accuracy, balanced accuracy, confusion matrix
  -> error, coefficient, and experiment interrogation
```

The important boundaries are behavioral rather than magical: `fit` learns from training examples, `predict` applies learned state to feature rows, and evaluation compares predictions with labels that the model did not receive during fitting.

## Run the mission

From the repository root, create an environment and install `requirements/m02.txt`. Open `labs/M02_first_ml_system.ipynb`, read each **Predict before running** prompt, record a prediction, then run the next code cell. A source-controlled notebook intentionally contains no execution output or learner response.

For maintainer validation:

```bash
python -m unittest tests.missions.test_m02 -v
python -m pytest tests/missions/test_m02.py -v
python -m jupyter nbconvert --to notebook --execute labs/M02_first_ml_system.ipynb --output M02_first_ml_system.executed.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=300
```

## Experimental discipline

For every experiment, write down:

1. the single variable you will change;
2. the direction you predict the result will move and why;
3. the constants you will preserve;
4. the observed metric and diagnostic evidence;
5. whether the result supports the prediction and what remains uncertain.

The notebook varies split, feature subset, model family, a selected hyperparameter, training-label integrity, and evaluation setup. These are probes, not a leaderboard. A metric difference can reflect sampling variation or evaluation design rather than a universally better model.

## Completion standard

Completion requires a fresh no-AI supervised run, a system map, a boundary explanation, an experiment table, diagnosis of both controlled failures, validation evidence, limitations, and a short explanation of how the work contributes to V00. Package implementation status is not learner completion.
