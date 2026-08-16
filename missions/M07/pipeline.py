"""Reusable, deterministic preprocessing and modeling for mission M07.

The module keeps learned preprocessing state inside the scikit-learn Pipeline.
Callers pass raw feature frames to ``fit`` and ``predict``; they never need to
duplicate training transformations at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


TARGET_COLUMN = "renewed"
IDENTIFIER_COLUMN = "customer_id"
NUMERIC_FEATURES = ("tenure_months", "monthly_spend", "support_tickets")
CATEGORICAL_FEATURES = ("plan", "region", "signup_channel")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "datasets" / "M07" / "customer_renewals.csv"
)

NumericStrategy = Literal["mean", "median"]
ScalerChoice = Literal["standard", "robust", "passthrough"]


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load the local fixture and validate its training contract."""

    frame = pd.read_csv(path)
    required = {IDENTIFIER_COLUMN, TARGET_COLUMN, *MODEL_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    if frame[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("customer_id values must be unique")
    target_values = set(frame[TARGET_COLUMN].dropna().unique())
    if target_values != {0, 1}:
        raise ValueError("renewed must contain both binary classes 0 and 1")
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the declared pre-outcome features and target.

    The explicit allow-list is a leakage boundary: identifiers, the target and
    any undeclared post-outcome columns are excluded even if present.
    """

    missing = set((*MODEL_FEATURES, TARGET_COLUMN)).difference(frame.columns)
    if missing:
        raise ValueError(f"training frame is missing columns: {sorted(missing)}")
    features = frame.loc[:, MODEL_FEATURES].copy()
    target = frame.loc[:, TARGET_COLUMN].astype("int64").copy()
    return features, target


def train_test_frames(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw rows before fitting any preprocessing state."""

    features, target = split_features_target(frame)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_pipeline(
    *,
    numeric_strategy: NumericStrategy = "median",
    scaler: ScalerChoice = "standard",
    random_state: int = 17,
) -> Pipeline:
    """Construct an unfitted preprocessing-and-classification pipeline."""

    if numeric_strategy not in {"mean", "median"}:
        raise ValueError("numeric_strategy must be 'mean' or 'median'")
    if scaler not in {"standard", "robust", "passthrough"}:
        raise ValueError("scaler must be 'standard', 'robust' or 'passthrough'")

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy=numeric_strategy)),
    ]
    if scaler == "standard":
        numeric_steps.append(("scale", StandardScaler()))
    elif scaler == "robust":
        numeric_steps.append(("scale", RobustScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(max_iter=1_000, random_state=random_state)
    return Pipeline([("preprocess", preprocessing), ("model", classifier)])


def transform_features(pipeline: Pipeline, raw_features: pd.DataFrame) -> np.ndarray:
    """Apply the fitted training transformations without refitting them."""

    transformed = pipeline.named_steps["preprocess"].transform(raw_features)
    return np.asarray(transformed)


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Serialize the fitted pipeline as one artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination


def load_pipeline(path: str | Path) -> Pipeline:
    """Reload a serialized M07 pipeline."""

    loaded = joblib.load(path)
    if not isinstance(loaded, Pipeline):
        raise TypeError("serialized artifact is not a scikit-learn Pipeline")
    return loaded
