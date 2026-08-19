"""Black-box neural-network training harness for Mission M21.

M21 deliberately treats the estimator as a whole system. The public contract exposes
inputs, declared training configuration, learning curves, validation evidence, held-out
metrics, and error counts. Weight matrices, neuron activations, manual forward passes,
and backpropagation mechanics are deferred to M22-M24.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import warnings


DEFAULT_SPLIT_SEED = 2101
DEFAULT_MODEL_SEED = 2101
DEFAULT_TEST_SIZE = 0.25
DEFAULT_HIDDEN_UNITS = 64
DEFAULT_MAX_ITER = 60
DEFAULT_LEARNING_RATE = 0.001


@dataclass(frozen=True)
class TrainingRun:
    """Observable evidence from one end-to-end black-box training run."""

    split_seed: int
    model_seed: int
    hidden_units: int
    max_iter: int
    learning_rate_init: float
    scale_inputs: bool
    shuffled_labels: bool
    label_seed: int | None
    train_size: int
    test_size: int
    feature_count: int
    class_count: int
    majority_baseline_accuracy: float
    train_accuracy: float  # scored against the labels passed to fit
    test_accuracy: float
    macro_f1: float
    iterations: int
    final_loss: float
    best_validation_score: float
    loss_curve: tuple[float, ...]
    validation_scores: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    classes: tuple[int, ...]


def _require_dependencies():
    try:
        import numpy as np
        from sklearn.datasets import load_digits
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError(
            "M21 requires numpy and scikit-learn; install requirements/m21.txt"
        ) from exc
    return {
        "np": np,
        "load_digits": load_digits,
        "ConvergenceWarning": ConvergenceWarning,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "train_test_split": train_test_split,
        "MLPClassifier": MLPClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def _validate_config(
    *,
    test_size: float,
    hidden_units: int,
    max_iter: int,
    learning_rate_init: float,
) -> None:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    if hidden_units < 1:
        raise ValueError("hidden_units must be at least one")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if not math.isfinite(learning_rate_init) or learning_rate_init <= 0:
        raise ValueError("learning_rate_init must be positive and finite")


def dataset_summary() -> dict[str, int | float]:
    """Return stable metadata for the bundled offline digits fixture."""

    d = _require_dependencies()
    X, y = d["load_digits"](return_X_y=True)
    counts = Counter(int(value) for value in y)
    return {
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "classes": int(len(counts)),
        "min_class_count": int(min(counts.values())),
        "max_class_count": int(max(counts.values())),
        "pixel_min": float(X.min()),
        "pixel_max": float(X.max()),
    }


def _split_digits(
    d: dict,
    *,
    test_size: float,
    split_seed: int,
):
    X, y = d["load_digits"](return_X_y=True)
    X_train, X_test, y_train, y_test = d["train_test_split"](
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
        stratify=y,
    )
    return X, y, X_train, X_test, y_train, y_test


def inspect_holdout_split(
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict[str, object]:
    """Return split sizes and majority baseline without fitting a network."""

    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    d = _require_dependencies()
    X, _y, X_train, X_test, y_train, y_test = _split_digits(
        d, test_size=test_size, split_seed=split_seed
    )
    counts = Counter(int(value) for value in y_test)
    majority_class, majority_count = max(counts.items(), key=lambda item: item[1])
    return {
        "split_seed": int(split_seed),
        "test_fraction": float(test_size),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "feature_count": int(X.shape[1]),
        "train_feature_count": int(X_train.shape[1]),
        "test_feature_count": int(X_test.shape[1]),
        "class_count": int(len(counts)),
        "test_class_counts": dict(sorted(counts.items())),
        "majority_class": int(majority_class),
        "majority_baseline_accuracy": float(majority_count / len(y_test)),
        "preprocessing": "StandardScaler fitted on training data only through a Pipeline",
        "early_stopping_split": "validation_fraction=0.15 taken from the training split only",
    }


def train_black_box(
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    model_seed: int = DEFAULT_MODEL_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    hidden_units: int = DEFAULT_HIDDEN_UNITS,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate_init: float = DEFAULT_LEARNING_RATE,
    scale_inputs: bool = True,
    shuffle_labels: bool = False,
    label_seed: int = 2121,
) -> TrainingRun:
    """Train and evaluate one declared MLP configuration without opening internals."""

    _validate_config(
        test_size=test_size,
        hidden_units=hidden_units,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
    )
    d = _require_dependencies()
    np = d["np"]
    X, _y, X_train, X_test, y_train, y_test = _split_digits(
        d, test_size=test_size, split_seed=split_seed
    )

    fit_labels = y_train
    effective_label_seed: int | None = None
    if shuffle_labels:
        effective_label_seed = int(label_seed)
        fit_labels = np.random.default_rng(effective_label_seed).permutation(y_train)

    scaler = d["StandardScaler"]() if scale_inputs else "passthrough"
    network = d["MLPClassifier"](
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate_init,
        batch_size=64,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=model_seed,
    )
    pipeline = d["Pipeline"]([("scale", scaler), ("network", network)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", d["ConvergenceWarning"])
        pipeline.fit(X_train, fit_labels)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    classes = tuple(int(value) for value in sorted(set(int(v) for v in y_test)))
    matrix = d["confusion_matrix"](y_test, test_predictions, labels=classes)
    counts = Counter(int(value) for value in y_test)
    baseline = max(counts.values()) / len(y_test)

    validation_scores = tuple(float(v) for v in (network.validation_scores_ or ()))
    best_validation = max(validation_scores) if validation_scores else float("nan")
    return TrainingRun(
        split_seed=int(split_seed),
        model_seed=int(model_seed),
        hidden_units=int(hidden_units),
        max_iter=int(max_iter),
        learning_rate_init=float(learning_rate_init),
        scale_inputs=bool(scale_inputs),
        shuffled_labels=bool(shuffle_labels),
        label_seed=effective_label_seed,
        train_size=int(len(y_train)),
        test_size=int(len(y_test)),
        feature_count=int(X.shape[1]),
        class_count=int(len(classes)),
        majority_baseline_accuracy=float(baseline),
        train_accuracy=float(d["accuracy_score"](fit_labels, train_predictions)),
        test_accuracy=float(d["accuracy_score"](y_test, test_predictions)),
        macro_f1=float(d["f1_score"](y_test, test_predictions, average="macro")),
        iterations=int(network.n_iter_),
        final_loss=float(network.loss_curve_[-1]),
        best_validation_score=float(best_validation),
        loss_curve=tuple(float(value) for value in network.loss_curve_),
        validation_scores=validation_scores,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        classes=classes,
    )


def most_confused_pair(run: TrainingRun) -> tuple[int, int, int]:
    """Return true class, predicted class, and largest off-diagonal error count."""

    best = (-1, -1, -1)
    for row_index, row in enumerate(run.confusion_matrix):
        for column_index, count in enumerate(row):
            if row_index == column_index:
                continue
            if count > best[2]:
                best = (run.classes[row_index], run.classes[column_index], count)
    return best


def compact_report(run: TrainingRun) -> dict[str, float | int | bool | None]:
    """Return the whole-system evidence used in the learner-facing comparison table."""

    return {
        "hidden_units": run.hidden_units,
        "max_iter": run.max_iter,
        "learning_rate_init": run.learning_rate_init,
        "model_seed": run.model_seed,
        "shuffled_labels": run.shuffled_labels,
        "scale_inputs": run.scale_inputs,
        "iterations": run.iterations,
        "majority_baseline_accuracy": round(run.majority_baseline_accuracy, 4),
        "train_accuracy": round(run.train_accuracy, 4),
        "test_accuracy": round(run.test_accuracy, 4),
        "macro_f1": round(run.macro_f1, 4),
        "first_loss": round(run.loss_curve[0], 4) if run.loss_curve else None,
        "final_loss": round(run.final_loss, 4),
        "n_loss_curve_points": len(run.loss_curve),
        "n_validation_scores": len(run.validation_scores),
        "best_validation_score": round(run.best_validation_score, 4),
    }


def print_run_evidence(run: TrainingRun, *, label: str = "run") -> None:
    """Print baseline, traces, metrics, and confusion matrix without opening internals."""

    print(f"=== {label} ===")
    print("majority_baseline_accuracy", round(run.majority_baseline_accuracy, 4))
    print("train_accuracy (vs labels used at fit)", round(run.train_accuracy, 4))
    if run.shuffled_labels:
        print("note: shuffled_labels=True; train_accuracy is not vs the original y_train")
    print("compact_report", compact_report(run))
    print("loss_curve", [round(value, 4) for value in run.loss_curve])
    print("validation_scores", [round(value, 4) for value in run.validation_scores])
    print("confusion_matrix classes", run.classes)
    for row in run.confusion_matrix:
        print(row)
    print("most_confused_pair", most_confused_pair(run))
