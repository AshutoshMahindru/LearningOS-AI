"""Transparent two-layer NumPy forward pass for Mission M23.

M23 reconstructs inference from array operations: affine maps, bias
broadcast, hidden activation, logits, and stable class-axis softmax.
Parameter gradients and framework autograd remain deferred to M24-M25.

The computational graph is the named sequence
x -> hidden_preactivation -> hidden_activation -> logits -> probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys


# First layer is the M22 row-batch dense fixture: X (2, 3), W1 (3, 2), b1 (2,)
REFERENCE_X = (
    (1.0, 0.0, -1.0),
    (0.5, 0.5, 0.5),
)
REFERENCE_W1 = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
)
REFERENCE_B1 = (0.0, 0.5)

# Second layer maps hidden units to three class logits: W2 (2, 3), b2 (3,)
REFERENCE_W2 = (
    (1.0, 0.0, -1.0),
    (0.0, 1.0, 0.5),
)
REFERENCE_B2 = (0.0, 0.0, 0.0)

# Hand-computed named intermediates for REFERENCE_X with hidden ReLU.
REFERENCE_HIDDEN_PREACTIVATION = (
    (0.0, -0.5),
    (1.0, 1.5),
)
REFERENCE_HIDDEN_ACTIVATION = (
    (0.0, 0.0),
    (1.0, 1.5),
)
REFERENCE_LOGITS = (
    (0.0, 0.0, 0.0),
    (1.0, 1.5, -0.25),
)

CLASS_AXIS = -1
BATCH_AXIS = 0
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 0.0
DEFAULT_DTYPE_NAME = "float64"
HIDDEN_ACTIVATION = "relu"
SUPPORTED_DEFECTS = (
    "none",
    "correct",
    "softmax_axis_batch",
    "omitted_hidden_activation",
)
GRAPH_NODES = (
    "x",
    "hidden_preactivation",
    "hidden_activation",
    "logits",
    "probabilities",
)
INTERMEDIATE_NAMES = GRAPH_NODES[1:]


def _softmax_tuple(logits: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(logits)
    shifted = tuple(math.exp(value - maximum) for value in logits)
    total = sum(shifted)
    return tuple(value / total for value in shifted)


REFERENCE_PROBABILITIES = (
    _softmax_tuple(REFERENCE_LOGITS[0]),
    _softmax_tuple(REFERENCE_LOGITS[1]),
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M23 requires numpy; install requirements/m23.txt") from exc
    return np


def _load_m22():
    """Load the trusted M22 single-layer reference without a package import."""

    name = "_learningos_m22_neuron_layer_core"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent.parent / "M22" / "neuron_layer_core.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load missions/M22/neuron_layer_core.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _as_1d(values) -> tuple[float, ...]:
    np = _require_numpy()
    array = np.asarray(values, dtype=float).reshape(-1)
    return tuple(float(value) for value in array)


def _as_2d(values):
    np = _require_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("expected a vector or a 2-D batch array")
    return array


def _require_finite(array, name: str) -> None:
    np = _require_numpy()
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")


def _nested_tuples(values) -> tuple:
    np = _require_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return (float(array),)
    if array.ndim == 1:
        return tuple(float(value) for value in array)
    return tuple(_nested_tuples(row) for row in array)


@dataclass(frozen=True)
class ForwardTrace:
    """Named intermediates for one two-layer classification forward pass."""

    x: object
    hidden_preactivation: object
    hidden_activation: object
    logits: object
    probabilities: object
    hidden_activation_name: str
    shapes: dict[str, tuple[int, ...]]
    defect: str = "none"

    def named(self) -> dict[str, object]:
        return {
            "hidden_preactivation": self.hidden_preactivation,
            "hidden_activation": self.hidden_activation,
            "logits": self.logits,
            "probabilities": self.probabilities,
        }


@dataclass(frozen=True)
class ScalarTrace:
    """Loop-level intermediates for one example; used to prove vectorized parity."""

    x: tuple[float, ...]
    hidden_preactivation: tuple[float, ...]
    hidden_activation: tuple[float, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    hidden_activation_name: str


def validate_stack_shapes(X, W1, b1, W2, b2) -> tuple[int, int, int, int]:
    """Return (batch, n_in, n_hidden, n_classes) or raise a shape contract error."""

    batch = _as_2d(X)
    weights1 = _as_2d(W1)
    bias1 = _as_1d(b1)
    weights2 = _as_2d(W2)
    bias2 = _as_1d(b2)
    n_batch, n_in = batch.shape
    n_in_w, n_hidden = weights1.shape
    n_hidden_w, n_classes = weights2.shape
    if n_in != n_in_w:
        raise ValueError(
            f"X has {n_in} features but W1 has {n_in_w} rows; expected W1 shape (n_in, n_hidden)"
        )
    if len(bias1) != n_hidden:
        raise ValueError(
            f"b1 has {len(bias1)} entries but W1 has {n_hidden} columns; expected b1 shape (n_hidden,)"
        )
    if n_hidden != n_hidden_w:
        raise ValueError(
            f"hidden width {n_hidden} does not match W2 rows {n_hidden_w}; expected W2 shape (n_hidden, n_classes)"
        )
    if len(bias2) != n_classes:
        raise ValueError(
            f"b2 has {len(bias2)} entries but W2 has {n_classes} columns; expected b2 shape (n_classes,)"
        )
    return n_batch, n_in, n_hidden, n_classes


def stacked_shapes(X, W1, b1, W2, b2) -> dict[str, tuple[int, ...]]:
    n_batch, n_in, n_hidden, n_classes = validate_stack_shapes(X, W1, b1, W2, b2)
    return {
        "x": (n_batch, n_in),
        "W1": (n_in, n_hidden),
        "b1": (n_hidden,),
        "hidden_preactivation": (n_batch, n_hidden),
        "hidden_activation": (n_batch, n_hidden),
        "W2": (n_hidden, n_classes),
        "b2": (n_classes,),
        "logits": (n_batch, n_classes),
        "probabilities": (n_batch, n_classes),
    }


def affine(X, W, bias):
    """Row-batch affine map Z = X @ W + b with trailing-axis bias broadcast."""

    np = _require_numpy()
    batch = _as_2d(X)
    weights = _as_2d(W)
    offset = np.asarray(_as_1d(bias), dtype=float)
    if batch.shape[1] != weights.shape[0]:
        raise ValueError(
            f"X has {batch.shape[1]} features but W has {weights.shape[0]} rows"
        )
    if offset.shape[0] != weights.shape[1]:
        raise ValueError(
            f"bias has {offset.shape[0]} entries but W has {weights.shape[1]} columns"
        )
    _require_finite(batch, "X")
    _require_finite(weights, "W")
    _require_finite(offset, "bias")
    return batch @ weights + offset


def apply_hidden_activation(preactivation, name: str = HIDDEN_ACTIVATION):
    """Apply an M22 activation after the affine map."""

    return _load_m22().apply_activation(preactivation, name)


def stable_softmax(logits, axis: int = CLASS_AXIS):
    """Numerically stable softmax along the class axis (last axis by default).

    Subtracts the per-axis maximum before exp so large logits stay finite.
    The batch axis is never the default reduction axis.
    """

    np = _require_numpy()
    array = np.asarray(logits, dtype=float)
    if array.ndim == 0:
        raise ValueError("softmax requires at least a 1-D logit vector")
    _require_finite(array, "logits")
    axis_index = axis if axis >= 0 else array.ndim + axis
    if axis_index < 0 or axis_index >= array.ndim:
        raise ValueError(f"softmax axis {axis} is out of range for shape {array.shape}")
    shifted = array - np.max(array, axis=axis, keepdims=True)
    unnormalized = np.exp(shifted)
    return unnormalized / np.sum(unnormalized, axis=axis, keepdims=True)


def scalar_affine(x, W, bias) -> tuple[float, ...]:
    """One-example affine map with explicit Python loops (no matmul)."""

    features = tuple(float(value) for value in _as_1d(x))
    weights = tuple(tuple(float(value) for value in row) for row in _as_2d(W))
    offset = tuple(float(value) for value in _as_1d(bias))
    if len(weights) != len(features):
        raise ValueError(
            f"feature count {len(features)} does not match W rows {len(weights)}"
        )
    n_out = len(weights[0]) if weights else 0
    if any(len(row) != n_out for row in weights):
        raise ValueError("W must be rectangular")
    if len(offset) != n_out:
        raise ValueError(f"bias has {len(offset)} entries but W has {n_out} columns")
    preactivation = []
    for unit in range(n_out):
        total = offset[unit]
        for feature, row in zip(features, weights):
            total += feature * row[unit]
        preactivation.append(total)
    return tuple(preactivation)


def scalar_activation(values, name: str = HIDDEN_ACTIVATION) -> tuple[float, ...]:
    key = str(name).lower()
    if key in {"identity", "linear"}:
        return tuple(float(value) for value in values)
    if key == "relu":
        return tuple(float(value) if float(value) > 0.0 else 0.0 for value in values)
    raise ValueError(f"scalar path supports identity or relu, not {name!r}")


def scalar_softmax(logits) -> tuple[float, ...]:
    values = tuple(float(value) for value in logits)
    if not values:
        raise ValueError("softmax requires at least one logit")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("logits must be finite")
    return _softmax_tuple(values)


def scalar_two_layer_one_example(
    x,
    W1,
    b1,
    W2,
    b2,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
) -> ScalarTrace:
    """Forward one example with loops: affine, activate, affine, softmax."""

    features = tuple(float(value) for value in _as_1d(x))
    hidden_preactivation = scalar_affine(features, W1, b1)
    hidden_activation_values = scalar_activation(hidden_preactivation, hidden_activation)
    logits = scalar_affine(hidden_activation_values, W2, b2)
    probabilities = scalar_softmax(logits)
    return ScalarTrace(
        x=features,
        hidden_preactivation=hidden_preactivation,
        hidden_activation=hidden_activation_values,
        logits=logits,
        probabilities=probabilities,
        hidden_activation_name=str(hidden_activation).lower(),
    )


def two_layer_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
) -> ForwardTrace:
    """Vectorized two-layer inference with inspectable named intermediates.

    Hidden activation is applied after the first affine map. The second affine
    map emits logits. Softmax is applied along the class axis only.
    """

    shapes = stacked_shapes(X, W1, b1, W2, b2)
    hidden_preactivation = affine(X, W1, b1)
    hidden = apply_hidden_activation(hidden_preactivation, hidden_activation)
    logits = affine(hidden, W2, b2)
    probabilities = stable_softmax(logits, axis=CLASS_AXIS)
    return ForwardTrace(
        x=_as_2d(X),
        hidden_preactivation=hidden_preactivation,
        hidden_activation=hidden,
        logits=logits,
        probabilities=probabilities,
        hidden_activation_name=str(hidden_activation).lower(),
        shapes=shapes,
        defect="none",
    )


def two_layer_forward_with_defect(
    X,
    W1,
    b1,
    W2,
    b2,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
    defect: str = "none",
) -> ForwardTrace:
    """Apply a named inference-boundary defect. Used by the controlled-failure lab.

    Defects still emit finite numbers. They do not train, differentiate, or
    introduce a loss.
    """

    key = str(defect).lower()
    if key in {"none", "correct", ""}:
        trace = two_layer_forward(
            X, W1, b1, W2, b2, hidden_activation=hidden_activation
        )
        return ForwardTrace(
            x=trace.x,
            hidden_preactivation=trace.hidden_preactivation,
            hidden_activation=trace.hidden_activation,
            logits=trace.logits,
            probabilities=trace.probabilities,
            hidden_activation_name=trace.hidden_activation_name,
            shapes=trace.shapes,
            defect="none",
        )
    if key in {"omitted_hidden_activation", "skip_hidden_activation"}:
        trace = two_layer_forward(
            X, W1, b1, W2, b2, hidden_activation="identity"
        )
        return ForwardTrace(
            x=trace.x,
            hidden_preactivation=trace.hidden_preactivation,
            hidden_activation=trace.hidden_activation,
            logits=trace.logits,
            probabilities=trace.probabilities,
            hidden_activation_name="identity",
            shapes=trace.shapes,
            defect="omitted_hidden_activation",
        )
    if key in {"softmax_axis_batch", "wrong_axis_softmax"}:
        trace = two_layer_forward(
            X, W1, b1, W2, b2, hidden_activation=hidden_activation
        )
        probabilities = stable_softmax(trace.logits, axis=BATCH_AXIS)
        return ForwardTrace(
            x=trace.x,
            hidden_preactivation=trace.hidden_preactivation,
            hidden_activation=trace.hidden_activation,
            logits=trace.logits,
            probabilities=probabilities,
            hidden_activation_name=trace.hidden_activation_name,
            shapes=trace.shapes,
            defect="softmax_axis_batch",
        )
    raise ValueError(f"unknown defect {defect!r}; use one of {SUPPORTED_DEFECTS}")


def m22_reference_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
) -> ForwardTrace:
    """Trusted stack: M22 dense_forward twice, then M23 class-axis softmax."""

    m22 = _load_m22()
    shapes = stacked_shapes(X, W1, b1, W2, b2)
    hidden_preactivation = m22.dense_forward(X, W1, b1, "identity")
    hidden = m22.dense_forward(X, W1, b1, hidden_activation)
    logits = m22.dense_forward(hidden, W2, b2, "identity")
    probabilities = stable_softmax(logits, axis=CLASS_AXIS)
    return ForwardTrace(
        x=_as_2d(X),
        hidden_preactivation=hidden_preactivation,
        hidden_activation=hidden,
        logits=logits,
        probabilities=probabilities,
        hidden_activation_name=str(hidden_activation).lower(),
        shapes=shapes,
        defect="none",
    )


def reference_forward() -> ForwardTrace:
    """Run the teaching fixture through the NumPy stack."""

    return two_layer_forward(
        REFERENCE_X,
        REFERENCE_W1,
        REFERENCE_B1,
        REFERENCE_W2,
        REFERENCE_B2,
        hidden_activation=HIDDEN_ACTIVATION,
    )


def arrays_close(left, right, *, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    np = _require_numpy()
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        return False
    return bool(np.allclose(left_arr, right_arr, atol=atol, rtol=rtol))


def intermediate_parity(
    left: ForwardTrace,
    right: ForwardTrace,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, bool]:
    """Compare every named intermediate; used as the M24 forward-parity contract."""

    return {
        name: arrays_close(getattr(left, name), getattr(right, name), atol=atol, rtol=rtol)
        for name in INTERMEDIATE_NAMES
    }


def singleton_batch_parity(
    X,
    W1,
    b1,
    W2,
    b2,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
    defect: str = "none",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, object]:
    """Row-by-row agreement between a batch run and one-example runs."""

    np = _require_numpy()
    batch = two_layer_forward_with_defect(
        X, W1, b1, W2, b2, hidden_activation=hidden_activation, defect=defect
    )
    rows = _as_2d(X)
    matches = []
    for index, row in enumerate(rows):
        single = two_layer_forward_with_defect(
            row, W1, b1, W2, b2, hidden_activation=hidden_activation, defect=defect
        )
        matches.append(
            arrays_close(
                single.probabilities[0],
                batch.probabilities[index],
                atol=atol,
                rtol=rtol,
            )
        )
    return {
        "row_matches": tuple(bool(flag) for flag in matches),
        "all_match": bool(all(matches)),
        "batch_row_sums": _nested_tuples(np.sum(batch.probabilities, axis=CLASS_AXIS)),
        "batch_column_sums": _nested_tuples(np.sum(batch.probabilities, axis=BATCH_AXIS)),
        "defect": batch.defect,
    }


def perturb_matrix(W, index: tuple[int, int], delta: float):
    np = _require_numpy()
    out = np.asarray(W, dtype=float).copy()
    row, column = index
    out[row, column] = out[row, column] + float(delta)
    return out


def reorder_rows(values, order):
    np = _require_numpy()
    array = _as_2d(values)
    permutation = tuple(int(index) for index in order)
    if sorted(permutation) != list(range(array.shape[0])):
        raise ValueError("order must be a permutation of batch indices")
    return array[list(permutation), :]


def shift_logits(logits, example_index: int, constant: float):
    np = _require_numpy()
    out = np.asarray(logits, dtype=float).copy()
    if out.ndim != 2:
        raise ValueError("shift_logits expects a (batch, n_classes) array")
    out[int(example_index), :] = out[int(example_index), :] + float(constant)
    return out


def probability_row_sums(probabilities):
    np = _require_numpy()
    array = np.asarray(probabilities, dtype=float)
    return np.sum(array, axis=CLASS_AXIS)


def forward_report(trace: ForwardTrace) -> dict[str, object]:
    """Compact observable evidence for one forward pass."""

    np = _require_numpy()
    probabilities = np.asarray(trace.probabilities, dtype=float)
    return {
        "shapes": dict(trace.shapes),
        "defect": trace.defect,
        "hidden_activation": trace.hidden_activation_name,
        "hidden_preactivation": _nested_tuples(trace.hidden_preactivation),
        "hidden_activation_values": _nested_tuples(trace.hidden_activation),
        "logits": _nested_tuples(trace.logits),
        "probabilities": _nested_tuples(probabilities),
        "probability_row_sums": _nested_tuples(np.sum(probabilities, axis=CLASS_AXIS)),
        "argmax_class": tuple(int(index) for index in np.argmax(probabilities, axis=CLASS_AXIS)),
    }
