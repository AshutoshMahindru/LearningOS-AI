"""Transparent neuron and dense-layer arithmetic for Mission M22.

M22 opens the affine neuron, bias, activations, and one dense layer as observable
arithmetic. Multi-layer inference stacks, logits/softmax, parameter gradients, and
framework autograd remain deferred to M23-M25.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# Hand-computable one-neuron fixture: z = 0.5*1 + (-0.25)*2 + 0.5 = 0.5
REFERENCE_X = (1.0, 2.0)
REFERENCE_WEIGHTS = (0.5, -0.25)
REFERENCE_BIAS = 0.5
REFERENCE_PREACTIVATION = 0.5

# Row-batch dense fixture: X (2, 3), W (3, 2), b (2,)
REFERENCE_LAYER_X = (
    (1.0, 0.0, -1.0),
    (0.5, 0.5, 0.5),
)
REFERENCE_LAYER_W = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
)
REFERENCE_LAYER_BIAS = (0.0, 0.5)

SUPPORTED_ACTIVATIONS = ("identity", "linear", "relu", "sigmoid", "tanh")


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M22 requires numpy; install requirements/m22.txt") from exc
    return np


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


def relu(preactivation):
    """Rectified linear unit: max(z, 0)."""

    np = _require_numpy()
    z = np.asarray(preactivation, dtype=float)
    return np.maximum(z, 0.0)


def sigmoid(preactivation):
    """Logistic sigmoid with a stable split at zero."""

    np = _require_numpy()
    z = np.asarray(preactivation, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def tanh(preactivation):
    np = _require_numpy()
    return np.tanh(np.asarray(preactivation, dtype=float))


def identity(preactivation):
    np = _require_numpy()
    return np.asarray(preactivation, dtype=float).copy()


def apply_activation(preactivation, name: str):
    key = str(name).lower()
    table = {
        "identity": identity,
        "linear": identity,
        "relu": relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
    }
    if key not in table:
        raise ValueError(f"unsupported activation {name!r}; use one of {SUPPORTED_ACTIVATIONS}")
    return table[key](preactivation)


def affine_preactivation(x, weights, bias: float) -> float:
    """Return z = w·x + b for one neuron. Does not apply an activation."""

    features = _as_1d(x)
    coeffs = _as_1d(weights)
    if len(features) != len(coeffs):
        raise ValueError(
            f"feature count {len(features)} does not match weight count {len(coeffs)}"
        )
    if not math.isfinite(float(bias)):
        raise ValueError("bias must be finite")
    return float(sum(feature * weight for feature, weight in zip(features, coeffs)) + float(bias))


@dataclass(frozen=True)
class NeuronTrace:
    x: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    weighted_sum: float
    preactivation: float
    activation_name: str
    output: float


def neuron_trace(x, weights, bias: float, activation: str = "relu") -> NeuronTrace:
    """Trace one affine neuron through weighted sum, bias, activation."""

    features = _as_1d(x)
    coeffs = _as_1d(weights)
    preactivation = affine_preactivation(features, coeffs, bias)
    weighted_sum = float(preactivation - float(bias))
    output = float(apply_activation(preactivation, activation).reshape(-1)[0])
    return NeuronTrace(
        x=features,
        weights=coeffs,
        bias=float(bias),
        weighted_sum=weighted_sum,
        preactivation=preactivation,
        activation_name=str(activation).lower(),
        output=output,
    )


def activation_sweep(preactivations, names: tuple[str, ...] = ("relu", "sigmoid", "tanh")) -> dict[str, tuple[float, ...]]:
    """Apply named activations to one shared pre-activation sequence."""

    values = _as_1d(preactivations)
    return {
        name: tuple(float(v) for v in apply_activation(values, name).reshape(-1))
        for name in names
    }


def validate_dense_shapes(X, W, bias) -> tuple[int, int, int]:
    """Return (batch, n_in, n_out) or raise a shape contract error."""

    batch = _as_2d(X)
    weights = _as_2d(W)
    offset = _as_1d(bias)
    n_batch, n_in = batch.shape
    n_in_w, n_out = weights.shape
    if n_in != n_in_w:
        raise ValueError(
            f"X has {n_in} features but W has {n_in_w} rows; expected W shape (n_in, n_out)"
        )
    if len(offset) != n_out:
        raise ValueError(
            f"bias has {len(offset)} entries but W has {n_out} columns; expected b shape (n_out,)"
        )
    return n_batch, n_in, n_out


def dense_forward(X, W, bias, activation: str = "relu"):
    """Row-batch dense layer: Y = activation(X @ W + b).

    X has shape (batch, n_in). W has shape (n_in, n_out). b has shape (n_out,).
    Activation is applied after the affine map, never before.
    """

    np = _require_numpy()
    validate_dense_shapes(X, W, bias)
    batch = _as_2d(X)
    weights = _as_2d(W)
    offset = np.asarray(_as_1d(bias), dtype=float)
    preactivation = batch @ weights + offset
    output = apply_activation(preactivation, activation)
    return output


def dense_forward_with_defect(X, W, bias, activation: str = "relu", *, defect: str = "none"):
    """Apply a named layer-boundary defect. Used by the controlled-failure lab.

    Defects emit numbers or raise a discriminating shape error; they do not
    train, differentiate, or stack a full inference network.
    """

    np = _require_numpy()
    key = str(defect).lower()
    if key in {"none", "correct", ""}:
        return dense_forward(X, W, bias, activation)
    batch = _as_2d(X)
    weights = _as_2d(W)
    offset = np.asarray(_as_1d(bias), dtype=float)
    if key in {"transposed_weights", "wrong_orientation"}:
        # Treat W as if it were (n_out, n_in) and multiply anyway when square,
        # or fail the (n_in, n_out) contract when rectangular.
        if weights.shape[0] == weights.shape[1]:
            preactivation = batch @ weights.T + offset
            return apply_activation(preactivation, activation)
        raise ValueError(
            "transposed W is incompatible with X @ W for rectangular (n_in, n_out) weights"
        )
    if key in {"activation_before_affine", "wrong_boundary"}:
        activated_input = apply_activation(batch, activation)
        return activated_input @ weights + offset
    raise ValueError(f"unknown defect {defect!r}")


def collapsed_affine(W1, b1, W2, b2) -> tuple:
    """Return (W_eq, b_eq) such that (X @ W1 + b1) @ W2 + b2 = X @ W_eq + b_eq."""

    np = _require_numpy()
    first = _as_2d(W1)
    second = _as_2d(W2)
    bias1 = np.asarray(_as_1d(b1), dtype=float)
    bias2 = np.asarray(_as_1d(b2), dtype=float)
    if first.shape[1] != second.shape[0]:
        raise ValueError("hidden width of W1 must match input width of W2")
    if first.shape[1] != len(bias1):
        raise ValueError("b1 length must match W1 column count")
    if second.shape[1] != len(bias2):
        raise ValueError("b2 length must match W2 column count")
    weights_eq = first @ second
    bias_eq = bias1 @ second + bias2
    return weights_eq, bias_eq


def compose_two_layers(X, W1, b1, W2, b2, *, hidden_activation: str = "identity"):
    """Compose two dense maps. Hidden identity collapses to one affine map."""

    hidden = dense_forward(X, W1, b1, hidden_activation)
    return dense_forward(hidden, W2, b2, "identity")


def reference_neuron_relu() -> NeuronTrace:
    return neuron_trace(REFERENCE_X, REFERENCE_WEIGHTS, REFERENCE_BIAS, "relu")


def layer_report(output) -> dict[str, object]:
    """Compact observable evidence for a dense-layer run."""

    np = _require_numpy()
    array = _as_2d(output)
    return {
        "batch": int(array.shape[0]),
        "n_out": int(array.shape[1]),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "rows": tuple(tuple(float(v) for v in row) for row in array),
    }
