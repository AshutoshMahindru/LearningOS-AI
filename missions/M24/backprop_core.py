"""Transparent reverse-mode credit assignment for Mission M24.

M24 assigns blame on the trusted M23 forward graph
x -> hidden_preactivation -> hidden_activation -> logits -> probabilities -> loss.
Local derivatives, chain rule, reverse accumulation, branches, gradient
checks, and one declared update live here. Framework autograd and a
training loop remain deferred to M25.

Canonical path: manual reverse-mode plus central finite differences.
This is not framework autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import sys
from typing import Callable, Sequence


# Restated M19 one-parameter fixture: y = w * x, L = mean((y - t)^2).
M19_XS = (1.0, 2.0)
M19_YS = (3.0, 6.0)
M19_WEIGHT = 1.0
M19_LEARNING_RATE = 0.1

# Scalar chain: z = w*x + b -> h = relu(z) -> y = v*h + c -> L = 0.5*(y - t)^2
SCALAR_X = 2.0
SCALAR_W = 0.5
SCALAR_B = 0.0
SCALAR_V = -1.5
SCALAR_C = 0.0
SCALAR_T = 0.5

# Shared node h feeding two MSE heads. dL/dh must equal the summed contributions.
BRANCH_H = 2.0
BRANCH_V1 = 1.0
BRANCH_V2 = -0.5
BRANCH_T1 = 0.0
BRANCH_T2 = 1.0

# ReLU boundary pair for the scalar chain (x = 2, b = 0): z = w * x.
ACTIVATION_W_NEGATIVE = -0.25  # z = -0.5, relu' = 0
ACTIVATION_W_POSITIVE = 0.25  # z = 0.5, relu' = 1

TEACHING_TARGETS = (0, 0)
TEACHING_ROW = 1
SIGN_PARAMETER = "W2"
SIGN_INDEX = (0, 0)
SIGN_DELTA = 0.1
ONE_STEP_LEARNING_RATE = 0.25
ONE_STEP_PARAMETER = "W2"
ONE_STEP_INDEX = (0, 0)

DEFAULT_EPSILON = 1e-5
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5
DEFAULT_DTYPE_NAME = "float64"
DEFAULT_REDUCTION = "mean"
DEFAULT_LOSS = "softmax_nll"
HIDDEN_ACTIVATION = "relu"
RELU_AT_ZERO = 0.0
DEFAULT_EPSILON_GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)

SUPPORTED_DEFECTS = (
    "none",
    "correct",
    "omitted_branch",
    "wrong_relu_derivative",
)
PARAMETER_NAMES = ("W1", "b1", "W2", "b2")
ACTIVATION_GRAD_NAMES = ("hidden_activation", "hidden_preactivation")
FORWARD_GRAPH_NODES = (
    "x",
    "hidden_preactivation",
    "hidden_activation",
    "logits",
    "probabilities",
    "loss",
)
REVERSE_GRAPH_NODES = (
    "loss",
    "logits",
    "hidden_activation",
    "hidden_preactivation",
    "x",
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M24 requires numpy; install requirements/m24.txt") from exc
    return np


def _load_m19():
    """Load the trusted M19 one-parameter gradient invariant."""

    name = "_learningos_m19_gradient_core"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent.parent / "M19" / "gradient_core.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load missions/M19/gradient_core.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_m23():
    """Import missions.M23.forward_core as the trusted forward graph."""

    name = "_learningos_m23_forward_core"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import missions.M23.forward_core as module
    except ImportError:
        path = Path(__file__).resolve().parent.parent / "M23" / "forward_core.py"
        spec = importlib.util.spec_from_file_location("missions.M23.forward_core", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load missions/M23/forward_core.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _as_1d(values):
    np = _require_numpy()
    array = np.asarray(values, dtype=float).reshape(-1)
    return array


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


def _normalize_defect(defect: str) -> str:
    key = str(defect).lower()
    if key in {"none", "correct", ""}:
        return "none"
    if key in {"omitted_branch", "omit_branch", "dropped_branch"}:
        return "omitted_branch"
    if key in {"wrong_relu_derivative", "wrong_activation_derivative", "relu_identity"}:
        return "wrong_relu_derivative"
    raise ValueError(f"unknown defect {defect!r}; use one of {SUPPORTED_DEFECTS}")


def _clone_params(W1, b1, W2, b2):
    np = _require_numpy()
    return (
        np.asarray(W1, dtype=float).copy(),
        np.asarray(b1, dtype=float).copy(),
        np.asarray(W2, dtype=float).copy(),
        np.asarray(b2, dtype=float).copy(),
    )


def arrays_close(left, right, *, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    np = _require_numpy()
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        return False
    return bool(np.allclose(left_arr, right_arr, atol=atol, rtol=rtol))


# ---------------------------------------------------------------------------
# M19 restatement (pure Python)
# ---------------------------------------------------------------------------


def restated_m19_invariant(
    xs: Sequence[float] = M19_XS,
    ys: Sequence[float] = M19_YS,
    weight: float = M19_WEIGHT,
    epsilon: float = DEFAULT_EPSILON,
    learning_rate: float = M19_LEARNING_RATE,
) -> dict[str, float | str]:
    """Replay the M19 one-parameter invariant on a tiny deterministic pair."""

    m19 = _load_m19()
    loss = m19.one_parameter_loss(xs, ys, weight)
    analytic = m19.analytic_weight_gradient(xs, ys, weight)
    estimated = m19.finite_difference_gradient(xs, ys, weight, epsilon)
    updated = m19.update_parameter(weight, analytic, learning_rate)
    return {
        "loss": float(loss),
        "analytic_gradient": float(analytic),
        "finite_difference": float(estimated),
        "updated_weight": float(updated),
        "learning_rate": float(learning_rate),
        "update_convention": "parameter - learning_rate * gradient",
    }


# ---------------------------------------------------------------------------
# Reverse-mode tape (pure Python; code-reading target)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TapeNode:
    """One stored forward value plus local derivatives to its parents."""

    name: str
    value: float
    parents: tuple[tuple[str, float], ...] = ()


@dataclass
class GradientTape:
    """Mutable reverse-mode state so reset versus accumulate stays visible."""

    nodes: tuple[TapeNode, ...]
    grads: dict[str, float] = field(default_factory=dict)

    def names(self) -> set[str]:
        names = {node.name for node in self.nodes}
        names.update(parent for node in self.nodes for parent, _ in node.parents)
        return names

    def reset(self) -> dict[str, float]:
        """Gradient-reset semantics: zero every known name before a reverse pass."""

        self.grads = {name: 0.0 for name in self.names()}
        return self.grads


def _fresh_reverse(
    nodes: Sequence[TapeNode],
    *,
    output: str,
    seed_gradient: float,
) -> dict[str, float]:
    """Compute one reverse pass into a new dictionary (does not add to stored grads)."""

    ordered = tuple(nodes)
    names = {node.name for node in ordered}
    names.update(parent for node in ordered for parent, _ in node.parents)
    if output not in names:
        raise ValueError(f"output {output!r} is not on the tape")
    if not math.isfinite(seed_gradient):
        raise ValueError("seed_gradient must be finite")
    delta = {name: 0.0 for name in names}
    delta[output] = float(seed_gradient)
    for node in reversed(ordered):
        upstream = delta[node.name]
        for parent, local in node.parents:
            delta[parent] += upstream * float(local)
    return delta


def reverse_accumulate(
    nodes: Sequence[TapeNode],
    *,
    output: str = "loss",
    seed_gradient: float = 1.0,
    grads: dict[str, float] | None = None,
    reset: bool = True,
) -> dict[str, float]:
    """Reverse-mode accumulation in reverse topological order.

    Each parent receives ``upstream_gradient * local_derivative``. At a
    branch, contributions **add**. A reverse pass computes a *fresh* delta
    and then adds it into ``grads``. If ``reset`` is false, a second reverse
    pass doubles stored parameter grads — the stale-gradient hazard M25
    will later name ``zero_grad``.
    """

    delta = _fresh_reverse(nodes, output=output, seed_gradient=seed_gradient)
    if grads is None:
        grads = {name: 0.0 for name in delta}
    elif reset:
        for name in delta:
            grads[name] = 0.0
    else:
        for name in delta:
            grads.setdefault(name, 0.0)
    for name, value in delta.items():
        grads[name] = grads.get(name, 0.0) + value
    return grads


def relu_local_derivative_scalar(preactivation: float) -> float:
    """relu'(z) = 1 if z > 0 else 0, including z == 0."""

    return 1.0 if float(preactivation) > 0.0 else RELU_AT_ZERO


def build_scalar_chain_tape(
    x: float = SCALAR_X,
    w: float = SCALAR_W,
    b: float = SCALAR_B,
    v: float = SCALAR_V,
    c: float = SCALAR_C,
    t: float = SCALAR_T,
) -> tuple[TapeNode, ...]:
    """Store values and local derivatives for the teaching scalar chain."""

    z = float(w) * float(x) + float(b)
    relu_prime = relu_local_derivative_scalar(z)
    h = z if relu_prime else 0.0
    y = float(v) * h + float(c)
    residual = y - float(t)
    loss = 0.5 * residual * residual
    return (
        TapeNode("z", z, (("w", float(x)), ("x", float(w)), ("b", 1.0))),
        TapeNode("h", h, (("z", relu_prime),)),
        TapeNode("y", y, (("v", h), ("h", float(v)), ("c", 1.0))),
        TapeNode("loss", loss, (("y", residual),)),
    )


# ---------------------------------------------------------------------------
# Scalar chain and branch (pure Python)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarChainTrace:
    x: float
    w: float
    b: float
    z: float
    h: float
    v: float
    c: float
    y: float
    t: float
    loss: float
    relu_prime: float
    grads: dict[str, float]
    defect: str = "none"


@dataclass(frozen=True)
class BranchTrace:
    h: float
    v1: float
    v2: float
    t1: float
    t2: float
    y1: float
    y2: float
    loss: float
    contribution_path1: float
    contribution_path2: float
    d_h: float
    defect: str = "none"


def scalar_chain(
    x: float = SCALAR_X,
    w: float = SCALAR_W,
    b: float = SCALAR_B,
    v: float = SCALAR_V,
    c: float = SCALAR_C,
    t: float = SCALAR_T,
    *,
    defect: str = "none",
) -> ScalarChainTrace:
    """Forward plus reverse for the scalar affine-ReLU-affine-MSE chain."""

    key = _normalize_defect(defect)
    z = float(w) * float(x) + float(b)
    true_prime = relu_local_derivative_scalar(z)
    h = z if true_prime else 0.0
    y = float(v) * h + float(c)
    residual = y - float(t)
    loss = 0.5 * residual * residual
    used_prime = 1.0 if key == "wrong_relu_derivative" else true_prime
    grads = reverse_accumulate(
        (
            TapeNode("z", z, (("w", float(x)), ("x", float(w)), ("b", 1.0))),
            TapeNode("h", h, (("z", used_prime),)),
            TapeNode("y", y, (("v", h), ("h", float(v)), ("c", 1.0))),
            TapeNode("loss", loss, (("y", residual),)),
        )
    )
    return ScalarChainTrace(
        x=float(x),
        w=float(w),
        b=float(b),
        z=z,
        h=h,
        v=float(v),
        c=float(c),
        y=y,
        t=float(t),
        loss=loss,
        relu_prime=true_prime,
        grads=grads,
        defect=key,
    )


def scalar_chain_loss(
    x: float = SCALAR_X,
    w: float = SCALAR_W,
    b: float = SCALAR_B,
    v: float = SCALAR_V,
    c: float = SCALAR_C,
    t: float = SCALAR_T,
) -> float:
    return scalar_chain(x=x, w=w, b=b, v=v, c=c, t=t).loss


def branch_gradients(
    h: float = BRANCH_H,
    v1: float = BRANCH_V1,
    v2: float = BRANCH_V2,
    t1: float = BRANCH_T1,
    t2: float = BRANCH_T2,
    *,
    defect: str = "none",
) -> BranchTrace:
    """A stored value feeding two downstream MSE paths; grads add at the branch."""

    key = _normalize_defect(defect)
    y1 = float(v1) * float(h)
    y2 = float(v2) * float(h)
    loss = 0.5 * (y1 - float(t1)) ** 2 + 0.5 * (y2 - float(t2)) ** 2
    d_y1 = y1 - float(t1)
    d_y2 = y2 - float(t2)
    contribution_path1 = d_y1 * float(v1)
    contribution_path2 = d_y2 * float(v2)
    d_h = contribution_path1 if key == "omitted_branch" else contribution_path1 + contribution_path2
    return BranchTrace(
        h=float(h),
        v1=float(v1),
        v2=float(v2),
        t1=float(t1),
        t2=float(t2),
        y1=y1,
        y2=y2,
        loss=loss,
        contribution_path1=contribution_path1,
        contribution_path2=contribution_path2,
        d_h=d_h,
        defect=key,
    )


def branch_loss(
    h: float = BRANCH_H,
    v1: float = BRANCH_V1,
    v2: float = BRANCH_V2,
    t1: float = BRANCH_T1,
    t2: float = BRANCH_T2,
) -> float:
    return branch_gradients(h=h, v1=v1, v2=v2, t1=t1, t2=t2).loss


def activation_boundary_pair(
    x: float = SCALAR_X,
    w_negative: float = ACTIVATION_W_NEGATIVE,
    w_positive: float = ACTIVATION_W_POSITIVE,
    b: float = SCALAR_B,
    v: float = SCALAR_V,
    c: float = SCALAR_C,
    t: float = SCALAR_T,
) -> tuple[ScalarChainTrace, ScalarChainTrace]:
    """Same downstream loss; only the pre-activation crosses the ReLU hinge."""

    left = scalar_chain(x=x, w=w_negative, b=b, v=v, c=c, t=t)
    right = scalar_chain(x=x, w=w_positive, b=b, v=v, c=c, t=t)
    return left, right


def scalar_one_step(
    *,
    x: float = SCALAR_X,
    w: float = SCALAR_W,
    b: float = SCALAR_B,
    v: float = SCALAR_V,
    c: float = SCALAR_C,
    t: float = SCALAR_T,
    learning_rate: float = ONE_STEP_LEARNING_RATE,
    parameter: str = "w",
) -> dict[str, float | str]:
    """One declared descent step on a single scalar parameter. Not a training loop."""

    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    before = scalar_chain(x=x, w=w, b=b, v=v, c=c, t=t)
    if parameter not in before.grads:
        raise ValueError(f"unknown scalar parameter {parameter!r}")
    gradient = before.grads[parameter]
    current = {"w": w, "b": b, "v": v, "c": c, "x": x}
    if parameter not in current:
        raise ValueError(f"{parameter} is not an updatable scalar parameter")
    updated = dict(current)
    updated[parameter] = current[parameter] - learning_rate * gradient
    after = scalar_chain(
        x=updated["x"],
        w=updated["w"],
        b=updated["b"],
        v=updated["v"],
        c=updated["c"],
        t=t,
    )
    return {
        "parameter": parameter,
        "value_before": float(current[parameter]),
        "value_after": float(updated[parameter]),
        "gradient": float(gradient),
        "learning_rate": float(learning_rate),
        "loss_before": float(before.loss),
        "loss_after": float(after.loss),
        "update_convention": "parameter - learning_rate * gradient",
    }


# ---------------------------------------------------------------------------
# Central finite differences
# ---------------------------------------------------------------------------


def central_finite_difference(
    fn: Callable[[float], float],
    value: float,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Estimate df/d(value) with a two-sided stencil. Epsilon must be positive."""

    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    plus = float(fn(float(value) + float(epsilon)))
    minus = float(fn(float(value) - float(epsilon)))
    if not math.isfinite(plus) or not math.isfinite(minus):
        raise ValueError("finite difference probe produced a non-finite loss")
    return (plus - minus) / (2.0 * float(epsilon))


def relative_error(analytic: float, estimated: float) -> float:
    scale = max(abs(float(analytic)), abs(float(estimated)), 1e-12)
    return abs(float(analytic) - float(estimated)) / scale


def finite_difference_agrees(
    analytic: float,
    estimated: float,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    return abs(float(analytic) - float(estimated)) <= atol + rtol * abs(float(analytic))


@dataclass(frozen=True)
class FiniteDifferenceReport:
    name: str
    analytic: float
    estimated: float
    epsilon: float
    absolute_error: float
    relative_error: float
    agrees: bool


def finite_difference_report(
    fn: Callable[[float], float],
    value: float,
    analytic: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    name: str = "parameter",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FiniteDifferenceReport:
    estimated = central_finite_difference(fn, value, epsilon)
    abs_err = abs(float(analytic) - estimated)
    return FiniteDifferenceReport(
        name=name,
        analytic=float(analytic),
        estimated=estimated,
        epsilon=float(epsilon),
        absolute_error=abs_err,
        relative_error=relative_error(analytic, estimated),
        agrees=finite_difference_agrees(analytic, estimated, atol=atol, rtol=rtol),
    )


def finite_difference_sweep(
    fn: Callable[[float], float],
    value: float,
    analytic: float,
    epsilons: Sequence[float] = DEFAULT_EPSILON_GRID,
    *,
    name: str = "parameter",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[FiniteDifferenceReport, ...]:
    reports = []
    for epsilon in epsilons:
        reports.append(
            finite_difference_report(
                fn,
                value,
                analytic,
                epsilon=float(epsilon),
                name=name,
                atol=atol,
                rtol=rtol,
            )
        )
    return tuple(reports)


# ---------------------------------------------------------------------------
# Dense-layer / softmax-NLL reverse pass on the M23 graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackwardTrace:
    """Parameter, activation, and loss gradients for one two-layer reverse pass."""

    loss: float
    targets: tuple[int, ...]
    probabilities: object
    d_logits: object
    d_hidden_activation: object
    d_hidden_preactivation: object
    d_x: object
    d_W1: object
    d_b1: object
    d_W2: object
    d_b2: object
    shapes: dict[str, tuple[int, ...]]
    defect: str = "none"
    reduction: str = DEFAULT_REDUCTION
    loss_name: str = DEFAULT_LOSS

    def parameter_grads(self) -> dict[str, object]:
        return {"W1": self.d_W1, "b1": self.d_b1, "W2": self.d_W2, "b2": self.d_b2}

    def activation_grads(self) -> dict[str, object]:
        return {
            "hidden_activation": self.d_hidden_activation,
            "hidden_preactivation": self.d_hidden_preactivation,
        }


def relu_local_derivative(preactivation):
    """Elementwise relu'(z): 1 if z > 0 else 0 (relu'(0) = 0)."""

    np = _require_numpy()
    z = np.asarray(preactivation, dtype=float)
    return (z > 0.0).astype(float)


def activation_local_derivative(preactivation, name: str = HIDDEN_ACTIVATION):
    """Local derivative of the hidden map used in the trusted forward pass."""

    np = _require_numpy()
    z = np.asarray(preactivation, dtype=float)
    key = str(name).lower()
    if key in {"identity", "linear"}:
        return np.ones_like(z, dtype=float)
    if key == "relu":
        return relu_local_derivative(z)
    raise ValueError(f"unsupported hidden activation {name!r}")


def relu_kink_parameter_entries(X, hidden_preactivation, *, atol: float = 1e-12):
    """Parameter entries whose perturbation moves a hidden unit sitting at relu(0).

    Central differences assume local smoothness. A unit with z == 0 is a hinge;
    plus and minus probes disagree, so those entries are not a valid check.
    """

    np = _require_numpy()
    batch = _as_2d(X)
    pre = np.asarray(hidden_preactivation, dtype=float)
    if pre.ndim == 1:
        pre = pre.reshape(1, -1)
    entries: list[tuple[str, tuple[int, ...]]] = []
    kink = np.abs(pre) <= float(atol)
    for example_index, unit in np.argwhere(kink):
        entries.append(("b1", (int(unit),)))
        for feature in range(batch.shape[1]):
            if abs(float(batch[int(example_index), feature])) > float(atol):
                entries.append(("W1", (int(feature), int(unit))))
    unique = []
    seen = set()
    for item in entries:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def softmax_nll(logits, targets, *, reduction: str = DEFAULT_REDUCTION) -> float:
    """Mean or sum of -log softmax probability of the target class."""

    np = _require_numpy()
    array = _as_2d(logits)
    _require_finite(array, "logits")
    index = np.asarray(targets, dtype=int).reshape(-1)
    if index.shape[0] != array.shape[0]:
        raise ValueError("targets must have one class index per example")
    if np.any(index < 0) or np.any(index >= array.shape[1]):
        raise ValueError("target class out of range")
    shifted = array - np.max(array, axis=1, keepdims=True)
    log_z = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    nll = -log_z[np.arange(array.shape[0]), index]
    if reduction == "mean":
        return float(np.mean(nll))
    if reduction == "sum":
        return float(np.sum(nll))
    if reduction == "none":
        raise ValueError("reduction 'none' is not part of the M24 teaching contract")
    raise ValueError(f"unknown reduction {reduction!r}")


def softmax_nll_logit_gradient(probabilities, targets, *, reduction: str = DEFAULT_REDUCTION):
    """dL/dlogits = (p - one_hot) / N for mean reduction, else (p - one_hot)."""

    np = _require_numpy()
    probs = _as_2d(probabilities)
    index = np.asarray(targets, dtype=int).reshape(-1)
    if index.shape[0] != probs.shape[0]:
        raise ValueError("targets must have one class index per example")
    _require_finite(probs, "probabilities")
    grad = probs.copy()
    grad[np.arange(probs.shape[0]), index] -= 1.0
    if reduction == "mean":
        grad = grad / float(probs.shape[0])
    elif reduction != "sum":
        raise ValueError(f"unknown reduction {reduction!r}")
    return grad


def affine_backward(X, W, d_preactivation):
    """Reverse of Z = X @ W + b for a row-batch affine map."""

    np = _require_numpy()
    batch = _as_2d(X)
    weights = _as_2d(W)
    incoming = np.asarray(d_preactivation, dtype=float)
    if incoming.ndim == 1:
        incoming = incoming.reshape(1, -1)
    if batch.shape[0] != incoming.shape[0]:
        raise ValueError("d_preactivation batch does not match X")
    if weights.shape[1] != incoming.shape[1]:
        raise ValueError("d_preactivation width does not match W columns")
    d_W = batch.T @ incoming
    d_b = incoming.sum(axis=0)
    d_X = incoming @ weights.T
    return d_X, d_W, d_b


def hidden_branch_contributions(d_logits_row, weights2_row) -> tuple[float, ...]:
    """Per-class contributions to one hidden unit: they must sum to dL/dh."""

    incoming = tuple(float(value) for value in _as_1d(d_logits_row))
    weights = tuple(float(value) for value in _as_1d(weights2_row))
    if len(incoming) != len(weights):
        raise ValueError("logit gradient and W2 row must have the same width")
    return tuple(incoming[index] * weights[index] for index in range(len(incoming)))


def two_layer_loss(
    X,
    W1,
    b1,
    W2,
    b2,
    targets=TEACHING_TARGETS,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
) -> float:
    m23 = _load_m23()
    trace = m23.two_layer_forward(
        X, W1, b1, W2, b2, hidden_activation=hidden_activation
    )
    return softmax_nll(trace.logits, targets, reduction=reduction)


def two_layer_backward(
    X,
    W1,
    b1,
    W2,
    b2,
    targets=TEACHING_TARGETS,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
    defect: str = "none",
):
    """Reverse pass on the M23 named graph. Defects change one local rule only."""

    np = _require_numpy()
    m23 = _load_m23()
    key = _normalize_defect(defect)
    forward = m23.two_layer_forward(
        X, W1, b1, W2, b2, hidden_activation=hidden_activation
    )
    loss = softmax_nll(forward.logits, targets, reduction=reduction)
    d_logits = softmax_nll_logit_gradient(
        forward.probabilities, targets, reduction=reduction
    )
    weights2 = _as_2d(W2)
    if key == "omitted_branch":
        # Each hidden unit feeds every class logit; omitting classes 1..K-1
        # drops those downstream contributions from dL/dH.
        d_hidden = d_logits[:, :1] @ weights2[:, :1].T
    else:
        d_hidden = d_logits @ weights2.T
    if key == "wrong_relu_derivative":
        d_pre = np.asarray(d_hidden, dtype=float)
    else:
        d_pre = np.asarray(d_hidden, dtype=float) * activation_local_derivative(
            forward.hidden_preactivation, hidden_activation
        )
    d_x, d_W1, d_b1 = affine_backward(forward.x, W1, d_pre)
    _, d_W2, d_b2 = affine_backward(forward.hidden_activation, W2, d_logits)
    shapes = {
        "d_logits": tuple(int(size) for size in np.asarray(d_logits).shape),
        "d_hidden_activation": tuple(int(size) for size in np.asarray(d_hidden).shape),
        "d_hidden_preactivation": tuple(int(size) for size in np.asarray(d_pre).shape),
        "d_W1": tuple(int(size) for size in np.asarray(d_W1).shape),
        "d_b1": tuple(int(size) for size in np.asarray(d_b1).shape),
        "d_W2": tuple(int(size) for size in np.asarray(d_W2).shape),
        "d_b2": tuple(int(size) for size in np.asarray(d_b2).shape),
    }
    return BackwardTrace(
        loss=loss,
        targets=tuple(int(value) for value in np.asarray(targets, dtype=int).reshape(-1)),
        probabilities=forward.probabilities,
        d_logits=d_logits,
        d_hidden_activation=d_hidden,
        d_hidden_preactivation=d_pre,
        d_x=d_x,
        d_W1=d_W1,
        d_b1=d_b1,
        d_W2=d_W2,
        d_b2=d_b2,
        shapes=shapes,
        defect=key,
        reduction=reduction,
        loss_name=DEFAULT_LOSS,
    )


def _parameter_payload(name: str, W1, b1, W2, b2):
    mapping = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    if name not in mapping:
        raise ValueError(f"unknown parameter {name!r}; use one of {PARAMETER_NAMES}")
    return mapping[name]


def _set_parameter_entry(array, index, value):
    np = _require_numpy()
    out = np.asarray(array, dtype=float).copy()
    out[index] = float(value)
    return out


def network_parameter_value(W1, b1, W2, b2, name: str, index) -> float:
    np = _require_numpy()
    payload = _parameter_payload(name, W1, b1, W2, b2)
    return float(np.asarray(payload, dtype=float)[index])


def network_loss_at_parameter(
    X,
    W1,
    b1,
    W2,
    b2,
    targets,
    name: str,
    index,
    value: float,
    *,
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
) -> float:
    W1c, b1c, W2c, b2c = _clone_params(W1, b1, W2, b2)
    replacements = {"W1": W1c, "b1": b1c, "W2": W2c, "b2": b2c}
    replacements[name] = _set_parameter_entry(replacements[name], index, value)
    return two_layer_loss(
        X,
        replacements["W1"],
        replacements["b1"],
        replacements["W2"],
        replacements["b2"],
        targets,
        hidden_activation=hidden_activation,
        reduction=reduction,
    )


def network_finite_difference(
    X,
    W1,
    b1,
    W2,
    b2,
    targets=TEACHING_TARGETS,
    *,
    name: str = SIGN_PARAMETER,
    index=SIGN_INDEX,
    epsilon: float = DEFAULT_EPSILON,
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
) -> float:
    current = network_parameter_value(W1, b1, W2, b2, name, index)

    def loss_at(theta: float) -> float:
        return network_loss_at_parameter(
            X,
            W1,
            b1,
            W2,
            b2,
            targets,
            name,
            index,
            theta,
            hidden_activation=hidden_activation,
            reduction=reduction,
        )

    return central_finite_difference(loss_at, current, epsilon)


def perturb_parameter_loss(
    X,
    W1,
    b1,
    W2,
    b2,
    targets=TEACHING_TARGETS,
    *,
    name: str = SIGN_PARAMETER,
    index=SIGN_INDEX,
    delta: float = SIGN_DELTA,
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
) -> dict[str, float]:
    """Loss at theta-delta, theta, theta+delta for a gradient-sign prediction."""

    current = network_parameter_value(W1, b1, W2, b2, name, index)
    loss_center = network_loss_at_parameter(
        X, W1, b1, W2, b2, targets, name, index, current,
        hidden_activation=hidden_activation, reduction=reduction,
    )
    loss_plus = network_loss_at_parameter(
        X, W1, b1, W2, b2, targets, name, index, current + float(delta),
        hidden_activation=hidden_activation, reduction=reduction,
    )
    loss_minus = network_loss_at_parameter(
        X, W1, b1, W2, b2, targets, name, index, current - float(delta),
        hidden_activation=hidden_activation, reduction=reduction,
    )
    return {
        "value": current,
        "delta": float(delta),
        "loss_minus": loss_minus,
        "loss_center": loss_center,
        "loss_plus": loss_plus,
        "predicted_sign": float(
            0.0
            if loss_plus == loss_minus
            else (1.0 if loss_plus > loss_minus else -1.0)
        ),
    }


def check_parameter_gradients(
    X,
    W1,
    b1,
    W2,
    b2,
    targets=TEACHING_TARGETS,
    *,
    epsilon: float = DEFAULT_EPSILON,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    defect: str = "none",
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
    skip_relu_kinks: bool = False,
) -> tuple[FiniteDifferenceReport, ...]:
    """Compare parameter entries against a central difference of the true loss."""

    np = _require_numpy()
    m23 = _load_m23()
    forward = m23.two_layer_forward(
        X, W1, b1, W2, b2, hidden_activation=hidden_activation
    )
    skip = set()
    if skip_relu_kinks and str(hidden_activation).lower() == "relu":
        skip = set(relu_kink_parameter_entries(X, forward.hidden_preactivation))
    backward = two_layer_backward(
        X,
        W1,
        b1,
        W2,
        b2,
        targets,
        hidden_activation=hidden_activation,
        reduction=reduction,
        defect=defect,
    )
    reports: list[FiniteDifferenceReport] = []
    analytic = backward.parameter_grads()
    for name in PARAMETER_NAMES:
        array = np.asarray(analytic[name], dtype=float)
        for index in np.ndindex(array.shape):
            if (name, tuple(int(part) for part in index)) in skip:
                continue
            value = network_parameter_value(W1, b1, W2, b2, name, index)
            report = finite_difference_report(
                lambda theta, name=name, index=index: network_loss_at_parameter(
                    X,
                    W1,
                    b1,
                    W2,
                    b2,
                    targets,
                    name,
                    index,
                    theta,
                    hidden_activation=hidden_activation,
                    reduction=reduction,
                ),
                value,
                float(array[index]),
                epsilon=epsilon,
                name=f"{name}{index}",
                atol=atol,
                rtol=rtol,
            )
            reports.append(report)
    return tuple(reports)


def first_mismatch(
    reports: Sequence[FiniteDifferenceReport],
) -> FiniteDifferenceReport | None:
    for report in reports:
        if not report.agrees:
            return report
    return None


def one_step_update(
    X,
    W1,
    b1,
    W2,
    b2,
    targets=TEACHING_TARGETS,
    *,
    name: str = ONE_STEP_PARAMETER,
    index=ONE_STEP_INDEX,
    learning_rate: float = ONE_STEP_LEARNING_RATE,
    hidden_activation: str = HIDDEN_ACTIVATION,
    reduction: str = DEFAULT_REDUCTION,
) -> dict[str, float | str | tuple]:
    """One declared parameter update. Local loss movement, not convergence."""

    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    backward = two_layer_backward(
        X,
        W1,
        b1,
        W2,
        b2,
        targets,
        hidden_activation=hidden_activation,
        reduction=reduction,
        defect="none",
    )
    np = _require_numpy()
    gradient = float(np.asarray(backward.parameter_grads()[name], dtype=float)[index])
    value_before = network_parameter_value(W1, b1, W2, b2, name, index)
    value_after = value_before - learning_rate * gradient
    loss_after = network_loss_at_parameter(
        X,
        W1,
        b1,
        W2,
        b2,
        targets,
        name,
        index,
        value_after,
        hidden_activation=hidden_activation,
        reduction=reduction,
    )
    return {
        "parameter": name,
        "index": tuple(int(part) for part in index),
        "value_before": value_before,
        "value_after": value_after,
        "gradient": gradient,
        "learning_rate": float(learning_rate),
        "loss_before": float(backward.loss),
        "loss_after": float(loss_after),
        "update_convention": "parameter - learning_rate * gradient",
    }


def reference_forward():
    return _load_m23().reference_forward()


def reference_backward(
    targets=TEACHING_TARGETS,
    *,
    reduction: str = DEFAULT_REDUCTION,
    defect: str = "none",
) -> BackwardTrace:
    """Trusted reverse pass on the M23 teaching fixture. Handoff to M25."""

    m23 = _load_m23()
    return two_layer_backward(
        m23.REFERENCE_X,
        m23.REFERENCE_W1,
        m23.REFERENCE_B1,
        m23.REFERENCE_W2,
        m23.REFERENCE_B2,
        targets,
        hidden_activation=m23.HIDDEN_ACTIVATION,
        reduction=reduction,
        defect=defect,
    )


def backward_report(trace: BackwardTrace) -> dict[str, object]:
    """Compact observable evidence for one reverse pass."""

    return {
        "loss": trace.loss,
        "targets": trace.targets,
        "defect": trace.defect,
        "reduction": trace.reduction,
        "loss_name": trace.loss_name,
        "shapes": dict(trace.shapes),
        "d_logits": _nested_tuples(trace.d_logits),
        "d_hidden_activation": _nested_tuples(trace.d_hidden_activation),
        "d_hidden_preactivation": _nested_tuples(trace.d_hidden_preactivation),
        "d_W1": _nested_tuples(trace.d_W1),
        "d_b1": _nested_tuples(trace.d_b1),
        "d_W2": _nested_tuples(trace.d_W2),
        "d_b2": _nested_tuples(trace.d_b2),
    }


# Bind trusted M23 fixtures after helpers exist. Import is lazy-safe for numpy.
_M23 = _load_m23()
REFERENCE_X = _M23.REFERENCE_X
REFERENCE_W1 = _M23.REFERENCE_W1
REFERENCE_B1 = _M23.REFERENCE_B1
REFERENCE_W2 = _M23.REFERENCE_W2
REFERENCE_B2 = _M23.REFERENCE_B2
REFERENCE_HIDDEN_PREACTIVATION = _M23.REFERENCE_HIDDEN_PREACTIVATION
REFERENCE_HIDDEN_ACTIVATION = _M23.REFERENCE_HIDDEN_ACTIVATION
REFERENCE_LOGITS = _M23.REFERENCE_LOGITS
REFERENCE_PROBABILITIES = _M23.REFERENCE_PROBABILITIES
M23_GRAPH_NODES = _M23.GRAPH_NODES
CLASS_AXIS = _M23.CLASS_AXIS
GRAPH_NODES = FORWARD_GRAPH_NODES
two_layer_forward = _M23.two_layer_forward
scalar_two_layer_one_example = _M23.scalar_two_layer_one_example
