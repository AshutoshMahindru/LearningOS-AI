"""Transparent single-head attention for Mission M29.

M29 traces queries, keys, values, scaled dots, masks, softmax over keys,
and weighted value aggregation. Multi-head split/merge, residuals,
normalization, and the feed-forward sublayer remain deferred to M30.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

KEY_AXIS = -1
QUERY_AXIS = -2
MASK_FILL = -1e9
DEFAULT_ATOL = 1e-9
ZERO_TOL = 1e-12
SUPPORTED_DEFECTS = (
    "none",
    "correct",
    "softmax_over_queries",
    "mask_after_softmax",
)
TRACE_CHECKPOINTS = (
    "q",
    "k",
    "v",
    "raw_scores",
    "scaled_scores",
    "masked_scores",
    "weights",
    "output",
)

# Two-key micro-case: q=[1,0], k0=[1,0], k1=[0,1], v0=[1,0], v1=[0,2].
# Unscaled scores are (1, 0); softmax is (e/(e+1), 1/(e+1)).
HAND_Q = ((1.0, 0.0),)
HAND_K = (
    (1.0, 0.0),
    (0.0, 1.0),
)
HAND_V = (
    (1.0, 0.0),
    (0.0, 2.0),
)
HAND_UNSCALED_SCORES = (1.0, 0.0)
HAND_UNSCALED_WEIGHTS = (math.e / (math.e + 1.0), 1.0 / (math.e + 1.0))
HAND_UNSCALED_OUTPUT = (HAND_UNSCALED_WEIGHTS[0], 2.0 * HAND_UNSCALED_WEIGHTS[1])

# Teaching self-attention: bank is geometrically ambiguous until context votes.
TOKEN_RIVER = (2.0, 0.0)
TOKEN_BANK = (1.0, 1.0)
TOKEN_CASH = (0.0, 2.0)
TOKEN_WATER = (3.0, 0.0)
CASH_TOKENS = ("river", "bank", "cash")
WATER_TOKENS = ("river", "bank", "water")
X_CASH_CONTEXT = (TOKEN_RIVER, TOKEN_BANK, TOKEN_CASH)
X_WATER_CONTEXT = (TOKEN_RIVER, TOKEN_BANK, TOKEN_WATER)
IDENTITY_2 = (
    (1.0, 0.0),
    (0.0, 1.0),
)
TEACHING_W_Q = (
    (1.0, 0.0),
    (0.0, 0.5),
)
TEACHING_W_K = (
    (0.5, 0.0),
    (0.0, 1.0),
)
TEACHING_W_V = IDENTITY_2
BANK_INDEX = 1
CONTEXT_INDEX = 2

# Large-dot scale fixture: aligned vs opposite halves in d_k=8.
SCALE_D_K = 8

# Fresh no-AI numbers; answers are not stored here.
TRANSFER_Q = (1.0, 2.0)
TRANSFER_K = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
)
TRANSFER_V = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
TRANSFER_QUERY_INDEX = 1

INTERPRETATION_LIMIT = (
    "Attention weights are not a causal explanation of intent, meaning, or what "
    "the model 'looked at' in a human sense; they are softmax-normalized scores."
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M29 requires numpy; install requirements/m29.txt") from exc
    return np


def _require_finite(array, name: str) -> None:
    np = _require_numpy()
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")


def as_sequence(values, *, name: str = "tensor", last_dim: int | None = None):
    """Return a float array with shape (batch, seq, dim)."""

    np = _require_numpy()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2:
        array = array.reshape(1, array.shape[0], array.shape[1])
    elif array.ndim != 3:
        raise ValueError(f"{name} must have rank 1, 2, or 3; got shape {array.shape}")
    if array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} must have nonempty sequence and feature axes")
    _require_finite(array, name)
    if last_dim is not None and int(array.shape[-1]) != int(last_dim):
        raise ValueError(f"{name} last dim must be {last_dim}; got {int(array.shape[-1])}")
    return array


def as_weight_matrix(values, *, name: str, in_dim: int):
    np = _require_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (in_dim, out_dim); got shape {array.shape}")
    if int(array.shape[0]) != int(in_dim):
        raise ValueError(f"{name} has {array.shape[0]} rows; expected {in_dim}")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have a positive output dimension")
    _require_finite(array, name)
    return array


def identity_matrix(width: int):
    np = _require_numpy()
    if int(width) < 1:
        raise ValueError("identity width must be positive")
    return np.eye(int(width), dtype=float)


def project_sequence(x, weight, *, name: str = "projection"):
    """Row-batch projection Y = X @ W with X (..., d_in) and W (d_in, d_out)."""

    sequence = as_sequence(x, name=name)
    matrix = as_weight_matrix(weight, name=f"{name} weight", in_dim=sequence.shape[-1])
    return sequence @ matrix


def project_qkv(x, w_q, w_k, w_v) -> tuple[Any, Any, Any]:
    """Project one sequence into Q, K, and V with three named matrices."""

    return (
        project_sequence(x, w_q, name="Q"),
        project_sequence(x, w_k, name="K"),
        project_sequence(x, w_v, name="V"),
    )


def resolve_scale(d_k: int, scale: float | str | None) -> float:
    width = int(d_k)
    if width < 1:
        raise ValueError("d_k must be positive")
    if scale is None or scale == "dk" or scale == "default":
        return 1.0 / math.sqrt(width)
    if scale in {"none", "unscaled", "raw"}:
        return 1.0
    value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be a positive finite number")
    return value


def dot_product_scores(q, k):
    """Return raw scores with shape (batch, queries, keys)."""

    queries = as_sequence(q, name="Q")
    keys = as_sequence(k, name="K")
    if queries.shape[0] != keys.shape[0]:
        raise ValueError(
            f"Q batch {queries.shape[0]} does not match K batch {keys.shape[0]}"
        )
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            f"Q dim {queries.shape[-1]} does not match K dim {keys.shape[-1]}"
        )
    return queries @ keys.transpose(0, 2, 1)


def scale_scores(scores, d_k: int, scale: float | str | None = "dk"):
    np = _require_numpy()
    array = np.asarray(scores, dtype=float)
    _require_finite(array, "scores")
    return array * resolve_scale(d_k, scale)


def causal_additive_mask(n_queries: int, n_keys: int | None = None):
    """Additive mask: 0 on and below the diagonal, MASK_FILL above it."""

    np = _require_numpy()
    queries = int(n_queries)
    keys = int(n_queries if n_keys is None else n_keys)
    if queries < 1 or keys < 1:
        raise ValueError("causal mask requires positive query and key counts")
    allowed = np.tril(np.ones((queries, keys), dtype=float), k=0)
    return np.where(allowed > 0.5, 0.0, MASK_FILL)


def padding_additive_mask(valid_keys):
    """Broadcastable key mask: True/1 keeps a key, False/0 blocks it."""

    np = _require_numpy()
    flags = np.asarray(valid_keys)
    if flags.ndim == 0:
        raise ValueError("padding mask must include a key axis")
    if flags.dtype == bool:
        keep = flags.astype(float)
    else:
        keep = (np.asarray(flags, dtype=float) > 0.5).astype(float)
    additive = np.where(keep > 0.5, 0.0, MASK_FILL)
    if additive.ndim == 1:
        return additive.reshape(1, 1, -1)
    if additive.ndim == 2:
        return additive.reshape(additive.shape[0], 1, additive.shape[1])
    return additive


def combine_additive_masks(*masks):
    np = _require_numpy()
    present = [np.asarray(mask, dtype=float) for mask in masks if mask is not None]
    if not present:
        return None
    combined = present[0]
    for mask in present[1:]:
        combined = combined + mask
    return combined


def _expand_mask(mask, scores):
    np = _require_numpy()
    additive = np.asarray(mask, dtype=float)
    if additive.shape == scores.shape:
        return additive
    try:
        return np.broadcast_to(additive, scores.shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"mask shape {additive.shape} cannot broadcast to scores {scores.shape}"
        ) from exc


def apply_additive_mask(scores, mask):
    np = _require_numpy()
    array = np.asarray(scores, dtype=float)
    if mask is None:
        return array.copy()
    return array + _expand_mask(mask, array)


def _softmax(array, axis: int):
    np = _require_numpy()
    if array.ndim == 0:
        raise ValueError("softmax requires at least a 1-D score vector")
    axis_index = axis if axis >= 0 else array.ndim + axis
    if axis_index < 0 or axis_index >= array.ndim:
        raise ValueError(f"softmax axis {axis} is out of range for shape {array.shape}")
    shifted = array - np.max(array, axis=axis, keepdims=True)
    unnormalized = np.exp(shifted)
    totals = np.sum(unnormalized, axis=axis, keepdims=True)
    if np.any(totals <= ZERO_TOL):
        raise ValueError("softmax encountered a zero partition function")
    return unnormalized / totals


def softmax_over_keys(scores):
    """Normalize each query over keys (last axis)."""

    np = _require_numpy()
    array = np.asarray(scores, dtype=float)
    _require_finite(array, "scores")
    return _softmax(array, axis=KEY_AXIS)


def softmax_over_queries(scores):
    """Defective normalize: softmax over the query axis instead of keys."""

    np = _require_numpy()
    array = np.asarray(scores, dtype=float)
    _require_finite(array, "scores")
    if array.ndim < 2:
        raise ValueError("query-axis softmax requires at least a 2-D score matrix")
    return _softmax(array, axis=QUERY_AXIS)


def aggregate_values(weights, values):
    """Weighted sum of values with weights (..., queries, keys)."""

    value_seq = as_sequence(values, name="V")
    np = _require_numpy()
    w = np.asarray(weights, dtype=float)
    if w.ndim == 2:
        w = w.reshape(1, w.shape[0], w.shape[1])
    if w.ndim != 3:
        raise ValueError(f"weights must have rank 2 or 3; got shape {w.shape}")
    if w.shape[0] != value_seq.shape[0]:
        raise ValueError("weight batch does not match V batch")
    if w.shape[-1] != value_seq.shape[1]:
        raise ValueError(
            f"weights cover {w.shape[-1]} keys but V has {value_seq.shape[1]} positions"
        )
    _require_finite(w, "weights")
    return w @ value_seq


def attention_shapes(q, k, v, scores, weights, output) -> dict[str, tuple[int, ...]]:
    return {
        "q": tuple(int(dim) for dim in as_sequence(q, name="Q").shape),
        "k": tuple(int(dim) for dim in as_sequence(k, name="K").shape),
        "v": tuple(int(dim) for dim in as_sequence(v, name="V").shape),
        "scores": tuple(int(dim) for dim in _require_numpy().asarray(scores).shape),
        "weights": tuple(int(dim) for dim in _require_numpy().asarray(weights).shape),
        "output": tuple(int(dim) for dim in as_sequence(output, name="output").shape),
    }


def _blocked_from_mask(mask, weights):
    np = _require_numpy()
    if mask is None:
        return np.zeros(np.asarray(weights).shape, dtype=bool)
    additive = _expand_mask(mask, np.asarray(weights, dtype=float))
    return additive <= (MASK_FILL / 2.0)


def weight_entropy(weights):
    np = _require_numpy()
    w = np.clip(np.asarray(weights, dtype=float), ZERO_TOL, 1.0)
    return -np.sum(w * np.log(w), axis=KEY_AXIS)


def weight_invariants(weights, *, mask=None, atol: float = DEFAULT_ATOL) -> dict[str, Any]:
    np = _require_numpy()
    w = np.asarray(weights, dtype=float)
    if w.ndim < 2:
        raise ValueError("weights must include a query axis and a key axis")
    row_sums = w.sum(axis=KEY_AXIS)
    blocked = _blocked_from_mask(mask, w)
    masked_mass = float(w[blocked].sum()) if blocked.any() else 0.0
    return {
        "row_sums": row_sums,
        "rows_sum_to_one": bool(np.allclose(row_sums, 1.0, atol=atol, rtol=0.0)),
        "nonnegative": bool(np.all(w >= -atol)),
        "masked_key_mass": masked_mass,
        "future_mass_zero": bool(masked_mass <= atol),
        "max_weight": float(np.max(w)),
        "entropy": weight_entropy(w),
    }


def _assert_unmasked_keys(masked_scores, mask) -> None:
    np = _require_numpy()
    if mask is None:
        return
    blocked = _blocked_from_mask(mask, masked_scores)
    if np.any(np.all(blocked, axis=KEY_AXIS)):
        raise ValueError("each query must have at least one unmasked key")


@dataclass(frozen=True)
class AttentionTrace:
    q: Any
    k: Any
    v: Any
    raw_scores: Any
    scaled_scores: Any
    mask: Any
    masked_scores: Any
    weights: Any
    output: Any
    scale: float
    d_k: int
    d_v: int
    softmax_axis: int
    mask_timing: str
    defect: str
    shapes: dict[str, tuple[int, ...]]
    tokens: tuple[str, ...] = ()

    @property
    def batch(self) -> int:
        return int(self.q.shape[0])

    @property
    def n_queries(self) -> int:
        return int(self.q.shape[1])

    @property
    def n_keys(self) -> int:
        return int(self.k.shape[1])

    def checkpoint(self, name: str):
        if name not in TRACE_CHECKPOINTS:
            raise KeyError(name)
        return getattr(self, name)

    def invariants(self, *, atol: float = DEFAULT_ATOL) -> dict[str, Any]:
        return weight_invariants(self.weights, mask=self.mask, atol=atol)


def _normalize_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none", "correct"}:
        return "none"
    if key not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return key


def scaled_dot_product_attention(
    q,
    k,
    v,
    *,
    mask=None,
    scale: float | str | None = "dk",
    defect: str | None = "none",
    tokens: tuple[str, ...] = (),
) -> AttentionTrace:
    """Single-head attention: softmax(scale * QK^T + mask) V.

    Softmax runs over keys. The named defects keep Q, K, V, and the intended
    mask fixed and change only the softmax axis or the mask/softmax order.
    """

    queries = as_sequence(q, name="Q")
    keys = as_sequence(k, name="K")
    values = as_sequence(v, name="V")
    if queries.shape[0] != values.shape[0] or keys.shape[0] != values.shape[0]:
        raise ValueError("Q, K, and V must share a batch dimension")
    if keys.shape[1] != values.shape[1]:
        raise ValueError("K and V must have the same sequence length")
    raw = dot_product_scores(queries, keys)
    d_k = int(queries.shape[-1])
    scale_value = resolve_scale(d_k, scale)
    scaled = raw * scale_value
    defect_key = _normalize_defect(defect)
    softmax_axis = KEY_AXIS
    mask_timing = "before_softmax"
    if defect_key == "softmax_over_queries":
        _assert_unmasked_keys(apply_additive_mask(scaled, mask), mask)
        masked_scores = apply_additive_mask(scaled, mask)
        weights = softmax_over_queries(masked_scores)
        softmax_axis = QUERY_AXIS
    elif defect_key == "mask_after_softmax":
        masked_scores = scaled.copy()
        weights = softmax_over_keys(scaled)
        if mask is not None:
            keep = (~_blocked_from_mask(mask, weights)).astype(float)
            weights = weights * keep
        mask_timing = "after_softmax"
    else:
        masked_scores = apply_additive_mask(scaled, mask)
        _assert_unmasked_keys(masked_scores, mask)
        weights = softmax_over_keys(masked_scores)
    output = aggregate_values(weights, values)
    shapes = attention_shapes(queries, keys, values, raw, weights, output)
    return AttentionTrace(
        q=queries,
        k=keys,
        v=values,
        raw_scores=raw,
        scaled_scores=scaled,
        mask=None if mask is None else _expand_mask(mask, scaled),
        masked_scores=masked_scores,
        weights=weights,
        output=output,
        scale=scale_value,
        d_k=d_k,
        d_v=int(values.shape[-1]),
        softmax_axis=softmax_axis,
        mask_timing=mask_timing,
        defect=defect_key,
        shapes=shapes,
        tokens=tuple(tokens),
    )


def attention_with_defect(
    q,
    k,
    v,
    *,
    mask=None,
    scale: float | str | None = "dk",
    defect: str = "softmax_over_queries",
    tokens: tuple[str, ...] = (),
) -> AttentionTrace:
    """Controlled-failure entry: one named defect, same Q/K/V/mask."""

    return scaled_dot_product_attention(
        q, k, v, mask=mask, scale=scale, defect=defect, tokens=tokens
    )


def repair_attention(trace: AttentionTrace) -> AttentionTrace:
    """Recompute from a defective trace's Q, K, V, mask, and scale."""

    return scaled_dot_product_attention(
        trace.q,
        trace.k,
        trace.v,
        mask=trace.mask,
        scale=trace.scale,
        defect="none",
        tokens=trace.tokens,
    )


def self_attention(
    x,
    w_q=None,
    w_k=None,
    w_v=None,
    *,
    mask=None,
    scale: float | str | None = "dk",
    defect: str | None = "none",
    tokens: tuple[str, ...] = (),
) -> AttentionTrace:
    """Project a token sequence, then run single-head attention."""

    sequence = as_sequence(x, name="X")
    width = int(sequence.shape[-1])
    query_w = identity_matrix(width) if w_q is None else w_q
    key_w = identity_matrix(width) if w_k is None else w_k
    value_w = identity_matrix(width) if w_v is None else w_v
    q, k, v = project_qkv(sequence, query_w, key_w, value_w)
    return scaled_dot_product_attention(
        q, k, v, mask=mask, scale=scale, defect=defect, tokens=tokens
    )


def replace_position(sequence, index: int, vector):
    np = _require_numpy()
    array = as_sequence(sequence, name="sequence").copy()
    replacement = np.asarray(vector, dtype=float).reshape(-1)
    if replacement.size != array.shape[-1]:
        raise ValueError(
            f"replacement has {replacement.size} dims; expected {array.shape[-1]}"
        )
    _require_finite(replacement, "replacement")
    position = int(index)
    if position < 0:
        position += int(array.shape[1])
    if position < 0 or position >= array.shape[1]:
        raise IndexError(index)
    array[:, position, :] = replacement
    return array


def teaching_batch():
    """Cash vs water contexts stacked as batch rows: shape (2, 3, 2)."""

    np = _require_numpy()
    return np.stack(
        [
            np.asarray(X_CASH_CONTEXT, dtype=float),
            np.asarray(X_WATER_CONTEXT, dtype=float),
        ]
    )


def scale_fixture_qkv():
    """Aligned vs opposite-half keys in d_k=8, used by the scale experiment."""

    np = _require_numpy()
    q = np.ones((1, 1, SCALE_D_K), dtype=float)
    aligned = np.ones((SCALE_D_K,), dtype=float)
    opposite = np.concatenate(
        [np.ones(SCALE_D_K // 2, dtype=float), -np.ones(SCALE_D_K // 2, dtype=float)]
    )
    k = np.stack([aligned, opposite]).reshape(1, 2, SCALE_D_K)
    v = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=float).reshape(1, 2, 2)
    return q, k, v


def observability_report(trace: AttentionTrace) -> dict[str, Any]:
    invariants = trace.invariants()
    return {
        "shapes": dict(trace.shapes),
        "d_k": trace.d_k,
        "d_v": trace.d_v,
        "scale": trace.scale,
        "softmax_axis": trace.softmax_axis,
        "mask_timing": trace.mask_timing,
        "defect": trace.defect,
        "n_queries": trace.n_queries,
        "n_keys": trace.n_keys,
        "row_sums": tuple(float(value) for value in np_ravel(invariants["row_sums"])),
        "rows_sum_to_one": invariants["rows_sum_to_one"],
        "masked_key_mass": invariants["masked_key_mass"],
        "future_mass_zero": invariants["future_mass_zero"],
        "max_weight": invariants["max_weight"],
        "entropy": tuple(float(value) for value in np_ravel(invariants["entropy"])),
        "interpretation_limit": INTERPRETATION_LIMIT,
        "checkpoints": TRACE_CHECKPOINTS,
        "tokens": trace.tokens,
    }


def np_ravel(values) -> tuple[float, ...]:
    np = _require_numpy()
    return tuple(float(value) for value in np.asarray(values, dtype=float).reshape(-1))
