"""Transparent transformer block for Mission M30.

M30 composes M29 single-head attention into a teaching block: parallel
head projections, split/merge, output projection, residual adds,
a declared pre-norm convention, and a position-wise feed-forward.

LLM training objectives, contamination, and inference/adaptation stay
deferred to M31-M32. Heads do not receive human-language job titles.
No loss, no optimizer, no next-token pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

DEFAULT_ATOL = 1e-9
LN_EPS = 1e-5
ZERO_TOL = 1e-12
TEACHING_CONVENTION = "pre_norm"
TEACHING_ACTIVATION = "relu"
BLOCK_VERSION = "v06-teaching-block-1"
SUPPORTED_CONVENTIONS = ("pre_norm", "post_norm")
SUPPORTED_ACTIVATIONS = ("relu", "identity")
SUPPORTED_DEFECTS = (
    "none",
    "correct",
    "residual_wrong_branch",
    "norm_wrong_boundary",
)
SUPPORTED_SKIPS = (None, "attn", "ffn")
TRACE_CHECKPOINTS = (
    "x",
    "attn_norm",
    "q",
    "k",
    "v",
    "q_heads",
    "k_heads",
    "v_heads",
    "head_weights",
    "head_outputs",
    "attn_concat",
    "attn_projected",
    "attn_add",
    "attn_residual",
    "ffn_norm",
    "ffn_hidden",
    "ffn_projected",
    "ffn_add",
    "ffn_residual",
    "output",
)

D_MODEL = 4
N_HEADS = 2
D_HEAD = 2
D_FF = 8
SEQ_LEN = 3
BANK_INDEX = 1
CONTEXT_INDEX = 2

# First two coordinates match M29 cash/water; last two are a second subspace.
TOKEN_RIVER = (2.0, 0.0, 1.0, 0.0)
TOKEN_BANK = (1.0, 1.0, 1.0, 0.0)
TOKEN_CASH = (0.0, 2.0, 0.0, 1.0)
TOKEN_WATER = (3.0, 0.0, 2.0, 0.0)
CASH_TOKENS = ("river", "bank", "cash")
WATER_TOKENS = ("river", "bank", "water")
X_CASH_CONTEXT = (TOKEN_RIVER, TOKEN_BANK, TOKEN_CASH)
X_WATER_CONTEXT = (TOKEN_RIVER, TOKEN_BANK, TOKEN_WATER)

IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
TEACHING_W_Q = IDENTITY_4
TEACHING_W_K = IDENTITY_4
TEACHING_W_V = IDENTITY_4
TEACHING_W_O = IDENTITY_4
TEACHING_W_O_HALF = (
    (0.5, 0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.0),
    (0.0, 0.0, 0.0, 0.5),
)
# Expand (4, 8): [I_4 | 0.5 I_4]. Project (8, 4): stack(I_4, 0.5 I_4).
TEACHING_W_FF1 = (
    (1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5),
)
TEACHING_B_FF1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TEACHING_W_FF2 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.0),
    (0.0, 0.0, 0.0, 0.5),
)
TEACHING_B_FF2 = (0.0, 0.0, 0.0, 0.0)
TEACHING_LN_GAMMA = (1.0, 1.0, 1.0, 1.0)
TEACHING_LN_BETA = (0.0, 0.0, 0.0, 0.0)

# Hand-computable micro-cases (independent of the 4-D teaching sequence).
RESIDUAL_STREAM = (1.0, 0.0, 2.0)
RESIDUAL_SUBLAYER = (0.0, 1.0, -1.0)
RESIDUAL_SUM = (1.0, 1.0, 1.0)
LN_VECTOR = (3.0, 1.0)
LN_MEAN = 2.0
LN_VAR = 1.0
LN_NORMALIZED = (1.0, -1.0)
FFN_X = (1.0, -2.0)
FFN_W1 = (
    (1.0, 0.0),
    (0.0, 1.0),
)
FFN_B1 = (0.0, 0.0)
FFN_HIDDEN = (1.0, 0.0)
FFN_W2 = (
    (1.0, 0.0),
    (0.0, 1.0),
)
FFN_B2 = (0.0, 0.0)
FFN_OUT = (1.0, 0.0)

# Fresh no-AI numbers; answers are not stored here.
TRANSFER_X = (2.0, -1.0, 0.5)
TRANSFER_SUBLAYER = (-0.5, 1.5, 0.0)
TRANSFER_LN = (0.0, 4.0)
TRANSFER_SHAPES = {
    "batch": 2,
    "seq": 5,
    "d_model": 8,
    "n_heads": 4,
    "d_head": 2,
    "d_ff": 16,
}

PRE_NORM_DIAGRAM = (
    "x\n"
    "  |- LN -> MHA -> attn_projected\n"
    "  +---------------------------- attn_add / attn_residual\n"
    "       |- LN -> FFN -> ffn_projected\n"
    "       +---------------------- ffn_add / output\n"
)
POST_NORM_DIAGRAM = (
    "x -> MHA -> attn_projected\n"
    "x + attn_projected -> LN -> attn_residual\n"
    "attn_residual -> FFN -> ffn_projected\n"
    "attn_residual + ffn_projected -> LN -> output\n"
)

# Frozen teaching-block checkpoints (cash, identity projections, pre-norm).
GOLDEN_CASH_ATTN_RESIDUAL = (
    (3.358275057656, -0.686883233830, 1.322222389650, -1.239753902486),
    (1.407730563841, 1.407730563841, 1.463052091447, -1.468860272845),
    (-0.686883233830, 3.358275057656, -0.405072991583, 0.715903884279),
)
GOLDEN_CASH_OUTPUT = (
    (5.199624208349, -0.686883233830, 1.759320346733, -1.239753902486),
    (2.110914282953, 2.110914282953, 2.221389988722, -1.468860272845),
    (-0.686883233830, 5.403154841144, -0.405072991583, 0.715903884279),
)

HEAD_INTERPRETATION_LIMIT = (
    "Heads are parallel learned projections. They do not come with human "
    "job titles (syntax, anaphora, sentiment) unless a later measurement "
    "justifies a claim."
)
RESIDUAL_INTERPRETATION_LIMIT = (
    "A residual is an elementwise add of matching shapes: stream + sublayer. "
    "It is not a training-stability theorem and not a second attention path."
)
NORM_INTERPRETATION_LIMIT = (
    "LayerNorm standardizes the last axis at each position, then applies an "
    "affine map. Pre-norm versus post-norm is an architecture convention, "
    "not a universal law."
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M30 requires numpy; install requirements/m30.txt") from exc
    return np


def _load_m29():
    """Load the trusted M29 attention core without rewriting softmax."""

    name = "_learningos_m29_attention_core"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent.parent / "M29" / "attention_core.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load missions/M29/attention_core.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_finite(array, name: str) -> None:
    np = _require_numpy()
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")


def as_sequence(values, *, name: str = "tensor", last_dim: int | None = None):
    return _load_m29().as_sequence(values, name=name, last_dim=last_dim)


def as_weight_matrix(values, *, name: str, in_dim: int):
    return _load_m29().as_weight_matrix(values, name=name, in_dim=in_dim)


def as_vector(values, *, name: str, width: int):
    np = _require_numpy()
    array = np.asarray(values, dtype=float).reshape(-1)
    if int(array.size) != int(width):
        raise ValueError(f"{name} has {array.size} entries; expected {width}")
    _require_finite(array, name)
    return array


def nested_tuples(values) -> tuple:
    np = _require_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return (float(array),)
    if array.ndim == 1:
        return tuple(float(value) for value in array)
    return tuple(nested_tuples(row) for row in array)


def arrays_close(left, right, *, atol: float = DEFAULT_ATOL, rtol: float = 0.0) -> bool:
    np = _require_numpy()
    return bool(np.allclose(left, right, atol=atol, rtol=rtol))


def identity_matrix(width: int):
    return _load_m29().identity_matrix(width)


@dataclass(frozen=True)
class BlockParams:
    w_q: Any
    w_k: Any
    w_v: Any
    w_o: Any
    w_ff1: Any
    b_ff1: Any
    w_ff2: Any
    b_ff2: Any
    ln1_gamma: Any
    ln1_beta: Any
    ln2_gamma: Any
    ln2_beta: Any
    n_heads: int
    d_model: int
    d_head: int
    d_ff: int
    ln_eps: float = LN_EPS
    activation: str = TEACHING_ACTIVATION
    version: str = BLOCK_VERSION

    @property
    def d_inner(self) -> int:
        return int(self.n_heads) * int(self.d_head)


def teaching_params(*, w_o=None, activation: str = TEACHING_ACTIVATION) -> BlockParams:
    activation_key = _normalize_activation(activation)
    return BlockParams(
        w_q=TEACHING_W_Q,
        w_k=TEACHING_W_K,
        w_v=TEACHING_W_V,
        w_o=TEACHING_W_O if w_o is None else w_o,
        w_ff1=TEACHING_W_FF1,
        b_ff1=TEACHING_B_FF1,
        w_ff2=TEACHING_W_FF2,
        b_ff2=TEACHING_B_FF2,
        ln1_gamma=TEACHING_LN_GAMMA,
        ln1_beta=TEACHING_LN_BETA,
        ln2_gamma=TEACHING_LN_GAMMA,
        ln2_beta=TEACHING_LN_BETA,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_ff=D_FF,
        ln_eps=LN_EPS,
        activation=activation_key,
        version=BLOCK_VERSION,
    )


def params_with_activation(params: BlockParams, activation: str) -> BlockParams:
    return BlockParams(
        w_q=params.w_q,
        w_k=params.w_k,
        w_v=params.w_v,
        w_o=params.w_o,
        w_ff1=params.w_ff1,
        b_ff1=params.b_ff1,
        w_ff2=params.w_ff2,
        b_ff2=params.b_ff2,
        ln1_gamma=params.ln1_gamma,
        ln1_beta=params.ln1_beta,
        ln2_gamma=params.ln2_gamma,
        ln2_beta=params.ln2_beta,
        n_heads=params.n_heads,
        d_model=params.d_model,
        d_head=params.d_head,
        d_ff=params.d_ff,
        ln_eps=params.ln_eps,
        activation=_normalize_activation(activation),
        version=params.version,
    )


def params_with_output_projection(params: BlockParams, w_o) -> BlockParams:
    return BlockParams(
        w_q=params.w_q,
        w_k=params.w_k,
        w_v=params.w_v,
        w_o=w_o,
        w_ff1=params.w_ff1,
        b_ff1=params.b_ff1,
        w_ff2=params.w_ff2,
        b_ff2=params.b_ff2,
        ln1_gamma=params.ln1_gamma,
        ln1_beta=params.ln1_beta,
        ln2_gamma=params.ln2_gamma,
        ln2_beta=params.ln2_beta,
        n_heads=params.n_heads,
        d_model=params.d_model,
        d_head=params.d_head,
        d_ff=params.d_ff,
        ln_eps=params.ln_eps,
        activation=params.activation,
        version=params.version,
    )


def _normalize_activation(activation: str | None) -> str:
    key = TEACHING_ACTIVATION if activation is None else str(activation).lower()
    if key not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"unsupported activation {activation!r}; use one of {SUPPORTED_ACTIVATIONS}"
        )
    return key


def _normalize_convention(convention: str | None) -> str:
    key = TEACHING_CONVENTION if convention is None else str(convention).lower()
    if key in {"pre", "prenorm", "pre-ln", "pre_ln"}:
        key = "pre_norm"
    if key in {"post", "postnorm", "post-ln", "post_ln"}:
        key = "post_norm"
    if key not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"unsupported convention {convention!r}; use one of {SUPPORTED_CONVENTIONS}"
        )
    return key


def _normalize_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none", "correct"}:
        return "none"
    if key not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return key


def _normalize_skip(skip_residual: str | None) -> str | None:
    if skip_residual is None or skip_residual == "" or skip_residual == "none":
        return None
    key = str(skip_residual).lower()
    if key not in {"attn", "ffn"}:
        raise ValueError("skip_residual must be None, 'attn', or 'ffn'")
    return key


def residual_add(stream, sublayer, *, name: str = "residual"):
    """Elementwise add of matching residual-stream tensors."""

    np = _require_numpy()
    left = np.asarray(stream, dtype=float)
    right = np.asarray(sublayer, dtype=float)
    _require_finite(left, f"{name} stream")
    _require_finite(right, f"{name} sublayer")
    if left.shape != right.shape:
        raise ValueError(
            f"{name} shapes {left.shape} and {right.shape} must match for an add"
        )
    return left + right


def layer_norm(values, gamma=None, beta=None, eps: float = LN_EPS, *, name: str = "x"):
    """Per-position LayerNorm over the last axis, then affine gamma/beta."""

    np = _require_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim < 1:
        raise ValueError(f"{name} must have a feature axis")
    _require_finite(array, name)
    width = int(array.shape[-1])
    if width < 1:
        raise ValueError(f"{name} feature width must be positive")
    mean = array.mean(axis=-1, keepdims=True)
    centered = array - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    scale = np.sqrt(var + float(eps))
    normalized = centered / scale
    if gamma is None:
        gamma_vec = np.ones((width,), dtype=float)
    else:
        gamma_vec = as_vector(gamma, name=f"{name} gamma", width=width)
    if beta is None:
        beta_vec = np.zeros((width,), dtype=float)
    else:
        beta_vec = as_vector(beta, name=f"{name} beta", width=width)
    return normalized * gamma_vec + beta_vec


def split_heads(tensor, n_heads: int, *, name: str = "tensor"):
    """(batch, seq, n_heads * d_head) -> (batch, seq, n_heads, d_head)."""

    np = _require_numpy()
    array = np.asarray(tensor, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (batch, seq, width); got {array.shape}")
    heads = int(n_heads)
    if heads < 1:
        raise ValueError("n_heads must be positive")
    width = int(array.shape[-1])
    if width % heads != 0:
        raise ValueError(f"{name} last dim {width} is not divisible by n_heads={heads}")
    d_head = width // heads
    return array.reshape(array.shape[0], array.shape[1], heads, d_head)


def merge_heads(tensor, *, name: str = "heads"):
    """(batch, seq, n_heads, d_head) -> (batch, seq, n_heads * d_head)."""

    np = _require_numpy()
    array = np.asarray(tensor, dtype=float)
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, seq, n_heads, d_head); got {array.shape}")
    batch, seq, heads, d_head = array.shape
    return array.reshape(batch, seq, int(heads) * int(d_head))


def apply_activation(values, activation: str = TEACHING_ACTIVATION):
    np = _require_numpy()
    array = np.asarray(values, dtype=float)
    key = _normalize_activation(activation)
    if key == "relu":
        return np.maximum(array, 0.0)
    return array.copy()


def feed_forward(values, w1, b1, w2, b2, *, activation: str = TEACHING_ACTIVATION):
    """Position-wise MLP: same weights at every token, no mixing across positions."""

    sequence = as_sequence(values, name="ffn input")
    hidden_w = as_weight_matrix(w1, name="W_ff1", in_dim=sequence.shape[-1])
    project_w = as_weight_matrix(w2, name="W_ff2", in_dim=hidden_w.shape[1])
    bias1 = as_vector(b1, name="b_ff1", width=hidden_w.shape[1])
    bias2 = as_vector(b2, name="b_ff2", width=project_w.shape[1])
    hidden_pre = sequence @ hidden_w + bias1
    hidden = apply_activation(hidden_pre, activation)
    projected = hidden @ project_w + bias2
    return hidden_pre, hidden, projected


def _heads_to_attention_batch(heads):
    """(batch, seq, n_heads, d) -> (batch * n_heads, seq, d) for M29."""

    np = _require_numpy()
    array = np.asarray(heads, dtype=float)
    batch, seq, n_heads, width = array.shape
    moved = np.transpose(array, (0, 2, 1, 3))
    return moved.reshape(batch * n_heads, seq, width), batch, n_heads, seq, width


def _attention_batch_to_heads(tensor, batch: int, n_heads: int, seq: int, width: int):
    np = _require_numpy()
    array = np.asarray(tensor, dtype=float).reshape(batch, n_heads, seq, width)
    return np.transpose(array, (0, 2, 1, 3))


def _expand_mask_for_heads(mask, batch: int, n_heads: int, n_queries: int, n_keys: int):
    if mask is None:
        return None
    np = _require_numpy()
    m29 = _load_m29()
    scores_shape = (batch * n_heads, n_queries, n_keys)
    dummy = np.zeros(scores_shape, dtype=float)
    return m29._expand_mask(mask, dummy)


@dataclass(frozen=True)
class MultiHeadTrace:
    q: Any
    k: Any
    v: Any
    q_heads: Any
    k_heads: Any
    v_heads: Any
    head_weights: Any
    head_outputs: Any
    attn_concat: Any
    attn_projected: Any
    attention: Any
    n_heads: int
    d_head: int
    shapes: dict[str, tuple[int, ...]]

    def head_slice(self, index: int):
        head = int(index)
        if head < 0 or head >= int(self.n_heads):
            raise IndexError(index)
        return {
            "q": self.q_heads[:, :, head, :],
            "k": self.k_heads[:, :, head, :],
            "v": self.v_heads[:, :, head, :],
            "weights": self.head_weights[:, head, :, :],
            "output": self.head_outputs[:, :, head, :],
        }


def multi_head_attention(x, params: BlockParams, *, mask=None) -> MultiHeadTrace:
    """Project, split heads, call M29 scaled_dot_product_attention, merge, apply W_O.

    Heads are a reshape of concatenated projections, not named roles.
    """

    m29 = _load_m29()
    sequence = as_sequence(x, name="X", last_dim=params.d_model)
    if int(params.n_heads) * int(params.d_head) != int(params.d_model):
        raise ValueError("teaching block requires d_model == n_heads * d_head")
    q = m29.project_sequence(sequence, params.w_q, name="Q")
    k = m29.project_sequence(sequence, params.w_k, name="K")
    v = m29.project_sequence(sequence, params.w_v, name="V")
    q_heads = split_heads(q, params.n_heads, name="Q")
    k_heads = split_heads(k, params.n_heads, name="K")
    v_heads = split_heads(v, params.n_heads, name="V")
    q_b, batch, n_heads, seq, d_head = _heads_to_attention_batch(q_heads)
    k_b, _, _, _, _ = _heads_to_attention_batch(k_heads)
    v_b, _, _, _, d_v = _heads_to_attention_batch(v_heads)
    head_mask = _expand_mask_for_heads(mask, batch, n_heads, seq, int(k_b.shape[1]))
    attention = m29.scaled_dot_product_attention(q_b, k_b, v_b, mask=head_mask, scale="dk")
    head_outputs = _attention_batch_to_heads(attention.output, batch, n_heads, seq, d_v)
    head_weights = attention.weights.reshape(batch, n_heads, seq, int(k_b.shape[1]))
    concat = merge_heads(head_outputs, name="head outputs")
    w_o = as_weight_matrix(params.w_o, name="W_O", in_dim=concat.shape[-1])
    if int(w_o.shape[1]) != int(params.d_model):
        raise ValueError("W_O must map concatenated heads back to d_model")
    projected = concat @ w_o
    shapes = {
        "q": tuple(int(dim) for dim in q.shape),
        "k": tuple(int(dim) for dim in k.shape),
        "v": tuple(int(dim) for dim in v.shape),
        "q_heads": tuple(int(dim) for dim in q_heads.shape),
        "head_weights": tuple(int(dim) for dim in head_weights.shape),
        "head_outputs": tuple(int(dim) for dim in head_outputs.shape),
        "attn_concat": tuple(int(dim) for dim in concat.shape),
        "attn_projected": tuple(int(dim) for dim in projected.shape),
    }
    return MultiHeadTrace(
        q=q,
        k=k,
        v=v,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        head_weights=head_weights,
        head_outputs=head_outputs,
        attn_concat=concat,
        attn_projected=projected,
        attention=attention,
        n_heads=int(params.n_heads),
        d_head=int(params.d_head),
        shapes=shapes,
    )


def _checkpoint_shapes(trace_fields: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    np = _require_numpy()
    shapes = {}
    for name, value in trace_fields.items():
        if value is None:
            continue
        array = np.asarray(value)
        shapes[name] = tuple(int(dim) for dim in array.shape)
    return shapes


@dataclass(frozen=True)
class BlockTrace:
    x: Any
    attn_norm: Any
    q: Any
    k: Any
    v: Any
    q_heads: Any
    k_heads: Any
    v_heads: Any
    head_weights: Any
    head_outputs: Any
    attn_concat: Any
    attn_projected: Any
    attn_add: Any
    attn_residual: Any
    ffn_norm: Any
    ffn_hidden: Any
    ffn_projected: Any
    ffn_add: Any
    ffn_residual: Any
    output: Any
    params: BlockParams
    convention: str
    declared_convention: str
    defect: str
    skip_residual: str | None
    mask: Any
    multi_head: MultiHeadTrace
    shapes: dict[str, tuple[int, ...]]
    tokens: tuple[str, ...] = ()

    def checkpoint(self, name: str):
        if name not in TRACE_CHECKPOINTS:
            raise KeyError(name)
        return getattr(self, name)

    def named(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in TRACE_CHECKPOINTS}


def transformer_block(
    x,
    params: BlockParams | None = None,
    *,
    mask=None,
    convention: str | None = None,
    defect: str | None = "none",
    skip_residual: str | None = None,
    tokens: tuple[str, ...] = (),
) -> BlockTrace:
    """Run one teaching transformer block with named checkpoints.

    Declared teaching convention is pre-norm:

        h = x + MHA(LN(x))
        y = h + FFN(LN(h))

    Post-norm is a labeled alternative, not a silent default:

        h = LN(x + MHA(x))
        y = LN(h + FFN(h))

    Named defects keep x and parameters fixed and change only residual
    addends or the first-norm boundary.
    """

    block_params = teaching_params() if params is None else params
    declared = _normalize_convention(convention)
    defect_key = _normalize_defect(defect)
    skip = _normalize_skip(skip_residual)
    if defect_key != "none" and skip is not None:
        raise ValueError("do not mix a named defect with residual ablation")
    sequence = as_sequence(x, name="X", last_dim=block_params.d_model)

    use_pre_attn_norm = True
    post_norm_after_residual = False
    if declared == "post_norm" and defect_key == "none":
        use_pre_attn_norm = False
        post_norm_after_residual = True
    if defect_key == "norm_wrong_boundary":
        # Declared pre-norm, but the first sublayer runs on raw x.
        use_pre_attn_norm = False
        post_norm_after_residual = False

    if use_pre_attn_norm:
        attn_norm = layer_norm(
            sequence,
            block_params.ln1_gamma,
            block_params.ln1_beta,
            block_params.ln_eps,
            name="attn_norm",
        )
    else:
        attn_norm = sequence.copy()

    mha = multi_head_attention(attn_norm, block_params, mask=mask)
    attn_projected = mha.attn_projected

    if skip == "attn":
        attn_add = attn_projected.copy()
    elif defect_key == "residual_wrong_branch":
        attn_add = residual_add(attn_norm, attn_projected, name="attn wrong-branch")
    else:
        attn_add = residual_add(sequence, attn_projected, name="attn")

    if post_norm_after_residual:
        attn_residual = layer_norm(
            attn_add,
            block_params.ln1_gamma,
            block_params.ln1_beta,
            block_params.ln_eps,
            name="post_attn_norm",
        )
        ffn_norm = attn_residual.copy()
    else:
        attn_residual = attn_add
        ffn_norm = layer_norm(
            attn_residual,
            block_params.ln2_gamma,
            block_params.ln2_beta,
            block_params.ln_eps,
            name="ffn_norm",
        )

    _, ffn_hidden, ffn_projected = feed_forward(
        ffn_norm,
        block_params.w_ff1,
        block_params.b_ff1,
        block_params.w_ff2,
        block_params.b_ff2,
        activation=block_params.activation,
    )
    if skip == "ffn":
        ffn_add = ffn_projected.copy()
    else:
        ffn_add = residual_add(attn_residual, ffn_projected, name="ffn")

    if post_norm_after_residual:
        output = layer_norm(
            ffn_add,
            block_params.ln2_gamma,
            block_params.ln2_beta,
            block_params.ln_eps,
            name="post_ffn_norm",
        )
    else:
        output = ffn_add

    fields = {
        "x": sequence,
        "attn_norm": attn_norm,
        "q": mha.q,
        "k": mha.k,
        "v": mha.v,
        "q_heads": mha.q_heads,
        "k_heads": mha.k_heads,
        "v_heads": mha.v_heads,
        "head_weights": mha.head_weights,
        "head_outputs": mha.head_outputs,
        "attn_concat": mha.attn_concat,
        "attn_projected": attn_projected,
        "attn_add": attn_add,
        "attn_residual": attn_residual,
        "ffn_norm": ffn_norm,
        "ffn_hidden": ffn_hidden,
        "ffn_projected": ffn_projected,
        "ffn_add": ffn_add,
        "ffn_residual": ffn_add,
        "output": output,
    }
    return BlockTrace(
        **fields,
        params=block_params,
        convention=declared,
        declared_convention=declared,
        defect=defect_key,
        skip_residual=skip,
        mask=None if mask is None else _load_m29()._expand_mask(
            mask,
            _require_numpy().zeros(
                (sequence.shape[0], sequence.shape[1], sequence.shape[1]),
                dtype=float,
            ),
        ),
        multi_head=mha,
        shapes=_checkpoint_shapes(fields),
        tokens=tuple(tokens),
    )


def independent_pre_norm_compose(x, params: BlockParams | None = None, *, mask=None):
    """Unrolled pre-norm graph using public primitives and numpy `+`.

    Residual adds are written as `stream + sublayer` here so a defect in
    `transformer_block`'s branch selection cannot hide inside this fixture.
    """

    np = _require_numpy()
    block_params = teaching_params() if params is None else params
    sequence = as_sequence(x, name="X", last_dim=block_params.d_model)
    attn_norm = layer_norm(
        sequence,
        block_params.ln1_gamma,
        block_params.ln1_beta,
        block_params.ln_eps,
        name="attn_norm",
    )
    mha = multi_head_attention(attn_norm, block_params, mask=mask)
    attn_add = np.asarray(sequence, dtype=float) + np.asarray(mha.attn_projected, dtype=float)
    attn_residual = attn_add
    ffn_norm = layer_norm(
        attn_residual,
        block_params.ln2_gamma,
        block_params.ln2_beta,
        block_params.ln_eps,
        name="ffn_norm",
    )
    _, ffn_hidden, ffn_projected = feed_forward(
        ffn_norm,
        block_params.w_ff1,
        block_params.b_ff1,
        block_params.w_ff2,
        block_params.b_ff2,
        activation=block_params.activation,
    )
    ffn_add = np.asarray(attn_residual, dtype=float) + np.asarray(ffn_projected, dtype=float)
    return {
        "x": sequence,
        "attn_norm": attn_norm,
        "attn_concat": mha.attn_concat,
        "attn_projected": mha.attn_projected,
        "attn_add": attn_add,
        "attn_residual": attn_residual,
        "ffn_norm": ffn_norm,
        "ffn_hidden": ffn_hidden,
        "ffn_projected": ffn_projected,
        "ffn_add": ffn_add,
        "output": ffn_add,
        "head_weights": mha.head_weights,
        "head_outputs": mha.head_outputs,
        "q_heads": mha.q_heads,
    }


def reference_pre_norm_block(
    x,
    params: BlockParams | None = None,
    *,
    mask=None,
    tokens: tuple[str, ...] = (),
) -> BlockTrace:
    """Trusted defect-free pre-norm teaching block (golden path)."""

    return transformer_block(
        x,
        params,
        mask=mask,
        convention="pre_norm",
        defect="none",
        tokens=tokens,
    )


def block_with_defect(
    x,
    params: BlockParams | None = None,
    *,
    mask=None,
    defect: str = "residual_wrong_branch",
    convention: str | None = None,
    tokens: tuple[str, ...] = (),
) -> BlockTrace:
    """Controlled-failure entry: one named defect, same x and parameters."""

    return transformer_block(
        x,
        params,
        mask=mask,
        convention=convention,
        defect=defect,
        tokens=tokens,
    )


def repair_block(trace: BlockTrace) -> BlockTrace:
    """Recompute from a defective trace's x, parameters, mask, and convention."""

    return transformer_block(
        trace.x,
        trace.params,
        mask=trace.mask,
        convention=trace.declared_convention,
        defect="none",
        skip_residual=None,
        tokens=trace.tokens,
    )


def ablate_residual(
    x,
    params: BlockParams | None = None,
    *,
    which: str = "attn",
    mask=None,
    convention: str | None = None,
    tokens: tuple[str, ...] = (),
) -> BlockTrace:
    """Controlled experiment: drop one residual add, keep parameters fixed."""

    return transformer_block(
        x,
        params,
        mask=mask,
        convention=convention,
        defect="none",
        skip_residual=which,
        tokens=tokens,
    )


def teaching_batch():
    np = _require_numpy()
    return np.stack(
        [
            np.asarray(X_CASH_CONTEXT, dtype=float),
            np.asarray(X_WATER_CONTEXT, dtype=float),
        ]
    )


def first_divergence(
    left: BlockTrace,
    right: BlockTrace,
    *,
    atol: float = DEFAULT_ATOL,
) -> str | None:
    """Return the first TRACE_CHECKPOINTS name that fails allclose."""

    np = _require_numpy()
    for name in TRACE_CHECKPOINTS:
        left_value = left.checkpoint(name)
        right_value = right.checkpoint(name)
        if not np.allclose(left_value, right_value, atol=atol, rtol=0.0):
            return name
    return None


def checkpoint_parity(
    left: BlockTrace,
    right: BlockTrace,
    *,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    diverged = first_divergence(left, right, atol=atol)
    return {
        "match": diverged is None,
        "first_divergence": diverged,
        "checkpoints": TRACE_CHECKPOINTS,
    }


def stream_l2(trace: BlockTrace) -> dict[str, Any]:
    np = _require_numpy()

    def norms(name: str):
        array = np.asarray(trace.checkpoint(name), dtype=float)
        return tuple(float(value) for value in np.linalg.norm(array, axis=-1).reshape(-1))

    return {
        "x": norms("x"),
        "attn_projected": norms("attn_projected"),
        "attn_residual": norms("attn_residual"),
        "ffn_projected": norms("ffn_projected"),
        "output": norms("output"),
    }


def observability_report(trace: BlockTrace) -> dict[str, Any]:
    parity_names = TRACE_CHECKPOINTS
    return {
        "version": trace.params.version,
        "declared_convention": trace.declared_convention,
        "defect": trace.defect,
        "skip_residual": trace.skip_residual,
        "activation": trace.params.activation,
        "n_heads": trace.params.n_heads,
        "d_model": trace.params.d_model,
        "d_head": trace.params.d_head,
        "d_ff": trace.params.d_ff,
        "ln_eps": trace.params.ln_eps,
        "shapes": dict(trace.shapes),
        "checkpoints": parity_names,
        "tokens": trace.tokens,
        "head_interpretation_limit": HEAD_INTERPRETATION_LIMIT,
        "residual_interpretation_limit": RESIDUAL_INTERPRETATION_LIMIT,
        "norm_interpretation_limit": NORM_INTERPRETATION_LIMIT,
        "handoff": (
            "M31 receives these named checkpoints and the declared convention; "
            "it does not receive a training loop from M30."
        ),
    }


def m29_head0_tokens(sequence_4d):
    """First two coordinates of a 4-D teaching sequence (M29 subspace)."""

    array = as_sequence(sequence_4d, name="X4")
    return array[:, :, :2]


def golden_pre_norm_trace(tokens: tuple[str, ...] = CASH_TOKENS) -> BlockTrace:
    return transformer_block(
        X_CASH_CONTEXT,
        teaching_params(),
        convention="pre_norm",
        defect="none",
        tokens=tokens,
    )
