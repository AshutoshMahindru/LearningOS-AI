"""Instrumented PyTorch training loop for Mission M25.

M25 wraps the trusted M24 two-layer graph as an ``nn.Module``, checks
autograd against reverse-mode numbers, and runs a CPU training loop:
zero_grad -> forward -> loss -> backward -> step. Train/eval mode,
no-grad inference, protected splits, checkpoints, and reproducibility
live here. Systematic multi-cause DL debugging remains deferred to M26.

Canonical path: local deterministic tensors. No weight download, no
CUDA requirement, no secrets.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import sys
from typing import Sequence


CANONICAL_DEVICE = "cpu"
CANONICAL_DTYPE_NAME = "float64"
DEFAULT_SEED = 2501
DEFAULT_SPLIT_SEED = 2501
DEFAULT_LEARNING_RATE = 0.25
TEACHING_LEARNING_RATE = 0.25
DEFAULT_MOMENTUM = 0.9
TEACHING_MOMENTUM = 0.0
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_DROPOUT_P = 0.0
MODE_DROPOUT_P = 0.5
N_IN = 3
N_HIDDEN = 2
TRAIN_HIDDEN = 8
N_OUT = 3
N_PER_CLASS = 12
TRAIN_PER_CLASS = 8
VAL_PER_CLASS = 2
HELD_OUT_PER_CLASS = 2
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5
PARITY_ATOL = 1e-10
PARITY_RTOL = 0.0
LOOP_ORDER = ("zero_grad", "forward", "loss", "backward", "step")
CHECKPOINT_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "global_step",
    "seed",
    "split_seed",
    "batch_size",
    "learning_rate",
    "momentum",
    "dropout_p",
    "n_in",
    "n_hidden",
    "n_out",
    "val_loss",
    "torch_rng_state",
    "policy",
)
SUPPORTED_DEFECTS = (
    "none",
    "correct",
    "stale_grad",
    "train_mode_eval",
)
SPLIT_NAMES = ("train", "val", "held_out")
PARAMETER_LAYOUT = "nn.Linear stores W as (out, in); M24 stores (in, out). Compare W.T."


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M25 requires torch; install requirements/m25.txt") from exc
    return torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_m24():
    """Import missions.M24.backprop_core as the trusted reverse-mode reference."""

    name = "_learningos_m24_backprop_core"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import missions.M24.backprop_core as module
    except ImportError:
        path = Path(__file__).resolve().parent.parent / "M24" / "backprop_core.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load missions/M24/backprop_core.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _normalize_defect(defect: str) -> str:
    key = str(defect).lower()
    if key in {"none", "correct", ""}:
        return "none"
    if key in {"stale_grad", "stale_gradient", "missing_zero_grad", "no_zero_grad"}:
        return "stale_grad"
    if key in {"train_mode_eval", "eval_in_train_mode", "dropout_eval"}:
        return "train_mode_eval"
    raise ValueError(f"unknown defect {defect!r}; use one of {SUPPORTED_DEFECTS}")


def _positive_int(name: str, value: int) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _positive_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _nested_tuples(values) -> tuple:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    if isinstance(values, (int, float)):
        return (float(values),)
    if isinstance(values, (list, tuple)):
        if not values:
            return ()
        if isinstance(values[0], (list, tuple)):
            return tuple(_nested_tuples(row) for row in values)
        return tuple(float(value) for value in values)
    raise TypeError(f"cannot nest {type(values)!r}")


def arrays_close(left, right, *, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    torch = _require_torch()
    left_t = left if torch.is_tensor(left) else torch.as_tensor(left, dtype=torch.float64)
    right_t = right if torch.is_tensor(right) else torch.as_tensor(right, dtype=torch.float64)
    if tuple(left_t.shape) != tuple(right_t.shape):
        return False
    return bool(torch.allclose(left_t.double(), right_t.double(), atol=atol, rtol=rtol))


def seed_cpu(seed: int = DEFAULT_SEED) -> int:
    """Seed CPU torch RNG only. CUDA/MPS RNGs are not part of the canonical path."""

    torch = _require_torch()
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(value)
    return value


def canonical_device():
    torch = _require_torch()
    return torch.device(CANONICAL_DEVICE)


def _float_tensor(values, *, requires_grad: bool = False):
    torch = _require_torch()
    tensor = torch.as_tensor(values, dtype=torch.float64, device=CANONICAL_DEVICE)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def _long_tensor(values):
    torch = _require_torch()
    return torch.as_tensor(values, dtype=torch.long, device=CANONICAL_DEVICE)


def _model_cls():
    torch = _require_torch()
    nn = torch.nn
    cached = getattr(_model_cls, "_cache", None)
    if cached is not None:
        return cached

    class TwoLayerNet(nn.Module):
        """Affine -> ReLU -> optional Dropout -> affine logits.

        ``nn.Linear`` owns ``weight`` as (out, in). M24's dense map is
        ``X @ W + b`` with ``W`` as (in, out). Teaching loads use ``W.T``.
        """

        def __init__(
            self,
            n_in: int = N_IN,
            n_hidden: int = N_HIDDEN,
            n_out: int = N_OUT,
            dropout_p: float = DEFAULT_DROPOUT_P,
        ):
            super().__init__()
            if min(int(n_in), int(n_hidden), int(n_out)) < 1:
                raise ValueError("layer widths must be positive")
            p = float(dropout_p)
            if not math.isfinite(p) or p < 0.0 or p >= 1.0:
                raise ValueError("dropout_p must be in [0, 1)")
            self.n_in = int(n_in)
            self.n_hidden = int(n_hidden)
            self.n_out = int(n_out)
            self.dropout_p = p
            self.fc1 = nn.Linear(self.n_in, self.n_hidden)
            self.dropout = nn.Dropout(p=p)
            self.fc2 = nn.Linear(self.n_hidden, self.n_out)
            self.to(dtype=torch.float64, device=CANONICAL_DEVICE)

        def forward(self, x):
            hidden_preactivation = self.fc1(x)
            hidden_activation = torch.relu(hidden_preactivation)
            hidden_activation = self.dropout(hidden_activation)
            return self.fc2(hidden_activation)

        def named_forward(self, x) -> dict[str, object]:
            hidden_preactivation = self.fc1(x)
            hidden_activation = torch.relu(hidden_preactivation)
            dropped = self.dropout(hidden_activation)
            logits = self.fc2(dropped)
            probabilities = torch.softmax(logits, dim=-1)
            return {
                "x": x,
                "hidden_preactivation": hidden_preactivation,
                "hidden_activation": hidden_activation,
                "dropped": dropped,
                "logits": logits,
                "probabilities": probabilities,
            }

    _model_cls._cache = TwoLayerNet
    return TwoLayerNet


def make_two_layer_net(
    *,
    n_in: int = N_IN,
    n_hidden: int = N_HIDDEN,
    n_out: int = N_OUT,
    dropout_p: float = DEFAULT_DROPOUT_P,
    seed: int | None = None,
):
    """Construct a CPU float64 two-layer module. Parameters are nn.Linear-owned."""

    if seed is not None:
        seed_cpu(seed)
    return _model_cls()(n_in=n_in, n_hidden=n_hidden, n_out=n_out, dropout_p=dropout_p)


def mean_softmax_nll():
    torch = _require_torch()
    return torch.nn.CrossEntropyLoss(reduction="mean")


def make_sgd(model, *, learning_rate: float, momentum: float = TEACHING_MOMENTUM):
    torch = _require_torch()
    lr = _positive_float("learning_rate", learning_rate)
    mom = float(momentum)
    if not math.isfinite(mom) or mom < 0.0 or mom >= 1.0:
        raise ValueError("momentum must be in [0, 1)")
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)


def parameter_ownership(model) -> tuple[dict[str, object], ...]:
    rows = []
    for name, param in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": tuple(int(size) for size in param.shape),
                "requires_grad": bool(param.requires_grad),
                "is_leaf": bool(param.is_leaf),
                "device": str(param.device),
                "dtype": str(param.dtype).replace("torch.", ""),
            }
        )
    return tuple(rows)


def snapshot_parameters(model) -> dict[str, tuple]:
    return {
        name: _nested_tuples(param.detach().clone())
        for name, param in model.named_parameters()
    }


def parameters_equal(left: dict[str, tuple], right: dict[str, tuple]) -> bool:
    if left.keys() != right.keys():
        return False
    for name in left:
        if not arrays_close(left[name], right[name], atol=0.0, rtol=0.0):
            return False
    return True


def grad_map(model) -> dict[str, object]:
    torch = _require_torch()
    out = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            out[name] = None
        else:
            out[name] = param.grad.detach().clone()
    return out


def total_grad_norm(model) -> float:
    torch = _require_torch()
    total = torch.zeros((), dtype=torch.float64)
    found = False
    for param in model.parameters():
        if param.grad is None:
            continue
        found = True
        total = total + param.grad.detach().double().square().sum()
    if not found:
        return 0.0
    return float(total.sqrt())


def total_param_norm(model) -> float:
    torch = _require_torch()
    total = torch.zeros((), dtype=torch.float64)
    for param in model.parameters():
        total = total + param.detach().double().square().sum()
    return float(total.sqrt())


def m24_layout_weights(model) -> dict[str, object]:
    """Convert nn.Linear weights to M24 (in, out) layout."""

    return {
        "W1": model.fc1.weight.detach().T.contiguous(),
        "b1": model.fc1.bias.detach().contiguous(),
        "W2": model.fc2.weight.detach().T.contiguous(),
        "b2": model.fc2.bias.detach().contiguous(),
    }


def load_teaching_parameters(model) -> None:
    """Copy M24 REFERENCE_* into an nn.Module with matching shapes."""

    m24 = _load_m24()
    torch = _require_torch()
    W1 = torch.as_tensor(m24.REFERENCE_W1, dtype=torch.float64)
    b1 = torch.as_tensor(m24.REFERENCE_B1, dtype=torch.float64)
    W2 = torch.as_tensor(m24.REFERENCE_W2, dtype=torch.float64)
    b2 = torch.as_tensor(m24.REFERENCE_B2, dtype=torch.float64)
    expected_in, expected_hidden = W1.shape
    expected_out = W2.shape[1]
    if tuple(model.fc1.weight.shape) != (expected_hidden, expected_in):
        raise ValueError("model fc1 does not match M24 W1")
    if tuple(model.fc2.weight.shape) != (expected_out, expected_hidden):
        raise ValueError("model fc2 does not match M24 W2")
    with torch.no_grad():
        model.fc1.weight.copy_(W1.T)
        model.fc1.bias.copy_(b1)
        model.fc2.weight.copy_(W2.T)
        model.fc2.bias.copy_(b2)


def teaching_module(*, dropout_p: float = DEFAULT_DROPOUT_P):
    model = make_two_layer_net(dropout_p=dropout_p)
    load_teaching_parameters(model)
    model.eval()
    return model


def teaching_batch():
    m24 = _load_m24()
    return _float_tensor(m24.REFERENCE_X), _long_tensor(m24.TEACHING_TARGETS)


def named_forward_on_teaching(model=None) -> dict[str, object]:
    if model is None:
        model = teaching_module()
    x, _targets = teaching_batch()
    was_training = model.training
    model.eval()
    try:
        with _require_torch().no_grad():
            return model.named_forward(x)
    finally:
        model.train(was_training)


def forward_parity_report(model=None) -> dict[str, object]:
    m24 = _load_m24()
    named = named_forward_on_teaching(model)
    logits_ok = arrays_close(named["logits"], m24.REFERENCE_LOGITS, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    hidden_ok = arrays_close(
        named["hidden_activation"],
        m24.REFERENCE_HIDDEN_ACTIVATION,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    pre_ok = arrays_close(
        named["hidden_preactivation"],
        m24.REFERENCE_HIDDEN_PREACTIVATION,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    probs_ok = arrays_close(
        named["probabilities"],
        m24.REFERENCE_PROBABILITIES,
        atol=1e-12,
        rtol=0.0,
    )
    return {
        "logits_agree": logits_ok,
        "hidden_activation_agree": hidden_ok,
        "hidden_preactivation_agree": pre_ok,
        "probabilities_agree": probs_ok,
        "agrees": bool(logits_ok and hidden_ok and pre_ok and probs_ok),
        "logits": _nested_tuples(named["logits"]),
        "layout": PARAMETER_LAYOUT,
    }


@dataclass(frozen=True)
class GradParityRow:
    name: str
    max_abs_error: float
    agrees: bool


def autograd_parameter_grads(model, x, targets, *, loss_fn=None) -> dict[str, object]:
    torch = _require_torch()
    criterion = loss_fn if loss_fn is not None else mean_softmax_nll()
    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()
    try:
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        layout = m24_layout_weights(model)
        grads = {
            "W1": model.fc1.weight.grad.detach().T.contiguous(),
            "b1": model.fc1.bias.grad.detach().contiguous(),
            "W2": model.fc2.weight.grad.detach().T.contiguous(),
            "b2": model.fc2.bias.grad.detach().contiguous(),
            "loss": float(loss.detach()),
            "logits": logits.detach(),
        }
        grads["weights"] = layout
        return grads
    finally:
        model.train(was_training)


def autograd_parity_report(model=None) -> dict[str, object]:
    """Compare selected autograd grads with M24 on the teaching micro-case."""

    m24 = _load_m24()
    if model is None:
        model = teaching_module()
    x, targets = teaching_batch()
    auto = autograd_parameter_grads(model, x, targets)
    manual = m24.two_layer_backward(
        m24.REFERENCE_X,
        m24.REFERENCE_W1,
        m24.REFERENCE_B1,
        m24.REFERENCE_W2,
        m24.REFERENCE_B2,
        m24.TEACHING_TARGETS,
    )
    rows = []
    mapping = {
        "W1": (auto["W1"], manual.d_W1),
        "b1": (auto["b1"], manual.d_b1),
        "W2": (auto["W2"], manual.d_W2),
        "b2": (auto["b2"], manual.d_b2),
    }
    torch = _require_torch()
    for name, (left, right) in mapping.items():
        delta = torch.as_tensor(left, dtype=torch.float64) - torch.as_tensor(right, dtype=torch.float64)
        err = float(delta.abs().max())
        rows.append(
            GradParityRow(
                name=name,
                max_abs_error=err,
                agrees=arrays_close(left, right, atol=PARITY_ATOL, rtol=PARITY_RTOL),
            )
        )
    selected = float(auto["W2"][0, 0])
    selected_m24 = float(torch.as_tensor(manual.d_W2, dtype=torch.float64)[0, 0])
    return {
        "loss_autograd": auto["loss"],
        "loss_m24": float(manual.loss),
        "loss_agrees": abs(auto["loss"] - float(manual.loss)) <= PARITY_ATOL,
        "rows": tuple(rows),
        "agrees": bool(all(row.agrees for row in rows) and abs(auto["loss"] - float(manual.loss)) <= PARITY_ATOL),
        "selected_name": "W2[0,0]",
        "selected_autograd": selected,
        "selected_m24": selected_m24,
        "layout": PARAMETER_LAYOUT,
    }


@dataclass(frozen=True)
class StepTrace:
    order: tuple[str, ...]
    loss: float
    grad_norm: float
    param_norm: float
    zero_grad_called: bool
    model_training: bool
    parameters_updated: bool
    selected_param: float
    selected_grad: float
    defect: str
    split: str = "train"


def _selected_w2_00(model) -> tuple[float, float]:
    weight = model.fc2.weight
    value = float(weight.detach()[0, 0])
    grad = float("nan") if weight.grad is None else float(weight.grad.detach()[0, 0])
    return value, grad


def training_step(
    model,
    batch_x,
    batch_y,
    optimizer,
    *,
    loss_fn=None,
    defect: str = "none",
    split: str = "train",
) -> StepTrace:
    """One canonical loop body. Defects change only gradient reset."""

    torch = _require_torch()
    key = _normalize_defect(defect)
    if key == "train_mode_eval":
        raise ValueError("train_mode_eval is an evaluation defect; use evaluate()")
    criterion = loss_fn if loss_fn is not None else mean_softmax_nll()
    before = snapshot_parameters(model)
    zeroed = key != "stale_grad"
    if zeroed:
        optimizer.zero_grad(set_to_none=False)
    model.train()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    _, selected_grad = _selected_w2_00(model)
    grad_norm = total_grad_norm(model)
    optimizer.step()
    after = snapshot_parameters(model)
    selected_param = float(model.fc2.weight.detach()[0, 0])
    return StepTrace(
        order=LOOP_ORDER if zeroed else ("forward", "loss", "backward", "step"),
        loss=float(loss.detach()),
        grad_norm=grad_norm,
        param_norm=total_param_norm(model),
        zero_grad_called=zeroed,
        model_training=bool(model.training),
        parameters_updated=not parameters_equal(before, after),
        selected_param=selected_param,
        selected_grad=selected_grad,
        defect=key,
        split=split,
    )


def canonical_training_step(
    *,
    learning_rate: float = TEACHING_LEARNING_RATE,
    defect: str = "none",
) -> dict[str, object]:
    """Teaching-batch zero_grad -> forward -> loss -> backward -> step."""

    model = teaching_module()
    optimizer = make_sgd(model, learning_rate=learning_rate, momentum=TEACHING_MOMENTUM)
    x, targets = teaching_batch()
    before = snapshot_parameters(model)
    loss_before = float(mean_softmax_nll()(model(x), targets).detach())
    trace = training_step(model, x, targets, optimizer, defect=defect)
    x2, targets2 = teaching_batch()
    model.eval()
    with _require_torch().no_grad():
        loss_after = float(mean_softmax_nll()(model(x2), targets2).detach())
    return {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "trace": trace,
        "parameters_moved": not parameters_equal(before, snapshot_parameters(model)),
        "learning_rate": float(learning_rate),
        "momentum": TEACHING_MOMENTUM,
        "update_convention": "SGD: parameter - learning_rate * gradient (momentum=0)",
        "order": LOOP_ORDER,
    }


@dataclass(frozen=True)
class SplitBundle:
    features: object
    labels: object
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    held_out_idx: tuple[int, ...]
    seed: int
    n_classes: int = N_OUT

    def _take(self, index: Sequence[int]):
        idx = _long_tensor(index)
        return self.features[idx], self.labels[idx]

    def train(self):
        return self._take(self.train_idx)

    def val(self):
        return self._take(self.val_idx)

    def held_out(self):
        return self._take(self.held_out_idx)


def assert_protected_splits(bundle: SplitBundle) -> bool:
    train, val, held = set(bundle.train_idx), set(bundle.val_idx), set(bundle.held_out_idx)
    if not train or not val or not held:
        raise ValueError("train, val, and held_out must each be non-empty")
    if train & val or train & held or val & held:
        raise ValueError("splits must be disjoint; held-out may not leak into train or val")
    n = int(bundle.features.shape[0])
    union = train | val | held
    if min(union) < 0 or max(union) >= n:
        raise ValueError("split index out of range")
    if len(union) != n:
        raise ValueError("every example must belong to exactly one split")
    return True


def make_classification_fixture(seed: int = DEFAULT_SPLIT_SEED) -> SplitBundle:
    """Tiny 3-class clusters. Split membership is deterministic and disjoint."""

    torch = _require_torch()
    seed_cpu(seed)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    centroids = torch.tensor(
        (
            (1.6, -0.4, 0.1),
            (-0.5, 1.5, 0.0),
            (0.1, -0.3, 1.6),
        ),
        dtype=torch.float64,
        device=CANONICAL_DEVICE,
    )
    features = []
    labels = []
    train_idx: list[int] = []
    val_idx: list[int] = []
    held_idx: list[int] = []
    cursor = 0
    for cls in range(N_OUT):
        noise = torch.randn(N_PER_CLASS, N_IN, generator=generator, dtype=torch.float64) * 0.18
        block = centroids[cls] + noise
        order = torch.randperm(N_PER_CLASS, generator=generator)
        block = block[order]
        features.append(block)
        labels.append(torch.full((N_PER_CLASS,), cls, dtype=torch.long, device=CANONICAL_DEVICE))
        local = list(range(cursor, cursor + N_PER_CLASS))
        train_idx.extend(local[:TRAIN_PER_CLASS])
        val_idx.extend(local[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS])
        held_idx.extend(local[TRAIN_PER_CLASS + VAL_PER_CLASS :])
        cursor += N_PER_CLASS
    bundle = SplitBundle(
        features=torch.cat(features, dim=0),
        labels=torch.cat(labels, dim=0),
        train_idx=tuple(int(i) for i in train_idx),
        val_idx=tuple(int(i) for i in val_idx),
        held_out_idx=tuple(int(i) for i in held_idx),
        seed=int(seed),
    )
    assert_protected_splits(bundle)
    return bundle


def make_loader(features, labels, *, batch_size: int, shuffle: bool, seed: int):
    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    batch = _positive_int("batch_size", batch_size)
    dataset = TensorDataset(features, labels)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


@dataclass(frozen=True)
class EvalTrace:
    split: str
    loss: float
    accuracy: float
    n: int
    model_training: bool
    grad_enabled: bool
    parameters_updated: bool
    defect: str
    logits: tuple


def evaluate(
    model,
    features,
    labels,
    *,
    loss_fn=None,
    split: str = "val",
    defect: str = "none",
) -> EvalTrace:
    """Score a split without an optimizer step. Evaluation cannot update parameters."""

    torch = _require_torch()
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}")
    key = _normalize_defect(defect)
    if key == "stale_grad":
        raise ValueError("stale_grad is a training-step defect; use training_step()")
    criterion = loss_fn if loss_fn is not None else mean_softmax_nll()
    before = snapshot_parameters(model)
    was_training = model.training
    mode_train = key == "train_mode_eval"
    model.train(mode_train)
    context = nullcontext() if mode_train else torch.no_grad()
    try:
        with context:
            logits = model(features)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            accuracy = float((preds == labels).to(torch.float64).mean().detach())
            loss_value = float(loss.detach())
            grad_enabled = bool(torch.is_grad_enabled())
            mode_during = bool(model.training)
        after = snapshot_parameters(model)
        updated = not parameters_equal(before, after)
        if updated:
            raise RuntimeError("evaluation updated parameters")
        return EvalTrace(
            split=split,
            loss=loss_value,
            accuracy=accuracy,
            n=int(labels.shape[0]),
            model_training=mode_during,
            grad_enabled=grad_enabled,
            parameters_updated=updated,
            defect=key,
            logits=_nested_tuples(logits),
        )
    finally:
        model.train(was_training)


@dataclass
class EpochTrace:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    n_steps: int
    grad_norm: float
    defect: str
    steps: tuple[StepTrace, ...] = field(default_factory=tuple)


@dataclass
class TrainRun:
    seed: int
    split_seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    dropout_p: float
    n_in: int
    n_hidden: int
    n_out: int
    device: str
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    held_out_idx: tuple[int, ...]
    epoch_traces: tuple[EpochTrace, ...]
    defect: str
    model: object = field(repr=False)
    optimizer: object = field(repr=False)
    splits: SplitBundle = field(repr=False)


def train_model(
    *,
    seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    momentum: float = DEFAULT_MOMENTUM,
    dropout_p: float = DEFAULT_DROPOUT_P,
    n_hidden: int = TRAIN_HIDDEN,
    defect: str = "none",
    splits: SplitBundle | None = None,
    model=None,
    optimizer=None,
) -> TrainRun:
    """Train on the train split only. Validate each epoch. Do not touch held-out."""

    key = _normalize_defect(defect)
    n_epochs = _positive_int("epochs", epochs)
    batch = _positive_int("batch_size", batch_size)
    seed_cpu(seed)
    bundle = splits if splits is not None else make_classification_fixture(split_seed)
    assert_protected_splits(bundle)
    net = model if model is not None else make_two_layer_net(
        n_hidden=n_hidden, dropout_p=dropout_p, seed=seed
    )
    opt = optimizer if optimizer is not None else make_sgd(
        net, learning_rate=learning_rate, momentum=momentum
    )
    criterion = mean_softmax_nll()
    train_x, train_y = bundle.train()
    val_x, val_y = bundle.val()
    epoch_rows: list[EpochTrace] = []
    for epoch in range(n_epochs):
        loader = make_loader(
            train_x,
            train_y,
            batch_size=batch,
            shuffle=True,
            seed=int(seed) + epoch,
        )
        step_rows: list[StepTrace] = []
        running = 0.0
        last_grad = 0.0
        n_steps = 0
        for batch_x, batch_y in loader:
            step = training_step(
                net, batch_x, batch_y, opt, loss_fn=criterion, defect=key, split="train"
            )
            step_rows.append(step)
            running += step.loss * int(batch_y.shape[0])
            last_grad = step.grad_norm
            n_steps += 1
        train_loss = running / float(train_y.shape[0])
        val_trace = evaluate(net, val_x, val_y, loss_fn=criterion, split="val", defect="none")
        epoch_rows.append(
            EpochTrace(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=val_trace.loss,
                val_accuracy=val_trace.accuracy,
                n_steps=n_steps,
                grad_norm=last_grad,
                defect=key,
                steps=tuple(step_rows),
            )
        )
    return TrainRun(
        seed=int(seed),
        split_seed=int(bundle.seed),
        batch_size=batch,
        epochs=n_epochs,
        learning_rate=float(learning_rate),
        momentum=float(momentum),
        dropout_p=float(getattr(net, "dropout_p", dropout_p)),
        n_in=int(getattr(net, "n_in", N_IN)),
        n_hidden=int(getattr(net, "n_hidden", n_hidden)),
        n_out=int(getattr(net, "n_out", N_OUT)),
        device=CANONICAL_DEVICE,
        train_idx=bundle.train_idx,
        val_idx=bundle.val_idx,
        held_out_idx=bundle.held_out_idx,
        epoch_traces=tuple(epoch_rows),
        defect=key,
        model=net,
        optimizer=opt,
        splits=bundle,
    )


def held_out_eval(run: TrainRun, *, defect: str = "none") -> EvalTrace:
    x, y = run.splits.held_out()
    return evaluate(run.model, x, y, split="held_out", defect=defect)


def batch_size_report(
    *,
    batch_sizes: Sequence[int] = (4, 12),
    epochs: int = 3,
    seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[dict[str, object], ...]:
    """Change only batch size. Do not claim the curves are invariant."""

    bundle = make_classification_fixture(split_seed)
    rows = []
    for batch in batch_sizes:
        run = train_model(
            seed=seed,
            split_seed=split_seed,
            epochs=epochs,
            batch_size=int(batch),
            splits=bundle,
        )
        rows.append(
            {
                "batch_size": int(batch),
                "epochs": int(epochs),
                "steps_per_epoch": run.epoch_traces[0].n_steps,
                "train_losses": tuple(row.train_loss for row in run.epoch_traces),
                "val_losses": tuple(row.val_loss for row in run.epoch_traces),
                "final_train_loss": run.epoch_traces[-1].train_loss,
                "final_val_loss": run.epoch_traces[-1].val_loss,
            }
        )
    return tuple(rows)


def save_checkpoint(path, run: TrainRun, *, epoch: int | None = None, policy: str = "unfilled") -> dict[str, object]:
    torch = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    val_loss = run.epoch_traces[-1].val_loss if run.epoch_traces else float("nan")
    payload = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in run.model.state_dict().items()},
        "optimizer_state_dict": run.optimizer.state_dict(),
        "epoch": int(run.epochs - 1 if epoch is None else epoch),
        "global_step": int(sum(row.n_steps for row in run.epoch_traces)),
        "seed": int(run.seed),
        "split_seed": int(run.split_seed),
        "batch_size": int(run.batch_size),
        "learning_rate": float(run.learning_rate),
        "momentum": float(run.momentum),
        "dropout_p": float(run.dropout_p),
        "n_in": int(run.n_in),
        "n_hidden": int(run.n_hidden),
        "n_out": int(run.n_out),
        "val_loss": float(val_loss),
        "torch_rng_state": torch.get_rng_state(),
        "policy": str(policy),
    }
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {missing}")
    torch.save(payload, destination)
    return {key: payload[key] if key not in {"model_state_dict", "optimizer_state_dict", "torch_rng_state"} else key for key in CHECKPOINT_KEYS}


def load_checkpoint(path, *, model=None, optimizer=None):
    torch = _require_torch()
    payload = torch.load(path, map_location=CANONICAL_DEVICE, weights_only=False)
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"checkpoint missing keys: {missing}")
    net = model if model is not None else make_two_layer_net(
        n_in=int(payload["n_in"]),
        n_hidden=int(payload["n_hidden"]),
        n_out=int(payload["n_out"]),
        dropout_p=float(payload["dropout_p"]),
    )
    net.load_state_dict(payload["model_state_dict"])
    opt = optimizer if optimizer is not None else make_sgd(
        net, learning_rate=float(payload["learning_rate"]), momentum=float(payload["momentum"])
    )
    opt.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"])
    net.eval()
    return net, opt, payload


def checkpoint_roundtrip(run: TrainRun, path) -> dict[str, object]:
    save_checkpoint(path, run)
    fresh = make_two_layer_net(n_in=run.n_in, n_hidden=run.n_hidden, n_out=run.n_out, dropout_p=run.dropout_p)
    fresh_opt = make_sgd(fresh, learning_rate=run.learning_rate, momentum=run.momentum)
    loaded, loaded_opt, payload = load_checkpoint(path, model=fresh, optimizer=fresh_opt)
    x, y = run.splits.held_out()
    first = evaluate(loaded, x, y, split="held_out")
    second = evaluate(loaded, x, y, split="held_out")
    original = evaluate(run.model, x, y, split="held_out")
    return {
        "payload_keys": tuple(sorted(payload)),
        "held_out_agrees": arrays_close(first.logits, original.logits, atol=1e-12, rtol=0.0),
        "replay_agrees": arrays_close(first.logits, second.logits, atol=0.0, rtol=0.0),
        "held_out_loss": first.loss,
        "parameters_updated": first.parameters_updated,
        "model_training": first.model_training,
        "optimizer_state_loaded": loaded_opt.state_dict()["state"] is not None,
        "epoch": payload["epoch"],
    }


def gradient_reset_experiment(*, learning_rate: float = TEACHING_LEARNING_RATE) -> dict[str, object]:
    """Same batch twice, with and without zero_grad. Gradients add if reset is skipped."""

    x, targets = teaching_batch()
    criterion = mean_softmax_nll()

    def _two_backwards(*, reset: bool) -> tuple[float, float]:
        model = teaching_module()
        model.train()
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), targets)
        loss.backward()
        first = float(model.fc2.weight.grad.detach()[0, 0])
        if reset:
            model.zero_grad(set_to_none=False)
        loss2 = criterion(model(x), targets)
        loss2.backward()
        second = float(model.fc2.weight.grad.detach()[0, 0])
        return first, second

    once, twice_reset = _two_backwards(reset=True)
    once_stale, twice_stale = _two_backwards(reset=False)

    def _two_steps(*, defect: str) -> tuple[float, float, float]:
        model = teaching_module()
        opt = make_sgd(model, learning_rate=learning_rate, momentum=TEACHING_MOMENTUM)
        s1 = training_step(model, x, targets, opt, defect=defect)
        s2 = training_step(model, x, targets, opt, defect=defect)
        return s1.selected_grad, s2.selected_grad, s2.selected_param

    correct_g1, correct_g2, correct_p = _two_steps(defect="none")
    stale_g1, stale_g2, stale_p = _two_steps(defect="stale_grad")
    return {
        "backward_once": once,
        "backward_twice_with_reset": twice_reset,
        "backward_twice_without_reset": twice_stale,
        "reset_keeps_second_equal_first": abs(twice_reset - once) <= 1e-12,
        "stale_second_is_sum": abs(twice_stale - (once_stale + once)) <= 1e-8,
        "correct_step_grads": (correct_g1, correct_g2),
        "stale_step_grads": (stale_g1, stale_g2),
        "correct_param_after": correct_p,
        "stale_param_after": stale_p,
        "updates_diverge": abs(correct_p - stale_p) > 1e-12,
    }


def train_mode_eval_experiment(*, seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Dropout makes train-mode evaluation disagree with eval/no-grad inference."""

    seed_cpu(seed)
    model = teaching_module(dropout_p=MODE_DROPOUT_P)
    x, targets = teaching_batch()
    correct = evaluate(model, x, targets, split="val", defect="none")
    seed_cpu(seed)
    wrong = evaluate(model, x, targets, split="val", defect="train_mode_eval")
    return {
        "correct_training_flag": correct.model_training,
        "wrong_training_flag": wrong.model_training,
        "correct_grad_enabled": correct.grad_enabled,
        "wrong_grad_enabled": wrong.grad_enabled,
        "correct_loss": correct.loss,
        "wrong_loss": wrong.loss,
        "logits_differ": not arrays_close(correct.logits, wrong.logits, atol=1e-12, rtol=0.0),
        "parameters_updated_correct": correct.parameters_updated,
        "parameters_updated_wrong": wrong.parameters_updated,
        "dropout_p": MODE_DROPOUT_P,
    }


def compact_run_report(run: TrainRun) -> dict[str, object]:
    return {
        "seed": run.seed,
        "batch_size": run.batch_size,
        "epochs": run.epochs,
        "learning_rate": run.learning_rate,
        "momentum": run.momentum,
        "dropout_p": run.dropout_p,
        "n_hidden": run.n_hidden,
        "device": run.device,
        "defect": run.defect,
        "n_train": len(run.train_idx),
        "n_val": len(run.val_idx),
        "n_held_out": len(run.held_out_idx),
        "train_losses": tuple(row.train_loss for row in run.epoch_traces),
        "val_losses": tuple(row.val_loss for row in run.epoch_traces),
        "val_accuracies": tuple(row.val_accuracy for row in run.epoch_traces),
        "steps_per_epoch": tuple(row.n_steps for row in run.epoch_traces),
    }
