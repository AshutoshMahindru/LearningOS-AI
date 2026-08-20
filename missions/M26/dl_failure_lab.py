"""Deep-learning failure diagnosis lab for Mission M26.

M26 does not reimplement the training loop. It imports the M25 instrumented
CPU loop and injects one named fault at a time into data, optimization,
gradient flow, capacity, regularization, or evaluation.

Canonical path: local deterministic tensors. No weight download, no CUDA
requirement, no secrets. Torch is loaded lazily through M25.

Chaos Day hides the defect category until the learner's diagnostic step.
Tests may call named seeded defects directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import sys
from typing import Sequence


# Aligned with missions/M25/pytorch_training.py. Runtime `_align_m25` checks.
CANONICAL_DEVICE = "cpu"
DEFAULT_SEED = 2501
DEFAULT_SPLIT_SEED = 2501
HIDDEN_PRACTICE_SEED = 2602
CHAOS_SEED = 2625
DEFAULT_LEARNING_RATE = 0.25
DEFAULT_MOMENTUM = 0.9
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_DROPOUT_P = 0.0
TRAIN_HIDDEN = 8
MODE_DROPOUT_P = 0.5
N_OUT = 3

HIGH_LEARNING_RATE = 8.0
LOW_LEARNING_RATE = 1.0e-6
SCALE_FACTOR = 20.0
SCALE_FEATURE = 0
LABEL_CORRUPT_FRACTION = 0.5
TINY_HIDDEN = 1
CRUSHING_DROPOUT_P = 0.8
TINY_SUBSET_N = 8
TINY_SUBSET_EPOCHS = 30
TINY_OVERFIT_LOSS = 0.2
FROZEN_MODULE = "fc1"
FIXTURE_BLOCK_SIZE = 12
FIT_VAL_ACCURACY = 0.9
TRAIN_DROP_RATIO = 0.5

# Centroids copied from M25.make_classification_fixture for cluster checks.
FIXTURE_CENTROIDS = (
    (1.6, -0.4, 0.1),
    (-0.5, 1.5, 0.0),
    (0.1, -0.3, 1.6),
)

SUPPORTED_DEFECTS = (
    "none",
    "correct",
    "label_shuffle",
    "feature_scale",
    "lr_high",
    "lr_low",
    "frozen_layer",
    "tiny_hidden",
    "crushing_dropout",
    "train_mode_eval",
    "val_leakage",
)

DIAGNOSIS_KEYS = (
    "symptom",
    "hypotheses",
    "discriminating_experiment",
    "observed_result",
    "root_cause",
    "smallest_repair",
    "verification",
    "regression_evidence",
)

HYPOTHESIS_IDS = (
    "labels_corrupted",
    "features_misfcaled",
    "lr_too_high",
    "lr_too_low",
    "blocked_gradient_path",
    "capacity_too_small",
    "regularization_too_strong",
    "eval_train_mode",
    "eval_split_leakage",
)

CATALOGUE = {
    "none": {"category": "none", "family": "control", "chaos": False},
    "label_shuffle": {"category": "data", "family": "labels", "chaos": True},
    "feature_scale": {"category": "data", "family": "scaling", "chaos": False},
    "lr_high": {"category": "optimization", "family": "learning_rate", "chaos": True},
    "lr_low": {"category": "optimization", "family": "learning_rate", "chaos": False},
    "frozen_layer": {"category": "gradient_flow", "family": "blocked_path", "chaos": True},
    "tiny_hidden": {"category": "architecture", "family": "capacity", "chaos": True},
    "crushing_dropout": {"category": "regularization", "family": "regularization", "chaos": False},
    "train_mode_eval": {"category": "evaluation", "family": "model_mode", "chaos": False},
    "val_leakage": {"category": "evaluation", "family": "split_leakage", "chaos": True},
}

UNFILLED = "[UNFILLED BY LEARNER]"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_m25():
    """Import missions.M25.pytorch_training as the trusted instrumented loop."""

    name = "_learningos_m25_pytorch_training"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import missions.M25.pytorch_training as module
    except ImportError:
        path = Path(__file__).resolve().parent.parent / "M25" / "pytorch_training.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load missions/M25/pytorch_training.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _align_m25(m25) -> None:
    if m25.DEFAULT_SEED != DEFAULT_SEED:
        raise RuntimeError("M26 DEFAULT_SEED drifted from M25")
    if m25.DEFAULT_SPLIT_SEED != DEFAULT_SPLIT_SEED:
        raise RuntimeError("M26 DEFAULT_SPLIT_SEED drifted from M25")
    if m25.DEFAULT_LEARNING_RATE != DEFAULT_LEARNING_RATE:
        raise RuntimeError("M26 DEFAULT_LEARNING_RATE drifted from M25")
    if m25.TRAIN_HIDDEN != TRAIN_HIDDEN:
        raise RuntimeError("M26 TRAIN_HIDDEN drifted from M25")
    if m25.CANONICAL_DEVICE != CANONICAL_DEVICE:
        raise RuntimeError("M26 canonical device drifted from M25")


def m25_loop_order() -> tuple[str, ...]:
    return tuple(_load_m25().LOOP_ORDER)


def _normalize_defect(defect: str) -> str:
    key = str(defect).lower().strip()
    aliases = {
        "": "none",
        "none": "none",
        "correct": "none",
        "healthy": "none",
        "label_shuffle": "label_shuffle",
        "shuffled_labels": "label_shuffle",
        "label_noise": "label_shuffle",
        "data_corruption": "label_shuffle",
        "feature_scale": "feature_scale",
        "scaling": "feature_scale",
        "unscaled": "feature_scale",
        "lr_high": "lr_high",
        "learning_rate_too_high": "lr_high",
        "lr_too_high": "lr_high",
        "learning_rate_failure": "lr_high",
        "lr_low": "lr_low",
        "learning_rate_too_low": "lr_low",
        "lr_too_low": "lr_low",
        "frozen_layer": "frozen_layer",
        "blocked_gradient": "frozen_layer",
        "blocked_gradient_path": "frozen_layer",
        "frozen_fc1": "frozen_layer",
        "gradient_flow": "frozen_layer",
        "tiny_hidden": "tiny_hidden",
        "tiny_capacity": "tiny_hidden",
        "capacity_mismatch": "tiny_hidden",
        "underfit_width": "tiny_hidden",
        "crushing_dropout": "crushing_dropout",
        "too_much_dropout": "crushing_dropout",
        "regularization": "crushing_dropout",
        "train_mode_eval": "train_mode_eval",
        "eval_in_train_mode": "train_mode_eval",
        "dropout_eval": "train_mode_eval",
        "val_leakage": "val_leakage",
        "leakage": "val_leakage",
        "train_as_val": "val_leakage",
        "evaluation_defect": "val_leakage",
    }
    if key not in aliases:
        raise ValueError(f"unknown defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return aliases[key]


def chaos_families() -> tuple[str, ...]:
    return tuple(name for name, meta in CATALOGUE.items() if meta.get("chaos"))


def defect_for_seed(seed: int) -> str:
    families = chaos_families()
    if not families:
        raise RuntimeError("no chaos families configured")
    index = (int(seed) * 7 + 3) % len(families)
    return families[index]


def category_for_defect(defect: str) -> str:
    key = _normalize_defect(defect)
    return str(CATALOGUE[key]["category"])


def empty_diagnosis_record() -> dict[str, str]:
    return {key: UNFILLED for key in DIAGNOSIS_KEYS}


def diagnosis_record_complete(record: dict[str, object]) -> bool:
    if set(DIAGNOSIS_KEYS) - set(record):
        return False
    for key in DIAGNOSIS_KEYS:
        value = record[key]
        if value is None or value == "" or value == UNFILLED:
            return False
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return False
    return True


@dataclass
class PreparedFault:
    """One injectable fault plus the objects a smallest repair must edit."""

    defect: str
    category: str
    seed: int
    split_seed: int
    learning_rate: float
    momentum: float
    dropout_p: float
    n_hidden: int
    epochs: int
    batch_size: int
    splits: object = field(repr=False)
    clean_labels: object = field(repr=False)
    clean_features: object = field(repr=False)
    freeze_fc1: bool = False
    leak_val: bool = False
    eval_in_train_mode: bool = False
    scale_applied: bool = False
    labels_corrupted: bool = False
    hidden: bool = False
    repaired: bool = False
    original_defect: str = "none"

    def __repr__(self) -> str:
        if self.hidden and not self.repaired:
            return (
                f"PreparedFault(hidden=True, seed={self.seed}, "
                f"epochs={self.epochs}, batch_size={self.batch_size})"
            )
        return (
            f"PreparedFault(defect={self.defect!r}, category={self.category!r}, "
            f"repaired={self.repaired}, seed={self.seed}, "
            f"learning_rate={self.learning_rate}, n_hidden={self.n_hidden}, "
            f"dropout_p={self.dropout_p}, freeze_fc1={self.freeze_fc1}, "
            f"leak_val={self.leak_val}, eval_in_train_mode={self.eval_in_train_mode})"
        )


@dataclass
class DiagnosticRun:
    train_run: object = field(repr=False)
    fault: PreparedFault = field(repr=False)
    init_params: dict = field(repr=False)
    layer_moved: dict
    grad_report: dict
    honest_val: object = field(repr=False)
    claimed_val: object = field(repr=False)
    held_out: object = field(repr=False)
    train_losses: tuple
    val_losses: tuple
    val_accuracies: tuple

    def __repr__(self) -> str:
        return f"DiagnosticRun(symptoms={public_symptoms(self)!r})"


def clone_split_bundle(bundle):
    m25 = _load_m25()
    return m25.SplitBundle(
        features=bundle.features.detach().clone(),
        labels=bundle.labels.detach().clone(),
        train_idx=tuple(int(i) for i in bundle.train_idx),
        val_idx=tuple(int(i) for i in bundle.val_idx),
        held_out_idx=tuple(int(i) for i in bundle.held_out_idx),
        seed=int(bundle.seed),
        n_classes=int(bundle.n_classes),
    )


def _corrupt_train_labels(labels, train_idx: Sequence[int], *, fraction: float, seed: int):
    m25 = _load_m25()
    torch = m25._require_torch()
    idx = torch.as_tensor(list(train_idx), dtype=torch.long, device=labels.device)
    n = max(1, int(round(float(fraction) * int(idx.numel()))))
    n = min(n, int(idx.numel()))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    pick = idx[torch.randperm(int(idx.numel()), generator=generator)[:n]]
    labels[pick] = (labels[pick] + 1) % N_OUT
    return tuple(int(i) for i in pick.tolist())


def prepare_fault(
    defect: str = "none",
    *,
    seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden: bool = False,
    splits=None,
) -> PreparedFault:
    """Inject one named fault into a clone of the M25 fixture. Does not train."""

    m25 = _load_m25()
    _align_m25(m25)
    key = _normalize_defect(defect)
    meta = CATALOGUE[key]
    bundle = clone_split_bundle(splits if splits is not None else m25.make_classification_fixture(split_seed))
    m25.assert_protected_splits(bundle)
    clean_labels = bundle.labels.detach().clone()
    clean_features = bundle.features.detach().clone()

    learning_rate = DEFAULT_LEARNING_RATE
    dropout_p = DEFAULT_DROPOUT_P
    n_hidden = TRAIN_HIDDEN
    freeze_fc1 = False
    leak_val = False
    eval_in_train_mode = False
    scale_applied = False
    labels_corrupted = False

    if key == "label_shuffle":
        _corrupt_train_labels(
            bundle.labels,
            bundle.train_idx,
            fraction=LABEL_CORRUPT_FRACTION,
            seed=seed,
        )
        labels_corrupted = True
    elif key == "feature_scale":
        bundle.features[:, SCALE_FEATURE] *= SCALE_FACTOR
        scale_applied = True
    elif key == "lr_high":
        learning_rate = HIGH_LEARNING_RATE
    elif key == "lr_low":
        learning_rate = LOW_LEARNING_RATE
    elif key == "frozen_layer":
        freeze_fc1 = True
    elif key == "tiny_hidden":
        n_hidden = TINY_HIDDEN
    elif key == "crushing_dropout":
        dropout_p = CRUSHING_DROPOUT_P
    elif key == "train_mode_eval":
        dropout_p = MODE_DROPOUT_P
        eval_in_train_mode = True
    elif key == "val_leakage":
        leak_val = True

    return PreparedFault(
        defect=key,
        category=str(meta["category"]),
        seed=int(seed),
        split_seed=int(bundle.seed),
        learning_rate=float(learning_rate),
        momentum=DEFAULT_MOMENTUM,
        dropout_p=float(dropout_p),
        n_hidden=int(n_hidden),
        epochs=int(epochs),
        batch_size=int(batch_size),
        splits=bundle,
        clean_labels=clean_labels,
        clean_features=clean_features,
        freeze_fc1=freeze_fc1,
        leak_val=leak_val,
        eval_in_train_mode=eval_in_train_mode,
        scale_applied=scale_applied,
        labels_corrupted=labels_corrupted,
        hidden=bool(hidden),
        repaired=False,
        original_defect=key,
    )


def repair_prepared(fault: PreparedFault) -> PreparedFault:
    """Smallest repair on the same prepared objects. Does not build a fresh healthy twin."""

    if fault.repaired:
        return fault
    original = fault.original_defect
    if original in {"label_shuffle", "feature_scale", "none"}:
        fault.splits.labels.copy_(fault.clean_labels)
        fault.splits.features.copy_(fault.clean_features)
        fault.scale_applied = False
        fault.labels_corrupted = False
    if original in {"lr_high", "lr_low"}:
        fault.learning_rate = DEFAULT_LEARNING_RATE
    if original == "frozen_layer":
        fault.freeze_fc1 = False
    if original == "tiny_hidden":
        fault.n_hidden = TRAIN_HIDDEN
    if original == "crushing_dropout":
        fault.dropout_p = DEFAULT_DROPOUT_P
    if original == "train_mode_eval":
        fault.eval_in_train_mode = False
        # Dropout can stay; the repair is eval/no-grad, not removing Dropout.
    if original == "val_leakage":
        fault.leak_val = False
    fault.repaired = True
    fault.hidden = False
    fault.defect = "none"
    fault.category = "none"
    return fault


def _snapshot_params(model) -> dict:
    return _load_m25().snapshot_parameters(model)


def _layer_moved(init_params: dict, final_params: dict) -> dict[str, bool]:
    m25 = _load_m25()
    moved = {}
    for name in init_params:
        moved[name] = not m25.arrays_close(init_params[name], final_params[name], atol=0.0, rtol=0.0)
    return moved


def apply_model_hooks(model, fault: PreparedFault):
    if fault.freeze_fc1:
        model.fc1.weight.requires_grad_(False)
        model.fc1.bias.requires_grad_(False)
    return model


def grad_activation_stats(model, features, labels) -> dict[str, object]:
    """One reverse pass for localization. Does not call optimizer.step."""

    m25 = _load_m25()
    torch = m25._require_torch()
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    named = model.named_forward(features)
    logits = named["logits"]
    loss = m25.mean_softmax_nll()(logits, labels)
    loss.backward()
    hidden = named["hidden_activation"].detach()
    dead_frac = float((hidden <= 0).to(torch.float64).mean().item())
    per_param = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            per_param[name] = {
                "requires_grad": bool(param.requires_grad),
                "grad_norm": 0.0,
                "grad_is_none": True,
            }
        else:
            per_param[name] = {
                "requires_grad": bool(param.requires_grad),
                "grad_norm": float(param.grad.detach().double().norm().item()),
                "grad_is_none": False,
            }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    fc1_norm = float(per_param.get("fc1.weight", {}).get("grad_norm", 0.0))
    fc2_norm = float(per_param.get("fc2.weight", {}).get("grad_norm", 0.0))
    return {
        "per_param": per_param,
        "fc1_grad_norm": fc1_norm,
        "fc2_grad_norm": fc2_norm,
        "fc1_grad_is_none": bool(per_param.get("fc1.weight", {}).get("grad_is_none", True)),
        "fc1_requires_grad": bool(per_param.get("fc1.weight", {}).get("requires_grad", False)),
        "dead_relu_fraction": dead_frac,
        "loss": float(loss.detach()),
        "parameters_updated": False,
    }


def cluster_label_agreement(splits) -> dict[str, float]:
    """Compare labels to the M25 fixture's class blocks of 12 rows.

    Feature scaling must not look like label corruption, so this check uses
    row identity (`i // 12`), not Euclidean distance in possibly scaled space.
    """

    m25 = _load_m25()
    torch = m25._require_torch()
    n = int(splits.labels.shape[0])
    expected = torch.arange(n, device=splits.labels.device) // FIXTURE_BLOCK_SIZE
    labels = splits.labels.detach()

    def _agree(index: Sequence[int]) -> float:
        if not index:
            return float("nan")
        idx = torch.as_tensor(list(index), dtype=torch.long, device=labels.device)
        return float((expected[idx] == labels[idx]).to(torch.float64).mean().item())

    return {
        "train_agreement": _agree(splits.train_idx),
        "val_agreement": _agree(splits.val_idx),
        "held_out_agreement": _agree(splits.held_out_idx),
    }


def feature_column_scales(splits) -> tuple[float, ...]:
    train_x, _ = splits.train()
    stds = train_x.detach().std(dim=0)
    return tuple(float(v) for v in stds.tolist())


def feature_scale_ratio(splits) -> float:
    scales = [s for s in feature_column_scales(splits) if s > 0.0]
    if not scales:
        return 1.0
    return max(scales) / min(scales)


def honest_eval(diag: DiagnosticRun):
    m25 = _load_m25()
    x, y = diag.train_run.splits.val()
    return m25.evaluate(diag.train_run.model, x, y, split="val", defect="none")


def claimed_eval(run, fault: PreparedFault):
    m25 = _load_m25()
    if fault.eval_in_train_mode:
        x, y = run.splits.val()
        return m25.evaluate(run.model, x, y, split="val", defect="train_mode_eval")
    if fault.leak_val:
        x, y = run.splits.train()
        return m25.evaluate(run.model, x, y, split="val", defect="none")
    x, y = run.splits.val()
    return m25.evaluate(run.model, x, y, split="val", defect="none")


def _wrap_run(train_run, fault: PreparedFault, init_params: dict) -> DiagnosticRun:
    m25 = _load_m25()
    final_params = _snapshot_params(train_run.model)
    train_x, train_y = train_run.splits.train()
    grad_report = grad_activation_stats(train_run.model, train_x, train_y)
    claimed = claimed_eval(train_run, fault)
    honest = m25.evaluate(train_run.model, *train_run.splits.val(), split="val", defect="none")
    held = m25.held_out_eval(train_run, defect="none")
    return DiagnosticRun(
        train_run=train_run,
        fault=fault,
        init_params=init_params,
        layer_moved=_layer_moved(init_params, final_params),
        grad_report=grad_report,
        honest_val=honest,
        claimed_val=claimed,
        held_out=held,
        train_losses=tuple(row.train_loss for row in train_run.epoch_traces),
        val_losses=tuple(row.val_loss for row in train_run.epoch_traces),
        val_accuracies=tuple(row.val_accuracy for row in train_run.epoch_traces),
    )


def run_prepared(fault: PreparedFault) -> DiagnosticRun:
    """Train with M25.train_model using the prepared objects. Does not re-inject."""

    m25 = _load_m25()
    _align_m25(m25)
    m25.seed_cpu(fault.seed)
    model = m25.make_two_layer_net(
        n_hidden=fault.n_hidden,
        dropout_p=fault.dropout_p,
        seed=fault.seed,
    )
    apply_model_hooks(model, fault)
    optimizer = m25.make_sgd(model, learning_rate=fault.learning_rate, momentum=fault.momentum)
    init_params = _snapshot_params(model)
    train_run = m25.train_model(
        seed=fault.seed,
        split_seed=fault.split_seed,
        epochs=fault.epochs,
        batch_size=fault.batch_size,
        learning_rate=fault.learning_rate,
        momentum=fault.momentum,
        dropout_p=fault.dropout_p,
        n_hidden=fault.n_hidden,
        defect="none",
        splits=fault.splits,
        model=model,
        optimizer=optimizer,
    )
    return _wrap_run(train_run, fault, init_params)


def known_good_m25_trace(
    *,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
) -> DiagnosticRun:
    """Useful whole: unmodified M25 loop, no M26 injection."""

    return run_prepared(prepare_fault("none", seed=seed, epochs=epochs, hidden=False))


def tiny_subset_overfit_check(
    splits,
    *,
    n: int = TINY_SUBSET_N,
    epochs: int = TINY_SUBSET_EPOCHS,
    n_hidden: int = TRAIN_HIDDEN,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    freeze_fc1: bool = False,
    dropout_p: float = DEFAULT_DROPOUT_P,
) -> dict[str, object]:
    """Can the pipeline memorize a handful of train rows? Cheap sanity check."""

    m25 = _load_m25()
    _align_m25(m25)
    source_idx = [int(i) for i in splits.train_idx]
    need = int(n) + 4
    if len(source_idx) < need:
        raise ValueError("not enough train rows for tiny-subset overfit check")
    chosen = source_idx[:need]
    tiny = m25.SplitBundle(
        features=splits.features[chosen].detach().clone(),
        labels=splits.labels[chosen].detach().clone(),
        train_idx=tuple(range(int(n))),
        val_idx=tuple(range(int(n), int(n) + 2)),
        held_out_idx=tuple(range(int(n) + 2, int(n) + 4)),
        seed=int(splits.seed),
        n_classes=int(splits.n_classes),
    )
    m25.assert_protected_splits(tiny)
    m25.seed_cpu(seed)
    model = m25.make_two_layer_net(n_hidden=n_hidden, dropout_p=dropout_p, seed=seed)
    if freeze_fc1:
        model.fc1.weight.requires_grad_(False)
        model.fc1.bias.requires_grad_(False)
    optimizer = m25.make_sgd(model, learning_rate=learning_rate, momentum=0.0)
    run = m25.train_model(
        seed=seed,
        split_seed=int(splits.seed),
        epochs=int(epochs),
        batch_size=min(int(n), DEFAULT_BATCH_SIZE),
        learning_rate=float(learning_rate),
        momentum=0.0,
        dropout_p=float(dropout_p),
        n_hidden=int(n_hidden),
        defect="none",
        splits=tiny,
        model=model,
        optimizer=optimizer,
    )
    final_loss = float(run.epoch_traces[-1].train_loss)
    first_loss = float(run.epoch_traces[0].train_loss)
    return {
        "n": int(n),
        "n_hidden": int(n_hidden),
        "learning_rate": float(learning_rate),
        "freeze_fc1": bool(freeze_fc1),
        "dropout_p": float(dropout_p),
        "final_train_loss": final_loss,
        "first_train_loss": first_loss,
        "overfit": bool(final_loss <= TINY_OVERFIT_LOSS),
        "train_losses": tuple(row.train_loss for row in run.epoch_traces),
    }


def _loss_oscillates(losses: Sequence[float]) -> bool:
    finite = [float(v) for v in losses if math.isfinite(v)]
    if len(finite) < 4:
        return False
    diffs = [finite[i + 1] - finite[i] for i in range(len(finite) - 1)]
    significant = [d for d in diffs if abs(d) >= 0.2]
    if len(significant) < 3:
        return False
    sign_flips = 0
    for i in range(len(significant) - 1):
        if significant[i] * significant[i + 1] < 0:
            sign_flips += 1
    return sign_flips >= 2


def _losses_finite(losses: Sequence[float]) -> bool:
    return bool(losses) and all(math.isfinite(float(v)) for v in losses)


def public_symptoms(diag: DiagnosticRun) -> dict[str, object]:
    """Learner-facing symptoms. Hidden runs omit defect, category, and knobs."""

    claimed = diag.claimed_val
    held = diag.held_out
    out: dict[str, object] = {
        "train_losses": diag.train_losses,
        "reported_val_losses": diag.val_losses,
        "reported_val_accuracies": diag.val_accuracies,
        "final_train_loss": diag.train_losses[-1] if diag.train_losses else float("nan"),
        "final_reported_val_loss": claimed.loss,
        "final_reported_val_accuracy": claimed.accuracy,
        "held_out_loss": held.loss,
        "held_out_accuracy": held.accuracy,
        "held_out_model_training": held.model_training,
        "claimed_model_training": claimed.model_training,
        "claimed_n": claimed.n,
        "held_out_n": held.n,
        "layer_moved": dict(diag.layer_moved),
        "fc1_grad_norm": diag.grad_report["fc1_grad_norm"],
        "fc2_grad_norm": diag.grad_report["fc2_grad_norm"],
        "fc1_requires_grad": diag.grad_report["fc1_requires_grad"],
        "dead_relu_fraction": diag.grad_report["dead_relu_fraction"],
        "finite_train_losses": _losses_finite(diag.train_losses),
        "train_loss_oscillates": _loss_oscillates(diag.train_losses),
        "epochs": diag.fault.epochs,
        "batch_size": diag.fault.batch_size,
        "device": CANONICAL_DEVICE,
    }
    if not diag.fault.hidden:
        out["defect"] = diag.fault.defect
        out["category"] = diag.fault.category
        out["learning_rate"] = diag.fault.learning_rate
        out["n_hidden"] = diag.fault.n_hidden
        out["dropout_p"] = diag.fault.dropout_p
        out["freeze_fc1"] = diag.fault.freeze_fc1
        out["leak_val"] = diag.fault.leak_val
        out["eval_in_train_mode"] = diag.fault.eval_in_train_mode
        out["repaired"] = diag.fault.repaired
        out["honest_val_loss"] = diag.honest_val.loss
        out["honest_val_accuracy"] = diag.honest_val.accuracy
    return out


def diagnostic_battery(diag: DiagnosticRun) -> dict[str, object]:
    """Cheap menu that distinguishes the catalogue families. Does not name a cause."""

    fault = diag.fault
    tiny_current = tiny_subset_overfit_check(
        fault.splits,
        n_hidden=fault.n_hidden,
        learning_rate=fault.learning_rate,
        seed=fault.seed,
        freeze_fc1=fault.freeze_fc1,
        dropout_p=fault.dropout_p,
    )
    tiny_wide = tiny_subset_overfit_check(
        fault.splits,
        n_hidden=TRAIN_HIDDEN,
        learning_rate=DEFAULT_LEARNING_RATE,
        seed=fault.seed,
        freeze_fc1=False,
        dropout_p=DEFAULT_DROPOUT_P,
    )
    return {
        "cluster_agreement": cluster_label_agreement(fault.splits),
        "feature_scales": feature_column_scales(fault.splits),
        "feature_scale_ratio": feature_scale_ratio(fault.splits),
        "layer_moved": dict(diag.layer_moved),
        "grad_report": {
            "fc1_grad_norm": diag.grad_report["fc1_grad_norm"],
            "fc2_grad_norm": diag.grad_report["fc2_grad_norm"],
            "fc1_requires_grad": diag.grad_report["fc1_requires_grad"],
            "fc1_grad_is_none": diag.grad_report["fc1_grad_is_none"],
            "dead_relu_fraction": diag.grad_report["dead_relu_fraction"],
        },
        "tiny_overfit_current_knobs": tiny_current,
        "tiny_overfit_restored_knobs": tiny_wide,
        "honest_val_loss": diag.honest_val.loss,
        "honest_val_accuracy": diag.honest_val.accuracy,
        "claimed_val_loss": diag.claimed_val.loss,
        "claimed_val_accuracy": diag.claimed_val.accuracy,
        "claimed_n": diag.claimed_val.n,
        "honest_n": diag.honest_val.n,
        "claimed_model_training": diag.claimed_val.model_training,
        "held_out_loss": diag.held_out.loss,
        "held_out_accuracy": diag.held_out.accuracy,
        "finite_train_losses": _losses_finite(diag.train_losses),
        "train_loss_oscillates": _loss_oscillates(diag.train_losses),
        "train_loss_delta": (
            diag.train_losses[-1] - diag.train_losses[0] if len(diag.train_losses) >= 2 else float("nan")
        ),
    }


def signals_from_battery(battery: dict[str, object], diag: DiagnosticRun | None = None) -> dict[str, object]:
    agreement = battery["cluster_agreement"]
    tiny_cur = battery["tiny_overfit_current_knobs"]
    tiny_wide = battery["tiny_overfit_restored_knobs"]
    train_losses = diag.train_losses if diag is not None else ()
    first = float(train_losses[0]) if train_losses else float("nan")
    last = float(train_losses[-1]) if train_losses else float("nan")
    dropped = bool(math.isfinite(first) and math.isfinite(last) and last < 0.85 * first)
    claimed_n = int(battery["claimed_n"])
    honest_n = int(battery["honest_n"])
    return {
        "labels_disagree_with_clusters": float(agreement["train_agreement"]) < 0.85,
        "feature_scale_ratio": float(battery["feature_scale_ratio"]),
        "features_misfcaled": float(battery["feature_scale_ratio"]) >= 10.0,
        "loss_non_finite": not bool(battery["finite_train_losses"]),
        "loss_oscillates": bool(battery["train_loss_oscillates"]),
        "train_loss_flat": bool(math.isfinite(last) and math.isfinite(first) and abs(last - first) < 0.05),
        "train_loss_dropped": dropped,
        "fc1_grad_norm": float(battery["grad_report"]["fc1_grad_norm"]),
        "fc2_grad_norm": float(battery["grad_report"]["fc2_grad_norm"]),
        "fc1_blocked": (not bool(battery["grad_report"]["fc1_requires_grad"]))
        or bool(battery["grad_report"]["fc1_grad_is_none"]),
        "fc1_moved": bool(battery["layer_moved"].get("fc1.weight", False)),
        "fc2_moved": bool(battery["layer_moved"].get("fc2.weight", False)),
        "tiny_current_overfit": bool(tiny_cur["overfit"]),
        "tiny_restored_overfit": bool(tiny_wide["overfit"]),
        "claimed_n_is_train_sized": claimed_n > honest_n,
        "claimed_val_better_than_held_out": float(battery["claimed_val_loss"]) + 0.05
        < float(battery["held_out_loss"]),
        "claimed_model_training": bool(battery["claimed_model_training"]),
        "honest_held_out_gap": abs(float(battery["honest_val_loss"]) - float(battery["held_out_loss"])),
        "underfit_full_data": bool(
            math.isfinite(last) and last > 0.25 and float(battery["honest_val_accuracy"]) < FIT_VAL_ACCURACY
        ),
        "train_loss_stayed_high": bool(math.isfinite(last) and last > 0.9),
    }


def rank_hypotheses(signals: dict[str, object]) -> tuple[str, ...]:
    """Deterministic ranking. Scores are evidence weights, not a trained model."""

    scores = {name: 0.0 for name in HYPOTHESIS_IDS}
    if signals.get("labels_disagree_with_clusters"):
        scores["labels_corrupted"] += 5.0
    if signals.get("features_misfcaled"):
        scores["features_misfcaled"] += 5.0
    if signals.get("loss_non_finite") or signals.get("loss_oscillates"):
        scores["lr_too_high"] += 5.0
    if signals.get("fc1_blocked") or (
        (not signals.get("fc1_moved")) and signals.get("fc2_moved")
    ):
        scores["blocked_gradient_path"] += 5.0
    restored_ok = bool(signals.get("tiny_restored_overfit"))
    if signals.get("train_loss_flat") and restored_ok:
        scores["lr_too_low"] += 5.0
    elif signals.get("train_loss_flat"):
        scores["lr_too_low"] += 2.0
    clean_underfit = (
        signals.get("underfit_full_data")
        and restored_ok
        and not signals.get("labels_disagree_with_clusters")
        and not signals.get("fc1_blocked")
        and not signals.get("features_misfcaled")
        and not signals.get("loss_oscillates")
        and not signals.get("train_loss_flat")
    )
    if clean_underfit and signals.get("train_loss_stayed_high"):
        scores["regularization_too_strong"] += 5.0
        scores["capacity_too_small"] += 2.0
    elif clean_underfit:
        scores["capacity_too_small"] += 5.0
    if signals.get("claimed_model_training"):
        scores["eval_train_mode"] += 5.0
    if signals.get("claimed_n_is_train_sized") or (
        signals.get("claimed_val_better_than_held_out") and signals.get("honest_held_out_gap", 99) < 0.2
    ):
        scores["eval_split_leakage"] += 4.0
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(name for name, score in ranked if score > 0.0) + tuple(
        name for name, score in ranked if score == 0.0
    )


def regression_flags(diag: DiagnosticRun) -> dict[str, bool]:
    agreement = cluster_label_agreement(diag.fault.splits)
    first = diag.train_losses[0] if diag.train_losses else float("nan")
    last = diag.train_losses[-1] if diag.train_losses else float("nan")
    trainable_moved = True
    for name, moved in diag.layer_moved.items():
        requires = True
        if name.startswith("fc1") and not diag.grad_report["fc1_requires_grad"]:
            requires = False
        if requires and not moved:
            trainable_moved = False
    return {
        "finite_losses": _losses_finite(diag.train_losses) and math.isfinite(diag.held_out.loss),
        "train_loss_dropped": bool(
            math.isfinite(first) and math.isfinite(last) and last < TRAIN_DROP_RATIO * first
        ),
        "held_out_in_eval_mode": not bool(diag.held_out.model_training),
        "claimed_eval_in_eval_mode": not bool(diag.claimed_val.model_training),
        "claimed_n_matches_val": int(diag.claimed_val.n) == int(diag.honest_val.n),
        "trainable_layers_moved": bool(trainable_moved),
        "labels_match_clusters": float(agreement["train_agreement"]) >= 0.85,
        "features_comparable_scale": feature_scale_ratio(diag.fault.splits) < 10.0,
        "fc1_receiving_gradients": bool(diag.grad_report["fc1_requires_grad"])
        and (not bool(diag.grad_report["fc1_grad_is_none"])),
        "fits_training_data": bool(diag.honest_val.accuracy >= FIT_VAL_ACCURACY)
        and bool(math.isfinite(last) and math.isfinite(first) and last < first),
        "train_loss_not_oscillating": not _loss_oscillates(diag.train_losses),
    }


def invariant_failures(diag: DiagnosticRun) -> tuple[str, ...]:
    flags = regression_flags(diag)
    return tuple(name for name, ok in flags.items() if not ok)


def healthy_invariants_hold(diag: DiagnosticRun) -> bool:
    return len(invariant_failures(diag)) == 0


def cheapest_discriminator(hypothesis: str) -> str:
    mapping = {
        "labels_corrupted": "cluster_label_agreement on the train split",
        "features_misfcaled": "per-column feature standard deviations",
        "lr_too_high": "finite/oscillating train-loss trace at fixed architecture",
        "lr_too_low": "tiny-subset overfit at the current learning rate versus a restored rate",
        "blocked_gradient_path": "per-parameter requires_grad, .grad presence, and layer movement",
        "capacity_too_small": "tiny-subset overfit at current width versus TRAIN_HIDDEN",
        "regularization_too_strong": "tiny-subset overfit with dropout disabled",
        "eval_train_mode": "model.training and dropout-sensitive logits on a frozen checkpoint",
        "eval_split_leakage": "claimed val n and loss versus the protected val indices",
    }
    if hypothesis not in mapping:
        raise ValueError(f"unknown hypothesis {hypothesis!r}")
    return mapping[hypothesis]


def chaos_day(seed: int, *, reveal: bool = False) -> dict[str, object]:
    """Phase-end hidden fault. Default view is symptoms only."""

    defect = defect_for_seed(seed)
    fault = prepare_fault(defect, seed=seed, hidden=not reveal)
    diag = run_prepared(fault)
    payload = {
        "symptoms": public_symptoms(diag),
        "seed": int(seed),
        "fault": fault,
        "diag": diag,
    }
    if reveal:
        payload["defect"] = defect
        payload["category"] = category_for_defect(defect)
    return payload


def hidden_fault_run(seed: int, *, reveal: bool = False) -> dict[str, object]:
    return chaos_day(seed, reveal=reveal)


def signature_for_defect(defect: str) -> tuple[str, ...]:
    """Expected healthy-invariant failures for a named defect. Used by regression tests."""

    key = _normalize_defect(defect)
    mapping = {
        "none": (),
        "label_shuffle": ("labels_match_clusters",),
        "feature_scale": ("features_comparable_scale",),
        "lr_high": ("train_loss_dropped", "train_loss_not_oscillating"),
        "lr_low": ("train_loss_dropped", "fits_training_data"),
        "frozen_layer": ("fc1_receiving_gradients",),
        "tiny_hidden": ("fits_training_data",),
        "crushing_dropout": ("fits_training_data",),
        "train_mode_eval": ("claimed_eval_in_eval_mode",),
        "val_leakage": ("claimed_n_matches_val",),
    }
    return mapping[key]


def needs_retrain(fault: PreparedFault) -> bool:
    """Evaluation defects are repaired on the frozen checkpoint, not by retraining."""

    return fault.original_defect not in {"train_mode_eval", "val_leakage"}


def repair_and_verify(diag: DiagnosticRun) -> dict[str, object]:
    """Drive the smallest repair from the broken objects and re-check evidence."""

    m25 = _load_m25()
    fault = diag.fault
    if not needs_retrain(fault):
        before = _snapshot_params(diag.train_run.model)
        repair_prepared(fault)
        repaired_claimed = claimed_eval(diag.train_run, fault)
        after = _snapshot_params(diag.train_run.model)
        return {
            "retrained": False,
            "same_model": True,
            "same_splits": True,
            "parameters_unchanged": m25.parameters_equal(before, after),
            "claimed": repaired_claimed,
            "diag": diag,
            "failures": tuple(
                name
                for name, ok in (
                    ("claimed_eval_in_eval_mode", not bool(repaired_claimed.model_training)),
                    ("claimed_n_matches_val", int(repaired_claimed.n) == int(diag.honest_val.n)),
                )
                if not ok
            ),
        }
    repair_prepared(fault)
    repaired = run_prepared(fault)
    return {
        "retrained": True,
        "same_model": False,
        "same_splits": repaired.fault.splits is fault.splits,
        "parameters_unchanged": False,
        "claimed": repaired.claimed_val,
        "diag": repaired,
        "failures": invariant_failures(repaired),
    }
