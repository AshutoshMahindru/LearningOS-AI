"""Inference controls and adaptation rubric for Mission M32.

M32 consumes an already-produced language-model identity (M31
``StageAwareCheckpoint``: ``training_time=False``, ``inference_ready=True``)
and opens logits-to-token selection: greedy decoding, temperature, top-k /
top-p teaching filters, seeds, stop / max-token budgets, prompt/context
effects, and a decision rubric among prompt, retrieval, tools, and
parameter change.

The canonical decoder is a local deterministic score fixture, not a
downloaded model. Semantic search, RAG, and tool execution remain
deferred to M33, M34, and M37.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping

SEED = 3201
SEED_OTHER = 3203
INFERENCE_VERSION = "v07-teaching-inference-1"
FILTER_VALUE = float("-inf")
VOCAB = ("alpha", "beta", "gamma", "stop")
STOP_ID = 3
VOCAB_SIZE = 4

# Independent arithmetic fixtures (no NumPy, no model hub).
# GREEDY_LOGITS argmax is index 1 ("beta"). Ties break toward the lowest index.
GREEDY_LOGITS = (1.0, 3.0, 2.0, 0.0)
GREEDY_INDEX = 1
TIE_LOGITS = (2.0, 2.0, 1.0, 0.0)
TIE_INDEX = 0

# softmax(log 3, 0) at T=1 is (0.75, 0.25); at T=0.5 is (0.9, 0.1).
TEMP_LOGITS = (math.log(3.0), 0.0)
TEMPERATURES = (0.5, 1.0, 2.0)

# softmax(log 10, log 6, log 3, log 1) = (0.5, 0.3, 0.15, 0.05).
FILTER_COUNTS = (10, 6, 3, 1)
FILTER_LOGITS = tuple(math.log(float(count)) for count in FILTER_COUNTS)
TOP_K = 2
TOP_P_KEEP_TWO = 0.80
TOP_P_KEEP_THREE = 0.81
TOP_P_KEEP_ONE = 0.50

# Fresh no-AI numbers: softmax(log 6, log 3, log 1) = (0.6, 0.3, 0.1).
TRANSFER_LOGITS = (math.log(6.0), math.log(3.0), math.log(1.0))
TRANSFER_GREEDY_INDEX = 0

PROMPT_A = (0,)
PROMPT_B = (1,)

SUPPORTED_DEFECTS = ("none", "uncontrolled_settings", "wrong_adaptation")
ADAPTATION_ROUTES = ("prompt", "retrieval", "tools", "parameters")
SIGNAL_KEYS = (
    "freshness",
    "private_knowledge",
    "deterministic_computation",
    "style_or_format",
    "needs_weight_change",
)

TRACE_CHECKPOINTS = (
    "checkpoint_id",
    "prompt_ids",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "max_tokens",
    "stop_token_ids",
    "do_sample",
    "generated_ids",
    "stop_reason",
)

SCALE_LIMIT = (
    "Teaching scale: a 4-token local score table, CPU stdlib sampling, no "
    "downloaded weights, no paid API, no search index, no tool executor. "
    "Temperature and top-k/p are not quality knobs in isolation. "
    "This fixture is not a production decoder."
)

SYSTEM_MAP = (
    "M31 StageAwareCheckpoint (training_time=False, inference_ready=True)\n"
    "  -> prompt / context ids\n"
    "  -> local score table -> logits\n"
    "  -> temperature scale\n"
    "  -> top-k then top-p teaching filters\n"
    "  -> greedy argmax OR seeded sample\n"
    "  -> stop token / max_tokens\n"
    "  -> tokens + InferenceConfig evidence\n"
    "adaptation rubric (decision only): prompt -> retrieval -> tools -> parameters"
)

ADAPTATION_HIERARCHY = (
    "1. prompt/context: change what a frozen checkpoint sees (style, format, few-shot).\n"
    "2. retrieval: supply current or corpus-private facts; do not fine-tune for freshness.\n"
    "3. tools: deterministic computation and systems of record (M37 executes them).\n"
    "4. parameters: instruction-tuning, LoRA/QLoRA, or full fine-tune when behavior "
    "must live in weights after cheaper levers fail or cannot hold."
)

FILTER_ORDER = "temperature scale -> top-k mask -> top-p mask -> softmax -> sample"

# Prefix -> next-token logits. Greedy from PROMPT_A emits beta, gamma, stop.
PREFIX_SCORES: dict[tuple[int, ...], tuple[float, ...]] = {
    (): (3.0, 1.0, 0.0, -2.0),
    (0,): (0.0, 3.0, 1.0, -1.0),
    (0, 1): (0.0, 0.0, 3.0, 1.0),
    (0, 1, 2): (-2.0, 0.0, 0.0, 4.0),
    (1,): (0.0, 0.0, 3.0, 0.0),
    (1, 2): (-1.0, 0.0, 0.0, 4.0),
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "email_tone": {
        "problem": "Answers are factually fine but too casual for a regulated client.",
        "signals": {
            "freshness": False,
            "private_knowledge": False,
            "deterministic_computation": False,
            "style_or_format": True,
            "needs_weight_change": False,
        },
    },
    "vendor_policy": {
        "problem": "The answer must cite this company's private vendor policy PDF.",
        "signals": {
            "freshness": False,
            "private_knowledge": True,
            "deterministic_computation": False,
            "style_or_format": False,
            "needs_weight_change": False,
        },
    },
    "invoice_vat": {
        "problem": "The user needs the exact VAT total on a 14-line invoice.",
        "signals": {
            "freshness": False,
            "private_knowledge": False,
            "deterministic_computation": True,
            "style_or_format": False,
            "needs_weight_change": False,
        },
    },
    "always_refuse_competitor_praise": {
        "problem": "The product must refuse competitor praise across every surface, not one prompt.",
        "signals": {
            "freshness": False,
            "private_knowledge": False,
            "deterministic_computation": False,
            "style_or_format": False,
            "needs_weight_change": True,
        },
    },
    "stale_site_hours": {
        "problem": "Answers still quote last year's holiday hours for Site B.",
        "signals": {
            "freshness": True,
            "private_knowledge": True,
            "deterministic_computation": False,
            "style_or_format": False,
            "needs_weight_change": False,
        },
    },
}

TRANSFER_MENU_SIGNALS = {
    "freshness": True,
    "private_knowledge": True,
    "deterministic_computation": False,
    "style_or_format": False,
    "needs_weight_change": False,
}


class LiveAdapterUnavailable(RuntimeError):
    """Raised when a live model is requested; canonical tests must not need one."""


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M32 optional NumPy path requires requirements/m32.txt") from exc
    return np


def _load_sibling(mission_id: str, filename: str, module_name: str):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent.parent / mission_id / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load missions/{mission_id}/{filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_m31():
    """Load the trusted M31 training boundary; do not rewrite the objective."""

    return _load_sibling("M31", "llm_training_core.py", "_learningos_m31_llm_training_core")


def _normalize_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none"}:
        return "none"
    if key not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return key


def _as_logits(logits) -> tuple[float, ...]:
    values = tuple(float(value) for value in logits)
    if not values:
        raise ValueError("logits must be non-empty")
    return values


def greedy_token(logits) -> int:
    """Argmax with lowest-index tie-break. Invariant to temperature for T > 0."""

    values = _as_logits(logits)
    best = max(values)
    for index, value in enumerate(values):
        if value == best:
            return index
    raise RuntimeError("argmax failed")


def softmax_probs(logits, temperature: float = 1.0) -> tuple[float, ...]:
    """Numerically stable softmax after dividing logits by temperature."""

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0; use greedy_token for the T→0 limit")
    values = _as_logits(logits)
    scaled = [value / float(temperature) for value in values]
    peak = max(scaled)
    shifted = [math.exp(value - peak) for value in scaled]
    total = sum(shifted)
    if total <= 0.0:
        raise ValueError("softmax total is not positive")
    return tuple(mass / total for mass in shifted)


def entropy(probs) -> float:
    masses = tuple(float(mass) for mass in probs)
    if abs(sum(masses) - 1.0) > 1e-9:
        raise ValueError("probs must sum to 1")
    return -sum(mass * math.log(mass) for mass in masses if mass > 0.0)


def apply_top_k(logits, top_k: int | None, *, filter_value: float = FILTER_VALUE) -> tuple[float, ...]:
    values = _as_logits(logits)
    if top_k is None or int(top_k) >= len(values):
        return values
    k = int(top_k)
    if k < 1:
        raise ValueError("top_k must be >= 1 when provided")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    kept = set(order[:k])
    return tuple(values[index] if index in kept else float(filter_value) for index in range(len(values)))


def apply_top_p(logits, top_p: float | None, *, filter_value: float = FILTER_VALUE) -> tuple[float, ...]:
    """Keep the smallest descending prefix whose cumulative softmax mass >= top_p."""

    values = _as_logits(logits)
    if top_p is None:
        return values
    bound = float(top_p)
    if bound <= 0.0 or bound > 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if bound >= 1.0:
        return values
    finite = tuple(value if math.isfinite(value) else float("-inf") for value in values)
    if not any(math.isfinite(value) for value in finite):
        raise ValueError("top-p requires at least one finite logit")
    probs = softmax_probs(finite, temperature=1.0)
    order = sorted(range(len(probs)), key=lambda index: (-probs[index], index))
    cumulative = 0.0
    kept: list[int] = []
    for index in order:
        kept.append(index)
        cumulative += probs[index]
        if cumulative >= bound:
            break
    kept_set = set(kept)
    return tuple(values[index] if index in kept_set else float(filter_value) for index in range(len(values)))


def prepare_distribution(
    logits,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Teaching filter order: temperature scale, top-k, top-p, then softmax."""

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0; use greedy_token for the T→0 limit")
    values = _as_logits(logits)
    scaled = tuple(value / float(temperature) for value in values)
    filtered = apply_top_k(scaled, top_k)
    filtered = apply_top_p(filtered, top_p)
    return filtered, softmax_probs(filtered, temperature=1.0)


def sample_token(probs, rng: random.Random) -> int:
    masses = tuple(float(mass) for mass in probs)
    if abs(sum(masses) - 1.0) > 1e-9:
        raise ValueError("probs must sum to 1")
    draw = rng.random()
    total = 0.0
    for index, mass in enumerate(masses):
        total += mass
        if draw < total:
            return index
    return len(masses) - 1


def sample_token_with_seed(probs, seed: int, *, offset: int = 0) -> int:
    rng = random.Random(int(seed))
    for _ in range(int(offset)):
        rng.random()
    return sample_token(probs, rng)


def local_logits(prefix, scores: Mapping[tuple[int, ...], tuple[float, ...]] | None = None) -> tuple[float, ...]:
    table = PREFIX_SCORES if scores is None else scores
    key = tuple(int(token) for token in prefix)
    if key in table:
        return tuple(float(value) for value in table[key])
    return tuple(float(value) for value in table[()])


def numpy_softmax(logits, temperature: float = 1.0):
    """Optional NumPy parity helper. Required tests use softmax_probs instead."""

    np = _require_numpy()
    values = np.asarray(_as_logits(logits), dtype=float)
    scaled = values / float(temperature)
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_id: str
    prompt_ids: tuple[int, ...]
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = SEED
    max_tokens: int = 8
    stop_token_ids: tuple[int, ...] = (STOP_ID,)
    do_sample: bool = False
    adaptation_stage: str = "pretrained"
    version: str = INFERENCE_VERSION
    training_time: bool = False

    def __post_init__(self) -> None:
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        if int(self.max_tokens) < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.training_time:
            raise ValueError("inference config cannot have training_time=True")
        if self.do_sample and self.top_k is not None and int(self.top_k) < 1:
            raise ValueError("top_k must be >= 1 when provided")
        if self.top_p is not None and not (0.0 < float(self.top_p) <= 1.0):
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class DecodeStep:
    prefix: tuple[int, ...]
    logits: tuple[float, ...]
    filtered_logits: tuple[float, ...]
    probs: tuple[float, ...]
    token_id: int
    greedy: bool
    temperature: float
    seed: int


@dataclass(frozen=True)
class GenerationTrace:
    defect: str
    config: InferenceConfig
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    steps: tuple[DecodeStep, ...]
    stop_reason: str
    checkpoint_id: str
    version: str = INFERENCE_VERSION

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.generated_ids

    def checkpoint_field(self, name: str):
        if name not in TRACE_CHECKPOINTS:
            raise KeyError(name)
        if name == "checkpoint_id":
            return self.checkpoint_id
        if name == "prompt_ids":
            return self.prompt_ids
        if name == "generated_ids":
            return self.generated_ids
        if name == "stop_reason":
            return self.stop_reason
        return getattr(self.config, name)


@dataclass(frozen=True)
class AdaptationDecision:
    case_id: str
    problem: str
    signals: dict[str, bool]
    chosen_route: str
    defect: str = "none"
    rationale: str = ""
    version: str = INFERENCE_VERSION


@dataclass(frozen=True)
class FailureTrace:
    defect: str
    claim: str
    left: GenerationTrace | None = None
    right: GenerationTrace | None = None
    decision: AdaptationDecision | None = None
    mismatched_fields: tuple[str, ...] = ()
    reference_config: InferenceConfig | None = None
    version: str = INFERENCE_VERSION
    audit: dict[str, Any] = field(default_factory=dict)


def attach_m31_checkpoint():
    """Identity-only consume of M31's produced-LM boundary.

    M32 does not reload the (V, V) teaching table or retrain. The local
    4-token score fixture is the decoder; the checkpoint records lineage.
    """

    m31 = _load_m31()
    lineage = m31.SplitLineage(
        dataset_version=m31.DATASET_VERSION,
        train_ids=("d01", "d02", "d03", "d04", "d05", "d06"),
        eval_ids=("e01", "e02"),
        tokenizer_name="m27-word-teaching",
        tokenizer_version="v06.1",
        tokenizer_scheme=m31.TOKENIZER_SCHEME,
    )
    checkpoint = m31.make_checkpoint(
        lineage=lineage,
        seed=m31.SEED,
        steps=m31.STEPS,
        context_length=m31.TEACHING_CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        defect="none",
        alignment="correct",
        train_loss=0.0,
        eval_loss=0.0,
    )
    if checkpoint.training_time:
        raise ValueError("M31 checkpoint is still marked training_time")
    if not checkpoint.inference_ready:
        raise ValueError("M31 checkpoint is not inference_ready")
    return checkpoint


def training_time_boundary() -> str:
    return _load_m31().TRAINING_TIME_BOUNDARY


def _require_inference_checkpoint(checkpoint) -> None:
    if bool(getattr(checkpoint, "training_time", False)):
        raise ValueError("refuse to sample a training_time checkpoint; finish training first")
    if not bool(getattr(checkpoint, "inference_ready", False)):
        raise ValueError("checkpoint is not inference_ready")


def make_config(
    *,
    checkpoint=None,
    prompt_ids: tuple[int, ...] = PROMPT_A,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = SEED,
    max_tokens: int = 8,
    stop_token_ids: tuple[int, ...] = (STOP_ID,),
    do_sample: bool = False,
) -> InferenceConfig:
    ckpt = attach_m31_checkpoint() if checkpoint is None else checkpoint
    _require_inference_checkpoint(ckpt)
    return InferenceConfig(
        checkpoint_id=str(ckpt.checkpoint_id),
        prompt_ids=tuple(int(token) for token in prompt_ids),
        temperature=float(temperature),
        top_k=None if top_k is None else int(top_k),
        top_p=None if top_p is None else float(top_p),
        seed=int(seed),
        max_tokens=int(max_tokens),
        stop_token_ids=tuple(int(token) for token in stop_token_ids),
        do_sample=bool(do_sample),
        adaptation_stage=str(getattr(ckpt, "adaptation_stage", "pretrained")),
        version=INFERENCE_VERSION,
        training_time=False,
    )


def config_fingerprint(config: InferenceConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def config_as_evidence(config: InferenceConfig) -> dict[str, Any]:
    """Inference configuration is reproducibility evidence, not a quality score."""

    return {
        "checkpoint_id": config.checkpoint_id,
        "prompt_ids": config.prompt_ids,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "stop_token_ids": config.stop_token_ids,
        "do_sample": config.do_sample,
        "adaptation_stage": config.adaptation_stage,
        "version": config.version,
        "training_time": config.training_time,
        "fingerprint": config_fingerprint(config),
        "filter_order": FILTER_ORDER,
    }


def run_inference(
    prompt_ids: tuple[int, ...] | None = None,
    config: InferenceConfig | None = None,
    *,
    scores: Mapping[tuple[int, ...], tuple[float, ...]] | None = None,
    defect: str | None = "none",
) -> GenerationTrace:
    """Prompt → local scores → filters → greedy or sample → stop → metadata.

    Fallback: missing checkpoint, training_time, empty logits, or live-adapter
    requests fail closed. This wrapper does not search, cite, or call tools.
    """

    defect_key = _normalize_defect(defect)
    cfg = config if config is not None else make_config(prompt_ids=prompt_ids or PROMPT_A)
    prefix = tuple(int(token) for token in (prompt_ids if prompt_ids is not None else cfg.prompt_ids))
    if cfg.training_time:
        raise ValueError("run_inference consumes a produced checkpoint; training_time must be False")
    rng = random.Random(cfg.seed)
    generated: list[int] = []
    steps: list[DecodeStep] = []
    stop_ids = set(cfg.stop_token_ids)
    stop_reason = "max_tokens"
    for _ in range(int(cfg.max_tokens)):
        logits = local_logits(prefix, scores=scores)
        if cfg.do_sample:
            filtered, probs = prepare_distribution(
                logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
            token = sample_token(probs, rng)
        else:
            filtered = logits
            probs = softmax_probs(logits, temperature=1.0)
            token = greedy_token(logits)
        steps.append(
            DecodeStep(
                prefix=prefix,
                logits=logits,
                filtered_logits=filtered,
                probs=probs,
                token_id=token,
                greedy=not cfg.do_sample,
                temperature=cfg.temperature,
                seed=cfg.seed,
            )
        )
        generated.append(token)
        prefix = prefix + (token,)
        if token in stop_ids:
            stop_reason = "stop_token"
            break
    return GenerationTrace(
        defect=defect_key,
        config=cfg,
        prompt_ids=tuple(int(token) for token in (prompt_ids if prompt_ids is not None else cfg.prompt_ids)),
        generated_ids=tuple(generated),
        steps=tuple(steps),
        stop_reason=stop_reason,
        checkpoint_id=cfg.checkpoint_id,
    )


def sample_once(
    logits,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = SEED,
    do_sample: bool = True,
) -> DecodeStep:
    """Single-step fixture used by greedy / temperature / filter / seed labs."""

    values = _as_logits(logits)
    if do_sample:
        filtered, probs = prepare_distribution(
            values, temperature=temperature, top_k=top_k, top_p=top_p
        )
        token = sample_token_with_seed(probs, seed)
        greedy = False
    else:
        filtered = values
        probs = softmax_probs(values, temperature=1.0)
        token = greedy_token(values)
        greedy = True
    return DecodeStep(
        prefix=(),
        logits=values,
        filtered_logits=filtered,
        probs=probs,
        token_id=token,
        greedy=greedy,
        temperature=float(temperature),
        seed=int(seed),
    )


def first_divergence(left: GenerationTrace, right: GenerationTrace) -> str | None:
    for name in TRACE_CHECKPOINTS:
        if left.checkpoint_field(name) != right.checkpoint_field(name):
            return name
    return None


def mismatched_config_fields(left: InferenceConfig, right: InferenceConfig) -> tuple[str, ...]:
    names = (
        "checkpoint_id",
        "prompt_ids",
        "temperature",
        "top_k",
        "top_p",
        "seed",
        "max_tokens",
        "stop_token_ids",
        "do_sample",
    )
    return tuple(name for name in names if getattr(left, name) != getattr(right, name))


def compare_outputs_naive(left: GenerationTrace, right: GenerationTrace) -> str:
    """Illegal comparison: treats token disagreement as a model change."""

    if left.generated_ids != right.generated_ids:
        return "model_changed"
    return "same_model"


def compare_outputs(left: GenerationTrace, right: GenerationTrace) -> str:
    if left.checkpoint_id != right.checkpoint_id:
        return "checkpoint_changed"
    diffs = mismatched_config_fields(left.config, right.config)
    if diffs:
        return "uncontrolled_settings"
    if left.generated_ids != right.generated_ids:
        return "nondeterministic_or_bug"
    return "reproduced"


def decide_adaptation(signals: Mapping[str, bool]) -> str:
    """Cheapest sufficient lever. Fine-tuning is last, not first.

    This is a teaching default for V07, not the learner's ADR.
    """

    flags = {key: bool(signals.get(key, False)) for key in SIGNAL_KEYS}
    unknown = set(signals) - set(SIGNAL_KEYS)
    if unknown:
        raise ValueError(f"unknown adaptation signals {sorted(unknown)}")
    if flags["deterministic_computation"]:
        return "tools"
    if flags["freshness"] or flags["private_knowledge"]:
        return "retrieval"
    if flags["needs_weight_change"]:
        return "parameters"
    if flags["style_or_format"]:
        return "prompt"
    return "prompt"


def decide_scenario(case_id: str) -> AdaptationDecision:
    if case_id not in SCENARIOS:
        raise ValueError(f"unknown scenario {case_id!r}")
    spec = SCENARIOS[case_id]
    route = decide_adaptation(spec["signals"])
    return AdaptationDecision(
        case_id=case_id,
        problem=str(spec["problem"]),
        signals=dict(spec["signals"]),
        chosen_route=route,
        defect="none",
        rationale=f"rubric chose {route}",
    )


def optional_live_complete(prompt: str, config: InferenceConfig) -> GenerationTrace:
    """Optional live-model adapter. Canonical validation must not call a network.

    The function exists so later missions can see where a provider would plug
    in. Live models are not required. It always fails closed here.
    """

    del prompt, config
    raise LiveAdapterUnavailable(
        "M32 canonical path uses local logits fixtures; live models are optional "
        "and not required for validation"
    )


def pipeline_with_defect(*, defect: str, checkpoint=None) -> FailureTrace:
    """Controlled-failure entry: one named defect, same teaching fixture."""

    key = _normalize_defect(defect)
    if key == "none":
        raise ValueError("pipeline_with_defect requires a named defect")
    ckpt = attach_m31_checkpoint() if checkpoint is None else checkpoint
    if key == "uncontrolled_settings":
        left_cfg = make_config(
            checkpoint=ckpt,
            prompt_ids=PROMPT_A,
            temperature=1.0,
            seed=SEED,
            do_sample=True,
            max_tokens=4,
        )
        right_cfg = make_config(
            checkpoint=ckpt,
            prompt_ids=PROMPT_A,
            temperature=1.7,
            seed=SEED_OTHER,
            do_sample=True,
            max_tokens=4,
        )
        left = run_inference(left_cfg.prompt_ids, left_cfg, defect=key)
        right = run_inference(right_cfg.prompt_ids, right_cfg, defect=key)
        return FailureTrace(
            defect=key,
            claim="model_changed",
            left=left,
            right=right,
            mismatched_fields=mismatched_config_fields(left_cfg, right_cfg),
            reference_config=left_cfg,
            audit={
                "naive_compare": compare_outputs_naive(left, right),
                "controlled_compare": compare_outputs(left, right),
            },
        )
    if key == "wrong_adaptation":
        spec = SCENARIOS["stale_site_hours"]
        decision = AdaptationDecision(
            case_id="stale_site_hours",
            problem=str(spec["problem"]),
            signals=dict(spec["signals"]),
            chosen_route="parameters",
            defect=key,
            rationale="team proposed fine-tune because answers were wrong",
        )
        return FailureTrace(
            defect=key,
            claim="fine_tune_for_freshness",
            decision=decision,
            audit={"rubric_would_choose": decide_adaptation(spec["signals"])},
        )
    raise ValueError(f"unsupported defect {defect!r}")


def repair_run(trace: FailureTrace) -> FailureTrace:
    """Recompute from the defective object's checkpoint, signals, and reference config.

    Does not start a second unrelated happy-path run from module defaults.
    """

    if trace.defect == "uncontrolled_settings":
        if trace.reference_config is None or trace.left is None:
            raise ValueError("defective trace is missing reference_config")
        cfg = trace.reference_config
        left = run_inference(cfg.prompt_ids, cfg, defect="none")
        right = run_inference(cfg.prompt_ids, cfg, defect="none")
        return FailureTrace(
            defect="none",
            claim="controls_held",
            left=left,
            right=right,
            mismatched_fields=(),
            reference_config=cfg,
            audit={
                "naive_compare": compare_outputs_naive(left, right),
                "controlled_compare": compare_outputs(left, right),
                "reused_checkpoint_id": cfg.checkpoint_id,
                "reused_seed": cfg.seed,
                "reused_temperature": cfg.temperature,
            },
        )
    if trace.defect == "wrong_adaptation":
        if trace.decision is None:
            raise ValueError("defective trace is missing the adaptation decision")
        route = decide_adaptation(trace.decision.signals)
        repaired = AdaptationDecision(
            case_id=trace.decision.case_id,
            problem=trace.decision.problem,
            signals=dict(trace.decision.signals),
            chosen_route=route,
            defect="none",
            rationale=f"smallest sufficient lever is {route}, not parameters",
        )
        return FailureTrace(
            defect="none",
            claim="adaptation_repaired",
            decision=repaired,
            audit={"from_case": trace.decision.case_id, "from_wrong_route": trace.decision.chosen_route},
        )
    raise ValueError(f"repair_run expects a named defect, not {trace.defect!r}")


def observability_report(trace: GenerationTrace) -> dict[str, Any]:
    return {
        "version": INFERENCE_VERSION,
        "checkpoint_id": trace.checkpoint_id,
        "defect": trace.defect,
        "prompt_ids": trace.prompt_ids,
        "generated_ids": trace.generated_ids,
        "stop_reason": trace.stop_reason,
        "config": config_as_evidence(trace.config),
        "training_time": trace.config.training_time,
        "scale_limit": SCALE_LIMIT,
        "training_time_boundary": training_time_boundary(),
        "adaptation_hierarchy": ADAPTATION_HIERARCHY,
        "handoff": (
            "M33/M34/M37 receive InferenceConfig (checkpoint, prompt, temperature, "
            "top-k/p, seed, stop, max_tokens) and this adaptation hierarchy; they "
            "do not receive a search index, RAG pack, or tool executor from M32."
        ),
    }


def token_names(ids) -> tuple[str, ...]:
    return tuple(VOCAB[int(token)] for token in ids)
