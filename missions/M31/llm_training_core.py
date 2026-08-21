"""Causal next-token teaching pipeline for Mission M31.

M31 attaches a causal language-model objective to the M27-M30 stack:
tokenize, window, shift input/target pairs, train a tiny bigram table,
record split lineage, and emit a stage-aware checkpoint.

The teaching model is a (V, V) next-token score table, not a stacked
transformer trainer. M30 remains a frozen block you can unembed for
shape inspection. Decoding, candidate filters, RAG, and production
recipes stay deferred to M32+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

SEED = 3101
STEPS = 60
LEARNING_RATE = 0.75
TEACHING_CONTEXT_LENGTH = 16
SHORT_CONTEXT_LENGTH = 4
TRAINING_VERSION = "v07-teaching-lm-1"
DATASET_VERSION = "v07-teaching-corpus-1"
TOKENIZER_SCHEME = "word"
ADD_SPECIAL_TOKENS = True
LEAK_DOC_ID = "e02"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

SUPPORTED_DEFECTS = ("none", "correct", "target_shift_wrong", "held_out_leak")
SUPPORTED_SPLITS = ("train", "eval")

# Independent arithmetic fixtures (no tokenizer, no NumPy).
SHIFT_TOKENS = (10, 20, 30, 40)
SHIFT_INPUTS = (10, 20, 30)
SHIFT_TARGETS = (20, 30, 40)
WRONG_SHIFT_TARGETS = (10, 20, 30)
CONTEXT_SEQUENCE = (1, 2, 3, 4, 5)
NLL_LOGITS = (0.0, 0.0)
NLL_TARGET = 1
NLL_THREE_LOGITS = (0.0, 0.0, 0.0)
NLL_THREE_TARGET = 0
TRANSFER_TOKENS = (4, 8, 15, 16, 23)
TRANSFER_WRONG_TARGETS = (4, 8, 15, 16)
TRANSFER_NLL_LOGITS = (0.0, 0.0, 0.0)
TRANSFER_NLL_TARGET = 0
TRANSFER_TRAIN_IDS = ("d01", "d02", "e02")
TRANSFER_EVAL_IDS = ("e01", "e02")

TRACE_CHECKPOINTS = (
    "train_doc_ids",
    "eval_doc_ids",
    "leaked_ids",
    "tokens",
    "window",
    "inputs",
    "targets",
    "n_targets",
)

SCALE_LIMIT = (
    "Teaching scale: one (V, V) bigram score table, eight synthetic documents, "
    "CPU NumPy SGD, no stacked-block trainer, no downloaded weights, no paid run. "
    "Falling loss here is not evidence of a production LLM."
)

TRAINING_TIME_BOUNDARY = (
    "Training time updates weights under a declared objective and a declared "
    "split. Inference time consumes a produced checkpoint; token selection is M32."
)

SYSTEM_MAP = (
    "tokens --M27--> ids --window--> ids[:C]\n"
    "ids[:C] --shift--> (inputs, targets)\n"
    "ids --embed--> x --M30 block--> h --unembed--> logits   [stack; not trained here]\n"
    "inputs --bigram table--> logits --NLL vs targets--> loss --SGD--> checkpoint\n"
    "checkpoint + protected eval --> evidence\n"
    "pretraining -> adaptation/post-training -> evaluation\n"
    "produced checkpoint | inference (M32, weights frozen)"
)

STAGE_DEFINITIONS = {
    "pretraining": (
        "Update weights on unlabeled token sequences under a causal "
        "next-token objective."
    ),
    "adaptation": (
        "Update a pretrained checkpoint on supervised instruction or "
        "task pairs. Weights change; the objective is no longer raw web-text NLL."
    ),
    "post_training": (
        "Further weight or policy updates from preference data or a reward "
        "signal after supervised adaptation. System-map depth only in M31."
    ),
    "evaluation": (
        "Score a frozen split with a frozen protocol. The eval ids are not "
        "training material."
    ),
    "inference": (
        "Consume a produced checkpoint with frozen weights. How tokens are "
        "selected from logits is M32, not a training-stage knob."
    ),
}

INTERVENTIONS = {
    "next_token_on_unlabeled_corpus": "pretraining",
    "instruction_supervised_pairs": "adaptation",
    "preference_ranking_or_rlhf": "post_training",
    "score_protected_held_out": "evaluation",
    "generate_from_frozen_checkpoint": "inference",
    "change_prompt_without_weight_update": "inference",
}

TEACHING_DOCUMENTS = (
    ("d01", "train", "the cat sat on the mat"),
    ("d02", "train", "the cat sat"),
    ("d03", "train", "please inspect ticket"),
    ("d04", "train", "please process invoice"),
    ("d05", "train", "inspect ticket then stop"),
    ("d06", "train", "process invoice then stop"),
    ("e01", "eval", "the cat sat on the ticket"),
    ("e02", "eval", "customer 88 paid the invoice"),
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M31 requires numpy; install requirements/m31.txt") from exc
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


def _load_m27():
    """Load the trusted M27 tokenizer; do not rewrite encoding."""

    return _load_sibling("M27", "tokenization_core.py", "_learningos_m27_tokenization_core")


def _load_m30():
    """Load the trusted M30 block; do not rewrite residuals or attention."""

    return _load_sibling("M30", "transformer_block.py", "_learningos_m30_transformer_block")


def default_corpus_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M31" / "corpus.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing bundled corpus fixture: {path}")
    return path


def document_checksum(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none", "correct"}:
        return "none"
    if key not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return key


def _normalize_alignment(alignment: str | None) -> str:
    key = "correct" if alignment is None else str(alignment).lower()
    if key in {"", "correct", "next", "causal"}:
        return "correct"
    if key in {"identity", "wrong", "target_shift_wrong", "no_shift"}:
        return "target_shift_wrong"
    raise ValueError(f"unsupported alignment {alignment!r}")


def alignment_for_defect(defect: str | None) -> str:
    key = _normalize_defect(defect)
    if key == "target_shift_wrong":
        return "target_shift_wrong"
    return "correct"


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    authored_split: str
    checksum: str

    def __post_init__(self) -> None:
        if self.authored_split not in SUPPORTED_SPLITS:
            raise ValueError(f"authored_split must be train or eval, not {self.authored_split!r}")


@dataclass(frozen=True)
class SplitLineage:
    dataset_version: str
    train_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    leaked_ids: tuple[str, ...] = ()
    tokenizer_name: str = ""
    tokenizer_version: str = ""
    tokenizer_scheme: str = TOKENIZER_SCHEME

    @property
    def contaminated(self) -> bool:
        return bool(set(self.train_ids) & set(self.eval_ids))

    @property
    def split_hash(self) -> str:
        payload = "|".join(
            (
                self.dataset_version,
                ",".join(self.train_ids),
                ",".join(self.eval_ids),
                ",".join(self.leaked_ids),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CausalPairs:
    tokens: tuple[int, ...]
    window: tuple[int, ...]
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    alignment: str
    doc_id: str = ""
    token_pieces: tuple[str, ...] = ()

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def causal(self) -> bool:
        return pairs_are_causal(self.window, self.inputs, self.targets)


@dataclass(frozen=True)
class StageAwareCheckpoint:
    checkpoint_id: str
    dataset_version: str
    split_hash: str
    train_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    leaked_ids: tuple[str, ...]
    adaptation_stage: str
    objective: str
    seed: int
    steps: int
    context_length: int
    tokenizer_name: str
    tokenizer_version: str
    vocab_size: int
    training_time: bool
    inference_ready: bool
    scale_limit: str
    version: str = TRAINING_VERSION
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingTrace:
    defect: str
    alignment: str
    documents: tuple[Document, ...]
    authored_lineage: SplitLineage
    used_lineage: SplitLineage
    windows: tuple[CausalPairs, ...]
    seed: int
    steps: int
    learning_rate: float
    context_length: int
    vocab_size: int
    pad_id: int
    scores: Any
    train_objective_losses: tuple[float, ...]
    eval_true_losses: tuple[float, ...]
    final_train_objective_loss: float
    final_eval_true_loss: float
    unseen_pair_nll: float
    checkpoint: StageAwareCheckpoint
    tokenizer_name: str
    tokenizer_version: str

    def checkpoint_field(self, name: str):
        if name not in TRACE_CHECKPOINTS:
            raise KeyError(name)
        if name == "tokens":
            return tuple(window.tokens for window in self.windows)
        if name == "window":
            return tuple(window.window for window in self.windows)
        if name == "inputs":
            return tuple(window.inputs for window in self.windows)
        if name == "targets":
            return tuple(window.targets for window in self.windows)
        if name == "n_targets":
            return tuple(window.n_targets for window in self.windows)
        if name == "train_doc_ids":
            return self.used_lineage.train_ids
        if name == "eval_doc_ids":
            return self.used_lineage.eval_ids
        if name == "leaked_ids":
            return self.used_lineage.leaked_ids
        raise KeyError(name)


def pairs_are_causal(tokens, inputs, targets) -> bool:
    sequence = tuple(int(value) for value in tokens)
    if len(sequence) < 2:
        return False
    return (
        tuple(int(value) for value in inputs) == sequence[:-1]
        and tuple(int(value) for value in targets) == sequence[1:]
    )


def n_prediction_targets(n_tokens: int, context_length: int) -> int:
    if int(context_length) < 2:
        raise ValueError("context_length must be at least 2 to form a next-token pair")
    if int(n_tokens) < 0:
        raise ValueError("n_tokens must be non-negative")
    window = min(int(n_tokens), int(context_length))
    return max(window - 1, 0)


def apply_context_length(ids, context_length: int) -> tuple[int, ...]:
    sequence = tuple(int(value) for value in ids)
    n = n_prediction_targets(len(sequence), context_length)
    width = n + 1 if n else min(len(sequence), int(context_length))
    return sequence[:width]


def context_length_effect(ids, lengths: tuple[int, ...]) -> tuple[dict[str, int], ...]:
    sequence = tuple(int(value) for value in ids)
    rows = []
    for length in lengths:
        window = apply_context_length(sequence, int(length))
        rows.append(
            {
                "context_length": int(length),
                "window_tokens": len(window),
                "n_targets": n_prediction_targets(len(sequence), int(length)),
            }
        )
    return tuple(rows)


def shift_tokens(
    ids,
    *,
    alignment: str = "correct",
    context_length: int | None = None,
    doc_id: str = "",
    token_pieces: tuple[str, ...] = (),
) -> CausalPairs:
    """Build causal (input, target) pairs. Correct: inputs=x[:-1], targets=x[1:]."""

    sequence = tuple(int(value) for value in ids)
    if len(sequence) < 2:
        raise ValueError("need at least two tokens to form a next-token pair")
    window = (
        sequence
        if context_length is None
        else apply_context_length(sequence, context_length)
    )
    if len(window) < 2:
        raise ValueError("context window is too short for a next-token pair")
    align = _normalize_alignment(alignment)
    inputs = window[:-1]
    if align == "correct":
        targets = window[1:]
    else:
        targets = window[:-1]
    pieces = token_pieces[: len(window)] if token_pieces else ()
    return CausalPairs(
        tokens=sequence,
        window=window,
        inputs=inputs,
        targets=targets,
        alignment=align,
        doc_id=doc_id,
        token_pieces=pieces,
    )


def softmax_nll(logits, target: int) -> float:
    """Stable mean-class NLL for one position. Independent of the trainer."""

    values = tuple(float(value) for value in logits)
    if not values:
        raise ValueError("logits must be non-empty")
    index = int(target)
    if index < 0 or index >= len(values):
        raise ValueError("target is outside the logit vector")
    peak = max(values)
    shifted = [math.exp(value - peak) for value in values]
    total = sum(shifted)
    prob = shifted[index] / total
    if prob <= 0.0:
        raise ValueError("target probability is not positive")
    return -math.log(prob)


def load_teaching_tokenizer():
    tokenizer = _load_m27().load_tokenizer(scheme=TOKENIZER_SCHEME)
    if tokenizer.identity.downloaded or tokenizer.identity.network_required:
        raise RuntimeError("M31 required path must use the bundled offline tokenizer")
    return tokenizer


def encode_text(text: str, tokenizer=None, *, add_special_tokens: bool = ADD_SPECIAL_TOKENS):
    tok = load_teaching_tokenizer() if tokenizer is None else tokenizer
    return tok.encode(text, add_special_tokens=add_special_tokens)


def _document_from_row(row: dict[str, Any] | tuple[str, str, str]) -> Document:
    if isinstance(row, dict):
        doc_id = str(row["id"])
        split = str(row["split"])
        text = str(row["text"])
    else:
        doc_id, split, text = row
    return Document(
        doc_id=doc_id,
        text=text,
        authored_split=split,
        checksum=document_checksum(text),
    )


def load_teaching_corpus(path: Path | None = None) -> tuple[Document, ...]:
    payload = json.loads(default_corpus_path().read_text(encoding="utf-8") if path is None else Path(path).read_text(encoding="utf-8"))
    if str(payload.get("dataset_version")) != DATASET_VERSION:
        raise ValueError("corpus dataset_version does not match M31 teaching version")
    documents = tuple(_document_from_row(row) for row in payload["documents"])
    expected = tuple(_document_from_row(row) for row in TEACHING_DOCUMENTS)
    if tuple((doc.doc_id, doc.text, doc.authored_split) for doc in documents) != tuple(
        (doc.doc_id, doc.text, doc.authored_split) for doc in expected
    ):
        raise ValueError("datasets/M31/corpus.json drifted from TEACHING_DOCUMENTS")
    return documents


def authored_lineage(
    documents: tuple[Document, ...] | None = None,
    *,
    tokenizer_name: str = "",
    tokenizer_version: str = "",
) -> SplitLineage:
    docs = documents if documents is not None else load_teaching_corpus()
    train_ids = tuple(doc.doc_id for doc in docs if doc.authored_split == "train")
    eval_ids = tuple(doc.doc_id for doc in docs if doc.authored_split == "eval")
    if not train_ids or not eval_ids:
        raise ValueError("teaching corpus must declare both train and eval documents")
    if set(train_ids) & set(eval_ids):
        raise ValueError("authored splits must be disjoint")
    return SplitLineage(
        dataset_version=DATASET_VERSION,
        train_ids=train_ids,
        eval_ids=eval_ids,
        leaked_ids=(),
        tokenizer_name=tokenizer_name,
        tokenizer_version=tokenizer_version,
        tokenizer_scheme=TOKENIZER_SCHEME,
    )


def lineage_with_leak(lineage: SplitLineage, leak_doc_id: str = LEAK_DOC_ID) -> SplitLineage:
    leak = str(leak_doc_id)
    if leak not in lineage.eval_ids:
        raise ValueError(f"{leak} is not an authored eval document")
    if leak in lineage.train_ids:
        raise ValueError(f"{leak} is already in train_ids")
    return SplitLineage(
        dataset_version=lineage.dataset_version,
        train_ids=lineage.train_ids + (leak,),
        eval_ids=lineage.eval_ids,
        leaked_ids=(leak,),
        tokenizer_name=lineage.tokenizer_name,
        tokenizer_version=lineage.tokenizer_version,
        tokenizer_scheme=lineage.tokenizer_scheme,
    )


def used_lineage_for_defect(lineage: SplitLineage, defect: str | None) -> SplitLineage:
    key = _normalize_defect(defect)
    if key == "held_out_leak":
        return lineage_with_leak(lineage)
    return lineage


def documents_for_ids(documents: tuple[Document, ...], doc_ids: tuple[str, ...]) -> tuple[Document, ...]:
    by_id = {doc.doc_id: doc for doc in documents}
    missing = [doc_id for doc_id in doc_ids if doc_id not in by_id]
    if missing:
        raise KeyError(f"unknown document ids: {missing}")
    return tuple(by_id[doc_id] for doc_id in doc_ids)


def encode_document(document: Document, tokenizer=None, *, context_length: int | None = None, alignment: str = "correct") -> CausalPairs:
    encoding = encode_text(document.text, tokenizer)
    return shift_tokens(
        encoding.ids,
        alignment=alignment,
        context_length=context_length,
        doc_id=document.doc_id,
        token_pieces=encoding.tokens,
    )


def pair_windows(
    documents: tuple[Document, ...],
    *,
    tokenizer=None,
    context_length: int = TEACHING_CONTEXT_LENGTH,
    alignment: str = "correct",
) -> tuple[CausalPairs, ...]:
    tok = load_teaching_tokenizer() if tokenizer is None else tokenizer
    return tuple(
        encode_document(document, tok, context_length=context_length, alignment=alignment)
        for document in documents
    )


def bigram_pairs(windows: tuple[CausalPairs, ...]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for window in windows:
        pairs.extend(zip(window.inputs, window.targets, strict=True))
    return tuple(pairs)


def _flatten_windows(windows: tuple[CausalPairs, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    inputs: list[int] = []
    targets: list[int] = []
    for window in windows:
        inputs.extend(window.inputs)
        targets.extend(window.targets)
    return tuple(inputs), tuple(targets)


def observed_bigrams(windows: tuple[CausalPairs, ...]) -> frozenset[tuple[int, int]]:
    return frozenset(bigram_pairs(windows))


def attach_unembed(hidden, unembed_w):
    """Map a residual-stream tensor (..., d_model) to vocab logits (..., V)."""

    np = _require_numpy()
    stream = np.asarray(hidden, dtype=float)
    weight = np.asarray(unembed_w, dtype=float)
    if stream.ndim < 1 or weight.ndim != 2:
        raise ValueError("hidden must be a tensor and unembed_w a (d_model, V) matrix")
    if int(stream.shape[-1]) != int(weight.shape[0]):
        raise ValueError("unembed last-dim of hidden must match unembed_w rows")
    return stream @ weight


def frozen_block_logits(token_vectors, unembed_w, **block_kwargs):
    """Forward-only M30 wrap. Not the teaching trainer."""

    m30 = _load_m30()
    trace = m30.transformer_block(token_vectors, **block_kwargs)
    return attach_unembed(trace.output, unembed_w), trace


def _softmax_rows(logits):
    np = _require_numpy()
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def batch_nll(scores, inputs, targets) -> float:
    np = _require_numpy()
    inp = np.asarray(inputs, dtype=int)
    tgt = np.asarray(targets, dtype=int)
    if inp.size == 0:
        raise ValueError("cannot score an empty pair batch")
    logits = np.asarray(scores, dtype=float)[inp]
    probs = _softmax_rows(logits)
    gathered = probs[np.arange(inp.shape[0]), tgt]
    return float(-np.mean(np.log(np.clip(gathered, 1e-12, 1.0))))


def unseen_pair_nll(scores, eval_windows: tuple[CausalPairs, ...], train_windows: tuple[CausalPairs, ...]) -> float:
    known = observed_bigrams(train_windows)
    held = [(inp, tgt) for inp, tgt in bigram_pairs(eval_windows) if (inp, tgt) not in known]
    if not held:
        raise ValueError("eval windows have no pairs absent from train")
    inputs, targets = zip(*held, strict=True)
    return batch_nll(scores, inputs, targets)


def _init_scores(vocab_size: int, seed: int):
    np = _require_numpy()
    rng = np.random.default_rng(int(seed))
    # Zeros would also be deterministic; a tiny jitter keeps rows from tying.
    return rng.normal(0.0, 1e-6, size=(int(vocab_size), int(vocab_size)))


def _sgd_epoch(scores, inputs, targets, learning_rate: float) -> float:
    np = _require_numpy()
    inp = np.asarray(inputs, dtype=int)
    tgt = np.asarray(targets, dtype=int)
    logits = scores[inp]
    probs = _softmax_rows(logits)
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(inp.shape[0]), tgt], 1e-12, 1.0))))
    grad = probs.copy()
    grad[np.arange(inp.shape[0]), tgt] -= 1.0
    grad /= float(inp.shape[0])
    dW = np.zeros_like(scores)
    np.add.at(dW, inp, grad)
    scores -= float(learning_rate) * dW
    return nll


def make_checkpoint(
    *,
    lineage: SplitLineage,
    seed: int,
    steps: int,
    context_length: int,
    vocab_size: int,
    defect: str,
    alignment: str,
    train_loss: float,
    eval_loss: float,
) -> StageAwareCheckpoint:
    checkpoint_id = (
        f"{TRAINING_VERSION}:{lineage.split_hash}:{int(seed)}:{int(steps)}:{_normalize_defect(defect)}"
    )
    return StageAwareCheckpoint(
        checkpoint_id=checkpoint_id,
        dataset_version=lineage.dataset_version,
        split_hash=lineage.split_hash,
        train_ids=lineage.train_ids,
        eval_ids=lineage.eval_ids,
        leaked_ids=lineage.leaked_ids,
        adaptation_stage="pretrained",
        objective="causal_next_token",
        seed=int(seed),
        steps=int(steps),
        context_length=int(context_length),
        tokenizer_name=lineage.tokenizer_name,
        tokenizer_version=lineage.tokenizer_version,
        vocab_size=int(vocab_size),
        training_time=False,
        inference_ready=True,
        scale_limit=SCALE_LIMIT,
        version=TRAINING_VERSION,
        audit={
            "defect": _normalize_defect(defect),
            "alignment": alignment,
            "contaminated": lineage.contaminated,
            "final_train_objective_loss": float(train_loss),
            "final_eval_true_loss": float(eval_loss),
            "training_time_boundary": TRAINING_TIME_BOUNDARY,
        },
    )


def run_causal_pipeline(
    *,
    defect: str | None = "none",
    seed: int = SEED,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    context_length: int = TEACHING_CONTEXT_LENGTH,
    documents: tuple[Document, ...] | None = None,
) -> TrainingTrace:
    """Tokenize, window, shift, batch, train, evaluate, checkpoint.

    Evaluation always uses correct next-token alignment on authored eval
    ids. Named defects change either the training shift or train membership.
    """

    defect_key = _normalize_defect(defect)
    alignment = alignment_for_defect(defect_key)
    tok = load_teaching_tokenizer()
    docs = documents if documents is not None else load_teaching_corpus()
    authored = authored_lineage(
        docs,
        tokenizer_name=tok.name,
        tokenizer_version=tok.version,
    )
    used = used_lineage_for_defect(authored, defect_key)
    train_docs = documents_for_ids(docs, used.train_ids)
    eval_docs = documents_for_ids(docs, authored.eval_ids)
    train_windows = pair_windows(
        train_docs,
        tokenizer=tok,
        context_length=context_length,
        alignment=alignment,
    )
    eval_true_windows = pair_windows(
        eval_docs,
        tokenizer=tok,
        context_length=context_length,
        alignment="correct",
    )
    authored_train_windows = pair_windows(
        documents_for_ids(docs, authored.train_ids),
        tokenizer=tok,
        context_length=context_length,
        alignment="correct",
    )
    train_inputs, train_targets = _flatten_windows(train_windows)
    eval_inputs, eval_targets = _flatten_windows(eval_true_windows)
    scores = _init_scores(tok.identity.vocab_size, seed)
    train_curve: list[float] = []
    eval_curve: list[float] = []
    for _ in range(int(steps)):
        train_curve.append(_sgd_epoch(scores, train_inputs, train_targets, learning_rate))
        eval_curve.append(batch_nll(scores, eval_inputs, eval_targets))
    unseen = unseen_pair_nll(scores, eval_true_windows, authored_train_windows)
    checkpoint = make_checkpoint(
        lineage=used,
        seed=seed,
        steps=steps,
        context_length=context_length,
        vocab_size=tok.identity.vocab_size,
        defect=defect_key,
        alignment=alignment,
        train_loss=train_curve[-1],
        eval_loss=eval_curve[-1],
    )
    return TrainingTrace(
        defect=defect_key,
        alignment=alignment,
        documents=docs,
        authored_lineage=authored,
        used_lineage=used,
        windows=train_windows,
        seed=int(seed),
        steps=int(steps),
        learning_rate=float(learning_rate),
        context_length=int(context_length),
        vocab_size=int(tok.identity.vocab_size),
        pad_id=int(tok.pad_id),
        scores=scores,
        train_objective_losses=tuple(train_curve),
        eval_true_losses=tuple(eval_curve),
        final_train_objective_loss=float(train_curve[-1]),
        final_eval_true_loss=float(eval_curve[-1]),
        unseen_pair_nll=float(unseen),
        checkpoint=checkpoint,
        tokenizer_name=tok.name,
        tokenizer_version=tok.version,
    )


def pipeline_with_defect(**kwargs) -> TrainingTrace:
    """Controlled-failure entry: one named defect, same corpus and budget."""

    if "defect" not in kwargs or _normalize_defect(kwargs.get("defect")) == "none":
        raise ValueError("pipeline_with_defect requires a named defect")
    return run_causal_pipeline(**kwargs)


def repair_run(trace: TrainingTrace) -> TrainingTrace:
    """Recompute from a defective trace's documents, seed, steps, and window."""

    return run_causal_pipeline(
        defect="none",
        seed=trace.seed,
        steps=trace.steps,
        learning_rate=trace.learning_rate,
        context_length=trace.context_length,
        documents=trace.documents,
    )


def first_divergence(left: TrainingTrace, right: TrainingTrace) -> str | None:
    for name in TRACE_CHECKPOINTS:
        if left.checkpoint_field(name) != right.checkpoint_field(name):
            return name
    return None


def classify_intervention(intervention_id: str) -> str:
    key = str(intervention_id)
    if key not in INTERVENTIONS:
        raise ValueError(
            f"unknown intervention {intervention_id!r}; use one of {tuple(INTERVENTIONS)}"
        )
    return INTERVENTIONS[key]


def stage_catalog() -> dict[str, str]:
    return dict(STAGE_DEFINITIONS)


def lineage_report(lineage: SplitLineage) -> dict[str, Any]:
    overlap = tuple(sorted(set(lineage.train_ids) & set(lineage.eval_ids)))
    return {
        "dataset_version": lineage.dataset_version,
        "train_ids": lineage.train_ids,
        "eval_ids": lineage.eval_ids,
        "leaked_ids": lineage.leaked_ids,
        "overlap": overlap,
        "contaminated": lineage.contaminated,
        "split_hash": lineage.split_hash,
        "eval_valid": not lineage.contaminated,
    }


def observability_report(trace: TrainingTrace) -> dict[str, Any]:
    return {
        "version": TRAINING_VERSION,
        "dataset_version": DATASET_VERSION,
        "defect": trace.defect,
        "alignment": trace.alignment,
        "causal_windows": all(window.causal for window in trace.windows),
        "contaminated": trace.used_lineage.contaminated,
        "authored_train_ids": trace.authored_lineage.train_ids,
        "used_train_ids": trace.used_lineage.train_ids,
        "eval_ids": trace.authored_lineage.eval_ids,
        "n_targets": tuple(window.n_targets for window in trace.windows),
        "final_train_objective_loss": trace.final_train_objective_loss,
        "final_eval_true_loss": trace.final_eval_true_loss,
        "unseen_pair_nll": trace.unseen_pair_nll,
        "checkpoint_id": trace.checkpoint.checkpoint_id,
        "adaptation_stage": trace.checkpoint.adaptation_stage,
        "inference_ready": trace.checkpoint.inference_ready,
        "training_time": trace.checkpoint.training_time,
        "scale_limit": SCALE_LIMIT,
        "training_time_boundary": TRAINING_TIME_BOUNDARY,
        "handoff": (
            "M32 receives a stage-aware checkpoint and this training-time versus "
            "inference-time boundary; it does not receive a decoder from M31."
        ),
    }


def true_next_token_nll(trace: TrainingTrace) -> float:
    """Score authored train windows with correct shift on the learned table."""

    tok = load_teaching_tokenizer()
    train_docs = documents_for_ids(trace.documents, trace.authored_lineage.train_ids)
    windows = pair_windows(
        train_docs,
        tokenizer=tok,
        context_length=trace.context_length,
        alignment="correct",
    )
    inputs, targets = _flatten_windows(windows)
    return batch_nll(trace.scores, inputs, targets)
