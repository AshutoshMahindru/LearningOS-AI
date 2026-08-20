"""Transparent teaching embeddings for Mission M28.

M28 retrieves with bundled sentence vectors, cosine ranking, a lexical
baseline, and an explicit provenance contract. Attention, Q/K/V, and a
production search service remain deferred to M29 and M33.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.ASCII)
COMPATIBILITY_FIELDS = (
    "family",
    "model",
    "version",
    "dimensions",
    "metric",
    "normalization",
    "pooling",
)
ZERO_TOL = 1e-12
RESIDUAL_SCALE = 0.08
DIGIT_TICKET_LOW = 1000
DIGIT_TICKET_HIGH = 9999

DIMENSION_NAMES = (
    "account_access",
    "credentials",
    "device_print",
    "refund",
    "negation",
    "payment",
    "magnitude",
    "ticket_ops",
    "weather",
    "legal",
    "glue",
    "residual",
)
ACCOUNT, CREDENTIALS, DEVICE_PRINT, REFUND, NEGATION, PAYMENT, MAGNITUDE, TICKET_OPS, WEATHER, LEGAL, GLUE, RESIDUAL = range(12)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "because",
        "for",
        "i",
        "in",
        "is",
        "my",
        "now",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "do",
    }
)


class ProvenanceError(ValueError):
    """Raised when query and corpus embedding contracts cannot be mixed."""

    def __init__(
        self,
        *,
        mismatches: tuple[str, ...],
        query: "Provenance",
        corpus: "Provenance",
    ):
        self.mismatches = mismatches
        self.query = query
        self.corpus = corpus
        detail = ", ".join(mismatches) if mismatches else "unknown"
        super().__init__(f"incompatible embedding provenance: {detail}")


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M28 requires numpy; install requirements/m28.txt") from exc
    return np


def default_dataset_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M28"
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled embedding fixtures: {path}")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_vector(values: Sequence[float], *, name: str = "vector", dimensions: int | None = None) -> tuple[float, ...]:
    """Return a finite 1-D float vector, optionally checking width."""

    np = _require_numpy()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of numbers")
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    if dimensions is not None and int(array.size) != int(dimensions):
        raise ValueError(f"{name} must have {dimensions} dimensions; got {int(array.size)}")
    return tuple(float(value) for value in array)


def l2_norm(vector: Sequence[float]) -> float:
    np = _require_numpy()
    array = np.asarray(as_vector(vector), dtype=float)
    return float(np.linalg.norm(array))


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    np = _require_numpy()
    array = np.asarray(as_vector(vector), dtype=float)
    magnitude = float(np.linalg.norm(array))
    if magnitude < ZERO_TOL:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(value) for value in (array / magnitude))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Directional agreement in [-1, 1] for nonzero equal-width vectors."""

    np = _require_numpy()
    a = np.asarray(as_vector(left, name="left vector"), dtype=float)
    b = np.asarray(as_vector(right, name="right vector"), dtype=float)
    if a.size != b.size:
        raise ValueError("vectors must have the same dimensions")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < ZERO_TOL:
        raise ValueError("cosine similarity is undefined for a zero vector")
    score = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, score))


def inner_product(left: Sequence[float], right: Sequence[float]) -> float:
    np = _require_numpy()
    a = np.asarray(as_vector(left, name="left vector"), dtype=float)
    b = np.asarray(as_vector(right, name="right vector"), dtype=float)
    if a.size != b.size:
        raise ValueError("vectors must have the same dimensions")
    return float(np.dot(a, b))


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Whitespace-insensitive word tokens for the lexical overlap baseline."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(TOKEN_RE.findall(text.lower()))


def content_tokens(text: str) -> tuple[str, ...]:
    """Lexical tokens with teaching stopwords removed."""

    return tuple(token for token in lexical_tokens(text) if token not in STOPWORDS)


def lexical_overlap(left: str, right: str) -> float:
    """Jaccard overlap on alphanumeric word tokens, including stopwords."""

    a = set(lexical_tokens(left))
    b = set(lexical_tokens(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_residual(token: str, *, scale: float = RESIDUAL_SCALE) -> float:
    digest = hashlib.sha256(f"m28-residual:{token}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:2], "big") / 65535.0
    return (raw * 2.0 - 1.0) * float(scale)


def numeric_semantic_prior(token: str) -> tuple[float, ...] | None:
    """Shared prior for digit strings so nearby ticket ids stay close."""

    if not token.isdigit():
        return None
    vec = [0.0] * 11
    number = int(token)
    if DIGIT_TICKET_LOW <= number <= DIGIT_TICKET_HIGH:
        vec[TICKET_OPS] = 0.75
        vec[MAGNITUDE] = min(number / 10000.0, 1.0) * 0.15
    else:
        vec[PAYMENT] = 0.35
        vec[MAGNITUDE] = min(number / 100000.0, 1.0)
        vec[TICKET_OPS] = 0.15
    return tuple(vec)


@dataclass(frozen=True)
class Provenance:
    family: str
    model: str
    version: str
    dimensions: int
    metric: str
    normalization: str
    pooling: str
    downloaded: bool
    network_required: bool
    dimension_names: tuple[str, ...] = ()
    not_sentence_transformers: bool = True
    not_model_hub: bool = True

    def fingerprint(self) -> dict[str, object]:
        payload = {field: getattr(self, field) for field in COMPATIBILITY_FIELDS}
        payload["downloaded"] = self.downloaded
        payload["network_required"] = self.network_required
        return payload


def provenance_from_mapping(payload: Mapping[str, object]) -> Provenance:
    names = payload.get("dimension_names") or ()
    return Provenance(
        family=str(payload["family"]),
        model=str(payload["model"]),
        version=str(payload["version"]),
        dimensions=int(payload["dimensions"]),
        metric=str(payload["metric"]),
        normalization=str(payload["normalization"]),
        pooling=str(payload["pooling"]),
        downloaded=bool(payload.get("downloaded", False)),
        network_required=bool(payload.get("network_required", False)),
        dimension_names=tuple(str(name) for name in names),
        not_sentence_transformers=bool(payload.get("not_sentence_transformers", True)),
        not_model_hub=bool(payload.get("not_model_hub", True)),
    )


def compatibility_report(query: Provenance, corpus: Provenance) -> dict[str, object]:
    mismatches = tuple(
        field for field in COMPATIBILITY_FIELDS if getattr(query, field) != getattr(corpus, field)
    )
    return {
        "compatible": not mismatches,
        "mismatches": mismatches,
        "query": query.fingerprint(),
        "corpus": corpus.fingerprint(),
    }


def assert_compatible(query: Provenance, corpus: Provenance) -> None:
    report = compatibility_report(query, corpus)
    if report["compatible"]:
        return
    raise ProvenanceError(
        mismatches=tuple(report["mismatches"]),
        query=query,
        corpus=corpus,
    )


def operational_score(
    left: Sequence[float],
    right: Sequence[float],
    *,
    metric: str,
    normalization: str,
) -> float:
    """Score a pair using the declared metric/normalization contract.

    Cosine is always the cosine formula (scale-invariant); ``normalization``
    does not change that path. Inner product (``dot`` / ``ip``) uses raw
    vectors unless ``normalization=="l2"``, in which case both sides are
    L2-normalized first so the inner product equals cosine.
    """

    key = str(metric).lower()
    if key == "cosine":
        return cosine_similarity(left, right)
    if key in {"dot", "ip", "inner"}:
        if str(normalization).lower() == "l2":
            return inner_product(l2_normalize(left), l2_normalize(right))
        return inner_product(left, right)
    raise ValueError("metric must be 'cosine' or 'dot'")


@dataclass(frozen=True)
class EmbeddedItem:
    id: str
    text: str
    vector: tuple[float, ...]
    role: str = "document"
    tags: tuple[str, ...] = ()

    @property
    def dimensions(self) -> int:
        return len(self.vector)

    @property
    def norm(self) -> float:
        return l2_norm(self.vector)


@dataclass(frozen=True)
class RankedItem:
    id: str
    text: str
    score: float
    rank: int
    lexical_overlap: float
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Retrieval:
    query_id: str
    query_text: str
    query_vector: tuple[float, ...]
    metric: str
    normalization: str
    provenance: Provenance
    results: tuple[RankedItem, ...]
    enforced: bool
    score_fn: str

    @property
    def top_id(self) -> str | None:
        return self.results[0].id if self.results else None

    def margin(self, leading: int = 0, trailing: int = 1) -> float | None:
        if trailing >= len(self.results) or leading >= len(self.results):
            return None
        return float(self.results[leading].score - self.results[trailing].score)

    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.results)


@dataclass(frozen=True)
class VectorSpace:
    provenance: Provenance
    items: tuple[EmbeddedItem, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        width = self.provenance.dimensions
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"duplicate item id: {item.id}")
            if len(item.vector) != width:
                raise ValueError(f"{item.id} has width {len(item.vector)}; expected {width}")
            seen.add(item.id)

    def get(self, item_id: str) -> EmbeddedItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def documents(self) -> tuple[EmbeddedItem, ...]:
        return tuple(item for item in self.items if item.role == "document")

    def queries(self) -> tuple[EmbeddedItem, ...]:
        return tuple(item for item in self.items if item.role == "query")

    def matrix(self, *, role: str | None = "document"):
        np = _require_numpy()
        chosen = [item for item in self.items if role is None or item.role == role]
        if not chosen:
            raise ValueError("vector space has no items for the requested role")
        return np.asarray([item.vector for item in chosen], dtype=float)


def _score_pair(
    query_vector: Sequence[float],
    candidate: Sequence[float],
    *,
    provenance: Provenance,
    score_fn: str,
) -> float:
    key = str(score_fn).lower()
    if key == "declared":
        return operational_score(
            query_vector,
            candidate,
            metric=provenance.metric,
            normalization=provenance.normalization,
        )
    if key == "cosine":
        return cosine_similarity(query_vector, candidate)
    if key in {"dot", "ip", "inner"}:
        return inner_product(query_vector, candidate)
    raise ValueError("score_fn must be 'declared', 'cosine', or 'dot'")


def rank_neighbors(
    query_vector: Sequence[float],
    corpus: VectorSpace | Sequence[EmbeddedItem],
    *,
    query_text: str = "",
    query_id: str = "query",
    query_provenance: Provenance | None = None,
    top_k: int | None = None,
    enforce_provenance: bool = True,
    score_fn: str = "declared",
) -> Retrieval:
    """Rank corpus items. Refuses mixed provenance unless explicitly unchecked."""

    if isinstance(corpus, VectorSpace):
        space = corpus
        items = space.documents() or space.items
        corpus_provenance = space.provenance
    else:
        items = tuple(corpus)
        if query_provenance is None:
            raise ValueError("query_provenance is required when corpus is a raw item list")
        corpus_provenance = query_provenance
        space_prov = query_provenance
        space = VectorSpace(provenance=space_prov, items=items)

    used_query_prov = query_provenance or space.provenance
    if enforce_provenance:
        assert_compatible(used_query_prov, corpus_provenance)
    query = as_vector(query_vector, name="query", dimensions=used_query_prov.dimensions)
    if top_k is not None and int(top_k) < 1:
        raise ValueError("top_k must be positive")

    scored: list[RankedItem] = []
    for item in items:
        score = _score_pair(
            query,
            item.vector,
            provenance=corpus_provenance if score_fn == "declared" else used_query_prov,
            score_fn=score_fn,
        )
        scored.append(
            RankedItem(
                id=item.id,
                text=item.text,
                score=float(score),
                rank=0,
                lexical_overlap=lexical_overlap(query_text, item.text) if query_text else 0.0,
                vector=item.vector,
            )
        )
    scored.sort(key=lambda item: (-item.score, item.id))
    if top_k is not None:
        scored = scored[: int(top_k)]
    ranked = tuple(
        RankedItem(
            id=item.id,
            text=item.text,
            score=item.score,
            rank=index + 1,
            lexical_overlap=item.lexical_overlap,
            vector=item.vector,
        )
        for index, item in enumerate(scored)
    )
    return Retrieval(
        query_id=query_id,
        query_text=query_text,
        query_vector=query,
        metric=corpus_provenance.metric,
        normalization=corpus_provenance.normalization,
        provenance=corpus_provenance,
        results=ranked,
        enforced=bool(enforce_provenance),
        score_fn=str(score_fn),
    )


def retrieve_unchecked(
    query_vector: Sequence[float],
    corpus: VectorSpace,
    *,
    query_text: str = "",
    query_id: str = "query",
    query_provenance: Provenance | None = None,
    top_k: int | None = None,
    score_fn: str = "declared",
) -> Retrieval:
    """Controlled defect: score without the provenance gate.

    The named change is ``enforce_provenance=False``. Scoring stays on the
    declared metric so a wrong neighbor is attributable to the store, not
    to a second scoring-knob change.
    """

    return rank_neighbors(
        query_vector,
        corpus,
        query_text=query_text,
        query_id=query_id,
        query_provenance=query_provenance or corpus.provenance,
        top_k=top_k,
        enforce_provenance=False,
        score_fn=score_fn,
    )


def pairwise_cosine(items: Sequence[EmbeddedItem] | VectorSpace, *, ids: Sequence[str] | None = None):
    """Return an (n, n) cosine matrix and the ordered ids."""

    np = _require_numpy()
    if isinstance(items, VectorSpace):
        chosen = list(items.items)
    else:
        chosen = list(items)
    if ids is not None:
        by_id = {item.id: item for item in chosen}
        chosen = [by_id[item_id] for item_id in ids]
    matrix = np.asarray([item.vector for item in chosen], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < ZERO_TOL):
        raise ValueError("cosine similarity is undefined for a zero vector")
    unit = matrix / norms
    scores = unit @ unit.T
    scores = np.clip(scores, -1.0, 1.0)
    return scores, tuple(item.id for item in chosen)


def compare_lexical_and_semantic(
    query_text: str,
    corpus: VectorSpace,
    *,
    query_vector: Sequence[float],
    query_id: str = "query",
    query_provenance: Provenance | None = None,
) -> tuple[dict[str, object], ...]:
    semantic = rank_neighbors(
        query_vector,
        corpus,
        query_text=query_text,
        query_id=query_id,
        query_provenance=query_provenance,
        enforce_provenance=True,
        score_fn="declared",
    )
    lexical_rows = sorted(
        (
            {
                "id": item.id,
                "lexical_overlap": lexical_overlap(query_text, item.text),
                "text": item.text,
            }
            for item in corpus.documents()
        ),
        key=lambda row: (-float(row["lexical_overlap"]), str(row["id"])),
    )
    lexical_rank = {row["id"]: index + 1 for index, row in enumerate(lexical_rows)}
    out: list[dict[str, object]] = []
    for item in semantic.results:
        out.append(
            {
                "id": item.id,
                "text": item.text,
                "cosine": item.score,
                "semantic_rank": item.rank,
                "lexical_overlap": item.lexical_overlap,
                "lexical_rank": lexical_rank[item.id],
                "disagrees": item.rank != lexical_rank[item.id],
            }
        )
    return tuple(out)


def swap_account_print(vector: Sequence[float]) -> tuple[float, ...]:
    """Axis swap used by the v06.2 mismatch store (account ↔ print)."""

    values = list(as_vector(vector, name="vector"))
    if len(values) < 3:
        raise ValueError("account/print swap requires at least 3 dimensions")
    values[ACCOUNT], values[DEVICE_PRINT] = values[DEVICE_PRINT], values[ACCOUNT]
    return tuple(values)


class TeachingEncoder:
    """Mean-pool a frozen token table. Not a downloaded encoder."""

    def __init__(self, spec: Mapping[str, object]):
        provenance = provenance_from_mapping(spec["provenance"])
        if provenance.downloaded or provenance.network_required:
            raise RuntimeError("M28 required path must use the bundled offline encoder")
        if provenance.dimensions != len(DIMENSION_NAMES):
            raise ValueError("teaching encoder ships 12 named dimensions")
        table_raw = spec["token_semantics"]
        table: dict[str, tuple[float, ...]] = {}
        for token, values in table_raw.items():
            semantic = as_vector(values, name=f"token {token}", dimensions=11)
            table[str(token)] = semantic
        self.provenance = provenance
        self.table = table
        self.stopwords = frozenset(str(word) for word in spec.get("stopwords", STOPWORDS))
        self.residual_scale = float(spec.get("residual_scale", RESIDUAL_SCALE))
        self.dimension_names = DIMENSION_NAMES

    def lookup_semantic(self, token: str) -> tuple[float, ...]:
        if token in self.table:
            return self.table[token]
        numeric = numeric_semantic_prior(token)
        if numeric is not None:
            return numeric
        return tuple(0.0 for _ in range(11))

    def lookup(self, token: str) -> tuple[float, ...]:
        semantic = self.lookup_semantic(token)
        return semantic + (token_residual(token, scale=self.residual_scale),)

    def encode(
        self,
        text: str,
        *,
        pooling: str | None = None,
        normalization: str | None = None,
    ) -> tuple[float, ...]:
        tokens = tuple(token for token in lexical_tokens(text) if token not in self.stopwords)
        if not tokens:
            raise ValueError("cannot encode text with no content tokens")
        np = _require_numpy()
        matrix = np.asarray([self.lookup(token) for token in tokens], dtype=float)
        pool = str(pooling or self.provenance.pooling).lower()
        if pool == "mean":
            raw = matrix.mean(axis=0)
        elif pool == "sum":
            raw = matrix.sum(axis=0)
        else:
            raise ValueError("pooling must be 'mean' or 'sum'")
        norm_policy = str(normalization or self.provenance.normalization).lower()
        if norm_policy == "l2":
            return l2_normalize(raw)
        if norm_policy == "none":
            return tuple(float(value) for value in raw)
        raise ValueError("normalization must be 'l2' or 'none'")

    def encode_report(self, text: str) -> dict[str, object]:
        kept = tuple(token for token in lexical_tokens(text) if token not in self.stopwords)
        vector = self.encode(text)
        hits = tuple(token for token in kept if token in self.table)
        numeric = tuple(token for token in kept if token not in self.table and token.isdigit())
        residual_only = tuple(token for token in kept if token not in self.table and not token.isdigit())
        return {
            "text": text,
            "content_tokens": kept,
            "table_hits": hits,
            "numeric_tokens": numeric,
            "residual_only": residual_only,
            "shape": (len(vector),),
            "norm": l2_norm(vector),
            "pooling": self.provenance.pooling,
            "normalization": self.provenance.normalization,
            "model": self.provenance.model,
            "version": self.provenance.version,
            "downloaded": self.provenance.downloaded,
        }


def _join_space(
    catalog: Mapping[str, object],
    embeddings: Mapping[str, object],
    *,
    role_filter: str | None = None,
) -> VectorSpace:
    provenance = provenance_from_mapping(embeddings["provenance"])
    texts: dict[str, dict[str, object]] = {}
    for row in catalog.get("corpus", []):
        texts[str(row["id"])] = {**row, "role": "document"}
    for row in catalog.get("queries", []):
        texts[str(row["id"])] = {**row, "role": "query"}
    items: list[EmbeddedItem] = []
    for row in embeddings["items"]:
        item_id = str(row["id"])
        meta = texts.get(item_id, {"id": item_id, "text": row.get("text", ""), "role": row.get("role", "document"), "tags": []})
        role = str(row.get("role") or meta.get("role") or "document")
        if role_filter is not None and role != role_filter:
            continue
        items.append(
            EmbeddedItem(
                id=item_id,
                text=str(meta.get("text") or row.get("text") or ""),
                vector=as_vector(row["vector"], name=item_id, dimensions=provenance.dimensions),
                role=role,
                tags=tuple(str(tag) for tag in meta.get("tags", ())),
            )
        )
    return VectorSpace(provenance=provenance, items=tuple(items))


@lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "catalog.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_token_spec(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "token_table.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_embeddings_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "embeddings.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_mismatch_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "mismatch.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_transfer_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "transfer.json"
    return _read_json(target)


@lru_cache(maxsize=2)
def load_encoder(path: str | None = None) -> TeachingEncoder:
    return TeachingEncoder(load_token_spec(path))


def load_canonical_space() -> VectorSpace:
    return _join_space(load_catalog(), load_embeddings_payload())


def load_mismatch_space() -> VectorSpace:
    catalog = load_catalog()
    payload = load_mismatch_payload()
    return _join_space(catalog, payload)


def load_transfer_space() -> VectorSpace:
    payload = load_transfer_payload()
    provenance = provenance_from_mapping(payload["provenance"])
    query = payload["query"]
    items = [
        EmbeddedItem(
            id=str(query["id"]),
            text=str(query["text"]),
            vector=as_vector(query["vector"], name="transfer query", dimensions=provenance.dimensions),
            role="query",
            tags=("transfer",),
        )
    ]
    for row in payload["corpus"]:
        items.append(
            EmbeddedItem(
                id=str(row["id"]),
                text=str(row["text"]),
                vector=as_vector(row["vector"], name=str(row["id"]), dimensions=provenance.dimensions),
                role="document",
                tags=("transfer",),
            )
        )
    return VectorSpace(provenance=provenance, items=tuple(items))


def retrieve_query(
    query_id: str,
    *,
    space: VectorSpace | None = None,
    top_k: int | None = None,
    enforce_provenance: bool = True,
    score_fn: str = "declared",
) -> Retrieval:
    catalog_space = space or load_canonical_space()
    query = catalog_space.get(query_id)
    return rank_neighbors(
        query.vector,
        catalog_space,
        query_text=query.text,
        query_id=query.id,
        query_provenance=catalog_space.provenance,
        top_k=top_k,
        enforce_provenance=enforce_provenance,
        score_fn=score_fn,
    )


def retrieval_report(result: Retrieval) -> dict[str, object]:
    return {
        "query_id": result.query_id,
        "query_text": result.query_text,
        "metric": result.metric,
        "normalization": result.normalization,
        "model": result.provenance.model,
        "version": result.provenance.version,
        "enforced": result.enforced,
        "score_fn": result.score_fn,
        "top_id": result.top_id,
        "margin_1_2": result.margin(),
        "ranking": tuple(
            {
                "rank": item.rank,
                "id": item.id,
                "score": item.score,
                "lexical_overlap": item.lexical_overlap,
            }
            for item in result.results
        ),
    }


def encode_text(text: str, *, pooling: str | None = None, normalization: str | None = None) -> tuple[float, ...]:
    return load_encoder().encode(text, pooling=pooling, normalization=normalization)
