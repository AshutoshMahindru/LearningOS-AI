"""Deterministic retrieval evaluation for Mission M35.

M35 is the V09 ranking/chunking boundary: freeze M34 labels, score
candidate lists, inspect per-query failures, then change one of
chunking, candidate k, or a local lexical reranker. Vector-database
internals, required ANN/Qdrant/HNSW, and hybrid fusion remain M36.
Decoding labs remain M32. Generation policy remains M34.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from missions.M33.semantic_search import (  # noqa: E402
    Chunk,
    Corpus,
    Document,
    ExactIndex,
    RankedHit,
    SearchResponse,
    build_index,
    compose_document,
    encode_query,
    load_canonical_corpus,
    load_canonical_index,
    search,
    source_hash,
    replace_chunk_text,
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.ASCII)
PROCEDURE_RE = re.compile(
    r"\b(how do i|what should|how can|what does|why does|which policy|which sentence)\b",
    re.IGNORECASE,
)
ACTION_HINT_RE = re.compile(r"^(please |do not |approve )", re.IGNORECASE)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "because",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "my",
        "now",
        "of",
        "on",
        "or",
        "please",
        "should",
        "that",
        "the",
        "their",
        "these",
        "they",
        "this",
        "those",
        "to",
        "user",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "with",
    }
)

PIPELINE_ID = "v09-retrieval-eval"
EVAL_VERSION = "m34.eval.v1"
CANONICAL_CORPUS_VERSION = "m33.corpus.v1"
CANONICAL_INDEX_ID = "v08-exact-memory"
SEED = 3501
DEFAULT_CANDIDATE_K = 3
RERANKER_IDENTITY = "identity"
RERANKER_LEX = "lex-overlap-v1"
RERANKER_OPTIONAL_MODEL = "optional-model"
SUPPORTED_RERANKERS = (RERANKER_IDENTITY, RERANKER_LEX)
GRADE_SUPPORT = 2.0
GRADE_RELEVANT = 1.0
CRITICAL_QUERY_IDS = frozenset(
    {
        "rag-ticket-4412",
        "rag-password-procedure",
        "rag-h-invoice",
        "rag-fifty",
        "rag-refund-deny",
    }
)
FAILURE_CANDIDATE_MISS = "candidate_miss"
FAILURE_TRAP_AT_1 = "trap_at_1"
FAILURE_RANKING_MISS = "ranking_miss"
FAILURE_SUBOPTIMAL = "suboptimal_ranking"
FAILURE_UNANSWERABLE = "unanswerable"
FAILURE_OK = "ok"

SYSTEM_MAP = (
    "frozen M34 labels --load/version--> FrozenQuery (eval_version, original chunk ids)\n"
    "source documents --chunk(size, overlap)--> versioned corpus (source_hash)\n"
    "versioned corpus --M33 encode/search/as_evidence--> candidate set (k, cosine ranks)\n"
    "candidate set --rerank(identity|lex-overlap-v1)--> ranked ids (same members)\n"
    "ranked ids x projected labels --metrics--> recall@k, MRR, nDCG@k, per-query report\n"
    "slices --compare--> aggregate vs critical; holdout stays unlabeled after scores"
)

DEFERRED_BOUNDARY = (
    "M35 measures ranking and chunking on an exact in-memory index. "
    "It does not deploy Qdrant, HNSW, ANN, or hybrid/sparse fusion (M36). "
    "It does not open temperature/top-p decoding (M32). "
    "It does not retune M34 extractive generation."
)


def default_dataset_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M35"
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled retrieval-eval fixtures: {path}")
    return path


def m34_questions_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M34" / "questions.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen M34 questions: {path}")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(str(text).lower()))


def stem_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ment") and len(token) > 6:
        return token[:-4]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in tokenize(text) if token not in STOPWORDS)


def content_stems(text: str) -> frozenset[str]:
    return frozenset(stem_token(token) for token in content_tokens(text))


def query_digits(text: str) -> frozenset[str]:
    return frozenset(token for token in tokenize(text) if token.isdigit())


def has_content_tokens(text: str) -> bool:
    return bool(tokenize(text))


@dataclass(frozen=True)
class FrozenQuery:
    query_id: str
    text: str
    experiment: str
    split: str
    answerable: bool
    gold_answer: str | None
    gold_aliases: tuple[str, ...]
    support_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...]
    traps: tuple[str, ...]
    notes: str = ""
    eval_version: str = EVAL_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "experiment": self.experiment,
            "split": self.split,
            "answerable": self.answerable,
            "gold_answer": self.gold_answer,
            "gold_aliases": self.gold_aliases,
            "support_chunk_ids": self.support_chunk_ids,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "traps": self.traps,
            "notes": self.notes,
            "eval_version": self.eval_version,
        }


@dataclass(frozen=True)
class ProjectedLabels:
    query_id: str
    corpus_version: str
    relevant_ids: tuple[str, ...]
    support_ids: tuple[str, ...]
    trap_ids: tuple[str, ...]
    mixed_ids: tuple[str, ...]
    source_relevant_ids: tuple[str, ...]
    source_support_ids: tuple[str, ...]

    def grades(self) -> dict[str, float]:
        mapping = {chunk_id: GRADE_RELEVANT for chunk_id in self.relevant_ids}
        for chunk_id in self.support_ids:
            mapping[chunk_id] = GRADE_SUPPORT
        return mapping


@dataclass(frozen=True)
class ChunkSpec:
    corpus_version: str
    mode: str
    size: int | None
    overlap: int
    description: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_version": self.corpus_version,
            "mode": self.mode,
            "size": self.size,
            "overlap": self.overlap,
            "description": self.description,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    corpus_version: str
    chunk_mode: str
    chunk_size: int | None
    chunk_overlap: int
    candidate_k: int
    reranker_id: str
    eval_version: str = EVAL_VERSION
    seed: int = SEED
    leaked: bool = False
    hard_negatives: bool = False
    relabeled: bool = False

    def __post_init__(self) -> None:
        if int(self.candidate_k) < 1:
            raise ValueError("candidate_k must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_size is not None and int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be positive when set")
        if (
            self.chunk_size is not None
            and self.chunk_overlap >= int(self.chunk_size)
        ):
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.reranker_id not in SUPPORTED_RERANKERS:
            raise ValueError(f"unsupported reranker: {self.reranker_id}")

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "corpus_version": self.corpus_version,
            "chunk_mode": self.chunk_mode,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "candidate_k": int(self.candidate_k),
            "reranker_id": self.reranker_id,
            "eval_version": self.eval_version,
            "seed": self.seed,
            "leaked": self.leaked,
            "hard_negatives": self.hard_negatives,
            "relabeled": self.relabeled,
            "pipeline_id": PIPELINE_ID,
        }

    def identity(self) -> str:
        return hashlib.sha256(_dump_canonical(self.as_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    original_rank: int
    score: float
    cosine_score: float
    rerank_score: float | None
    chunk_id: str
    document_id: str
    text: str
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "original_rank": self.original_rank,
            "score": self.score,
            "cosine_score": self.cosine_score,
            "rerank_score": self.rerank_score,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CandidateSet:
    query_id: str
    query_text: str
    candidate_k: int
    reranker_id: str
    items: tuple[RankedCandidate, ...]
    scored_candidates: int
    latency_ms: float
    index_id: str
    source_hash: str
    corpus_version: str
    model: str
    version: str
    metric: str
    normalization: str

    def ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.items)

    def id_set(self) -> frozenset[str]:
        return frozenset(self.ids())

    def scores(self) -> tuple[float, ...]:
        return tuple(item.score for item in self.items)

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "candidate_k": self.candidate_k,
            "reranker_id": self.reranker_id,
            "items": tuple(item.as_dict() for item in self.items),
            "ids": self.ids(),
            "scored_candidates": self.scored_candidates,
            "latency_ms": round(self.latency_ms, 3),
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "corpus_version": self.corpus_version,
            "model": self.model,
            "version": self.version,
            "metric": self.metric,
            "normalization": self.normalization,
        }


@dataclass(frozen=True)
class QueryReport:
    query_id: str
    query_text: str
    split: str
    answerable: bool
    experiment_id: str
    corpus_version: str
    index_id: str
    source_hash: str
    candidate_k: int
    reranker_id: str
    config_identity: str
    candidate_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    cosine_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    support_ids: tuple[str, ...]
    trap_ids: tuple[str, ...]
    mixed_ids: tuple[str, ...]
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    first_relevant_rank: int | None
    first_support_rank: int | None
    candidate_recall: float | None
    candidate_support_hit: bool | None
    trap_at_1: bool
    scored_candidates: int
    rerank_cost: int
    proxy_cost: int
    latency_ms: float
    failure_mode: str
    eval_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "split": self.split,
            "answerable": self.answerable,
            "experiment_id": self.experiment_id,
            "corpus_version": self.corpus_version,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "candidate_k": self.candidate_k,
            "reranker_id": self.reranker_id,
            "config_identity": self.config_identity,
            "candidate_ids": self.candidate_ids,
            "ranked_ids": self.ranked_ids,
            "cosine_ids": self.cosine_ids,
            "relevant_ids": self.relevant_ids,
            "support_ids": self.support_ids,
            "trap_ids": self.trap_ids,
            "mixed_ids": self.mixed_ids,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "first_relevant_rank": self.first_relevant_rank,
            "first_support_rank": self.first_support_rank,
            "candidate_recall": self.candidate_recall,
            "candidate_support_hit": self.candidate_support_hit,
            "trap_at_1": self.trap_at_1,
            "scored_candidates": self.scored_candidates,
            "rerank_cost": self.rerank_cost,
            "proxy_cost": self.proxy_cost,
            "latency_ms": round(self.latency_ms, 3),
            "failure_mode": self.failure_mode,
            "eval_version": self.eval_version,
        }


@dataclass(frozen=True)
class SetReport:
    config: ExperimentConfig
    config_identity: str
    eval_version: str
    corpus_version: str
    index_id: str
    source_hash: str
    n: int
    n_answerable: int
    mean_recall_at_k: float | None
    mean_mrr: float | None
    mean_ndcg_at_k: float | None
    mean_candidate_recall: float | None
    trap_at_1_rate: float
    scored_candidates: int
    mean_proxy_cost: float
    slices: dict[str, dict[str, object]]
    rows: tuple[QueryReport, ...]
    label_hash: str
    questions_sha256: str
    leaked: bool
    relabeled: bool

    def row_map(self) -> dict[str, QueryReport]:
        return {row.query_id: row for row in self.rows}

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "config_identity": self.config_identity,
            "eval_version": self.eval_version,
            "corpus_version": self.corpus_version,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "n": self.n,
            "n_answerable": self.n_answerable,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_mrr": self.mean_mrr,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "mean_candidate_recall": self.mean_candidate_recall,
            "trap_at_1_rate": self.trap_at_1_rate,
            "scored_candidates": self.scored_candidates,
            "mean_proxy_cost": self.mean_proxy_cost,
            "slices": self.slices,
            "rows": tuple(row.as_dict() for row in self.rows),
            "label_hash": self.label_hash,
            "questions_sha256": self.questions_sha256,
            "leaked": self.leaked,
            "relabeled": self.relabeled,
        }


def _labeled_from_payload(row: Mapping[str, object], *, eval_version: str) -> FrozenQuery:
    gold = row.get("gold_answer")
    return FrozenQuery(
        query_id=str(row["id"]),
        text=str(row["text"]),
        experiment=str(row.get("experiment") or ""),
        split=str(row.get("split") or "dev"),
        answerable=bool(row.get("answerable", True)),
        gold_answer=None if gold in (None, "") else str(gold),
        gold_aliases=tuple(str(item) for item in row.get("aliases", ())),
        support_chunk_ids=tuple(str(item) for item in row.get("support", ())),
        relevant_chunk_ids=tuple(str(item) for item in row.get("relevant", ())),
        traps=tuple(str(item) for item in row.get("traps", ())),
        notes=str(row.get("notes") or ""),
        eval_version=eval_version,
    )


@lru_cache(maxsize=4)
def load_m34_questions_payload(path: str | None = None) -> dict:
    target = Path(path) if path else m34_questions_path()
    payload = _read_json(target)
    if str(payload.get("eval_version")) != EVAL_VERSION:
        raise ValueError(f"expected frozen eval_version {EVAL_VERSION}")
    if payload.get("downloaded") or payload.get("network_required"):
        raise ValueError("M34 questions must stay offline")
    return payload


@lru_cache(maxsize=4)
def load_chunk_versions_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "chunk_versions.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_hard_negatives_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "hard_negatives.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_expected_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "expected.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_transfer_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "transfer.json"
    return _read_json(target)


def load_frozen_queries() -> tuple[FrozenQuery, ...]:
    payload = load_m34_questions_payload()
    eval_version = str(payload["eval_version"])
    return tuple(_labeled_from_payload(row, eval_version=eval_version) for row in payload["questions"])


def load_query_map() -> dict[str, FrozenQuery]:
    return {query.query_id: query for query in load_frozen_queries()}


def questions_sha256() -> str:
    return file_sha256(m34_questions_path())


def label_hash(queries: Sequence[FrozenQuery]) -> str:
    payload = [
        {
            "query_id": query.query_id,
            "text": query.text,
            "split": query.split,
            "answerable": query.answerable,
            "support": list(query.support_chunk_ids),
            "relevant": list(query.relevant_chunk_ids),
            "traps": list(query.traps),
            "eval_version": query.eval_version,
        }
        for query in queries
    ]
    return hashlib.sha256(_dump_canonical(payload).encode("utf-8")).hexdigest()


def load_chunk_specs() -> tuple[ChunkSpec, ...]:
    payload = load_chunk_versions_payload()
    specs = []
    for row in payload["versions"]:
        size = row.get("size")
        specs.append(
            ChunkSpec(
                corpus_version=str(row["corpus_version"]),
                mode=str(row["mode"]),
                size=None if size in (None, "") else int(size),
                overlap=int(row.get("overlap") or 0),
                description=str(row.get("description") or ""),
            )
        )
    return tuple(specs)


def chunk_spec_map() -> dict[str, ChunkSpec]:
    return {spec.corpus_version: spec for spec in load_chunk_specs()}


def baseline_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="baseline",
        corpus_version=CANONICAL_CORPUS_VERSION,
        chunk_mode="canonical",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_IDENTITY,
    )


def window_spans(text: str, size: int, overlap: int) -> tuple[tuple[int, int], ...]:
    if int(size) < 1:
        raise ValueError("size must be positive")
    if int(overlap) < 0 or int(overlap) >= int(size):
        raise ValueError("overlap must be >= 0 and < size")
    step = int(size) - int(overlap)
    n = len(text)
    if n == 0:
        return ()
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + int(size), n)
        spans.append((start, end))
        if end >= n:
            break
        start += step
    return tuple(spans)


def _meta_tuple(metadata: Mapping[str, object] | Sequence[tuple[str, str]] | None) -> tuple[tuple[str, str], ...]:
    if not metadata:
        return ()
    if isinstance(metadata, Mapping):
        items = [(str(key), str(value)) for key, value in metadata.items()]
    else:
        items = [(str(key), str(value)) for key, value in metadata]
    return tuple(sorted(items))


def with_corpus_version(corpus: Corpus, version: str) -> Corpus:
    return Corpus(
        version=version,
        documents=corpus.documents,
        embedding=corpus.embedding,
        authored_for=corpus.authored_for,
        source_mission=corpus.source_mission,
    )


def rechunk_document(document: Document, spec: ChunkSpec) -> Document:
    if spec.mode == "canonical":
        return document
    if spec.mode == "merged":
        chunk = Chunk(
            chunk_id=f"{document.document_id}::merged",
            document_id=document.document_id,
            text=document.text,
            span_start=0,
            span_end=len(document.text),
            m28_id="",
            metadata=document.metadata,
        )
        return Document(
            document_id=document.document_id,
            title=document.title,
            text=document.text,
            metadata=document.metadata,
            chunks=(chunk,),
        )
    if spec.mode != "windows":
        raise ValueError(f"unsupported chunk mode: {spec.mode}")
    if spec.size is None:
        raise ValueError("windows mode requires chunk size")
    chunks: list[Chunk] = []
    for start, end in window_spans(document.text, spec.size, spec.overlap):
        text = document.text[start:end]
        if not has_content_tokens(text):
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::s{start}-e{end}",
                document_id=document.document_id,
                text=text,
                span_start=start,
                span_end=end,
                m28_id="",
                metadata=document.metadata,
            )
        )
    if not chunks:
        raise ValueError(f"windowing produced no content chunks for {document.document_id}")
    return Document(
        document_id=document.document_id,
        title=document.title,
        text=document.text,
        metadata=document.metadata,
        chunks=tuple(chunks),
    )


def rechunk_corpus(source: Corpus, spec: ChunkSpec) -> Corpus:
    if spec.mode == "canonical":
        return with_corpus_version(source, spec.corpus_version)
    documents = tuple(rechunk_document(document, spec) for document in source.documents)
    return Corpus(
        version=spec.corpus_version,
        documents=documents,
        embedding=source.embedding,
        authored_for="M35",
        source_mission="M33",
    )


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def project_labels(
    query: FrozenQuery,
    target: Corpus,
    source: Corpus,
    *,
    extra_trap_ids: Sequence[str] = (),
) -> ProjectedLabels:
    target_ids = {chunk.chunk_id for chunk in target.chunks()}
    needed = set(query.relevant_chunk_ids) | set(query.support_chunk_ids) | set(query.traps)
    if needed <= target_ids:
        traps = tuple(dict.fromkeys((*query.traps, *tuple(str(item) for item in extra_trap_ids))))
        mixed = tuple(chunk_id for chunk_id in query.relevant_chunk_ids if chunk_id in set(query.traps))
        return ProjectedLabels(
            query_id=query.query_id,
            corpus_version=target.version,
            relevant_ids=query.relevant_chunk_ids,
            support_ids=query.support_chunk_ids,
            trap_ids=traps,
            mixed_ids=mixed,
            source_relevant_ids=query.relevant_chunk_ids,
            source_support_ids=query.support_chunk_ids,
        )

    source_map = {chunk.chunk_id: chunk for chunk in source.chunks()}

    def locate(ids: Sequence[str]) -> tuple[tuple[str, int, int], ...]:
        found = []
        for chunk_id in ids:
            chunk = source_map[chunk_id]
            found.append((chunk.document_id, chunk.span_start, chunk.span_end))
        return tuple(found)

    relevant_spans = locate(query.relevant_chunk_ids)
    support_spans = locate(query.support_chunk_ids)
    trap_spans = locate(query.traps)

    def overlaps(chunk: Chunk, spans: Sequence[tuple[str, int, int]]) -> bool:
        return any(
            chunk.document_id == document_id and _spans_overlap(chunk.span_start, chunk.span_end, start, end)
            for document_id, start, end in spans
        )

    relevant: list[str] = []
    support: list[str] = []
    traps: list[str] = []
    mixed: list[str] = []
    for chunk in target.chunks():
        is_support = overlaps(chunk, support_spans)
        is_relevant = is_support or overlaps(chunk, relevant_spans)
        is_trap = overlaps(chunk, trap_spans)
        if is_support:
            support.append(chunk.chunk_id)
        if is_relevant:
            relevant.append(chunk.chunk_id)
        if is_trap:
            traps.append(chunk.chunk_id)
        if is_relevant and is_trap:
            mixed.append(chunk.chunk_id)
    traps.extend(str(item) for item in extra_trap_ids)
    return ProjectedLabels(
        query_id=query.query_id,
        corpus_version=target.version,
        relevant_ids=tuple(dict.fromkeys(relevant)),
        support_ids=tuple(dict.fromkeys(support)),
        trap_ids=tuple(dict.fromkeys(traps)),
        mixed_ids=tuple(dict.fromkeys(mixed)),
        source_relevant_ids=query.relevant_chunk_ids,
        source_support_ids=query.support_chunk_ids,
    )


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    depth = int(k)
    if depth < 1:
        raise ValueError("k must be positive")
    total = 0.0
    for index, gain in enumerate(list(gains)[:depth], start=1):
        total += float(gain) / math.log2(index + 1)
    return total


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float | None:
    relevant = [chunk_id for chunk_id in relevant_ids if chunk_id]
    if not relevant:
        return None
    retrieved = set(list(ranked_ids)[: int(k)])
    return len(retrieved.intersection(relevant)) / len(set(relevant))


def mean_reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float | None:
    relevant = set(relevant_ids)
    if not relevant:
        return None
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int) -> float | None:
    positive = {chunk_id: float(gain) for chunk_id, gain in grades.items() if float(gain) > 0.0}
    if not positive:
        return None
    gains = [float(grades.get(chunk_id, 0.0)) for chunk_id in list(ranked_ids)[: int(k)]]
    ideal = sorted(positive.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return None
    return dcg_at_k(gains, k) / idcg


def first_rank(ranked_ids: Sequence[str], wanted: Sequence[str]) -> int | None:
    wanted_set = set(wanted)
    if not wanted_set:
        return None
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in wanted_set:
            return rank
    return None


def _longest_consecutive_overlap(query_text: str, chunk_text: str) -> float:
    query_tokens = tokenize(query_text)
    chunk_tokens = tokenize(chunk_text)
    if not query_tokens or not chunk_tokens:
        return 0.0
    chunk_len = len(chunk_tokens)
    query_len = len(query_tokens)
    for size in range(query_len, 0, -1):
        for start in range(0, query_len - size + 1):
            ngram = query_tokens[start : start + size]
            for cursor in range(0, chunk_len - size + 1):
                if chunk_tokens[cursor : cursor + size] == ngram:
                    return size / query_len
    return 0.0


def lexical_rerank_score(query_text: str, chunk_text: str) -> float:
    """Local teaching reranker. Does not read relevance labels."""

    q_digits = query_digits(query_text)
    chunk_token_set = frozenset(tokenize(chunk_text))
    q_stems = content_stems(query_text)
    c_stems = content_stems(chunk_text)
    digit_coverage = (len(q_digits & chunk_token_set) / len(q_digits)) if q_digits else 0.0
    extra_digits = {token for token in tokenize(chunk_text) if token.isdigit()} - q_digits
    digit_penalty = 1.0 if q_digits and extra_digits and not q_digits <= chunk_token_set else 0.0
    overlap = (len(q_stems & c_stems) / len(q_stems)) if q_stems else 0.0
    procedure = bool(PROCEDURE_RE.search(query_text))
    action = bool(ACTION_HINT_RE.match(str(chunk_text).strip()))
    procedure_bonus = 1.0 if procedure and action else 0.0
    query_neg = "not" in tokenize(query_text) or "never" in tokenize(query_text)
    chunk_neg = "not" in tokenize(chunk_text) or "never" in tokenize(chunk_text)
    negation_align = 1.0 if query_neg and chunk_neg else 0.0
    negation_mismatch = 1.0 if query_neg and not chunk_neg else 0.0
    consecutive = _longest_consecutive_overlap(query_text, chunk_text)
    return (
        4.0 * digit_coverage
        - 3.0 * digit_penalty
        + 2.0 * overlap
        + 1.5 * procedure_bonus
        + 1.5 * negation_align
        - 1.0 * negation_mismatch
        + 0.5 * consecutive
    )


def _hit_to_candidate(hit: RankedHit, *, reranker_id: str, rerank_score: float | None = None) -> RankedCandidate:
    evidence = hit.as_evidence()
    return RankedCandidate(
        rank=int(hit.rank),
        original_rank=int(hit.rank),
        score=float(hit.score),
        cosine_score=float(hit.score),
        rerank_score=rerank_score,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        text=hit.text,
        evidence=evidence,
    )


def generate_candidates(
    query_text: str,
    *,
    query_id: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    index: ExactIndex | None = None,
    corpus: Corpus | None = None,
) -> CandidateSet:
    """Query -> M33 search. Returns a candidate set, not an answer."""

    used_index = index or load_canonical_index()
    used_corpus = corpus or load_canonical_corpus()
    encoded = encode_query(query_text, query_id=query_id)
    started = time.perf_counter()
    response: SearchResponse = search(
        used_index,
        encoded,
        top_k=int(candidate_k),
        query_id=query_id,
        live_corpus=used_corpus,
        enforce_freshness=True,
        enforce_provenance=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    items = tuple(_hit_to_candidate(hit, reranker_id=RERANKER_IDENTITY) for hit in response.hits)
    return CandidateSet(
        query_id=query_id,
        query_text=query_text,
        candidate_k=int(candidate_k),
        reranker_id=RERANKER_IDENTITY,
        items=items,
        scored_candidates=int(response.scored_candidates),
        latency_ms=float(response.latency_ms if response.latency_ms else latency_ms),
        index_id=response.index_id,
        source_hash=response.source_hash,
        corpus_version=response.corpus_version,
        model=response.embedding.model,
        version=response.embedding.version,
        metric=response.metric,
        normalization=response.normalization,
    )


def rerank_candidates(
    candidates: CandidateSet,
    *,
    reranker_id: str,
    query_text: str | None = None,
) -> CandidateSet:
    """Reorder an identical candidate set. Members do not change."""

    if reranker_id == RERANKER_OPTIONAL_MODEL:
        raise ValueError("optional model reranker is not on the required path")
    if reranker_id not in SUPPORTED_RERANKERS:
        raise ValueError(f"unsupported reranker: {reranker_id}")
    query = query_text if query_text is not None else candidates.query_text
    if reranker_id == RERANKER_IDENTITY:
        items = tuple(
            RankedCandidate(
                rank=item.original_rank,
                original_rank=item.original_rank,
                score=item.cosine_score,
                cosine_score=item.cosine_score,
                rerank_score=None,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                evidence=dict(item.evidence),
            )
            for item in candidates.items
        )
        return CandidateSet(
            query_id=candidates.query_id,
            query_text=candidates.query_text,
            candidate_k=candidates.candidate_k,
            reranker_id=RERANKER_IDENTITY,
            items=items,
            scored_candidates=candidates.scored_candidates,
            latency_ms=candidates.latency_ms,
            index_id=candidates.index_id,
            source_hash=candidates.source_hash,
            corpus_version=candidates.corpus_version,
            model=candidates.model,
            version=candidates.version,
            metric=candidates.metric,
            normalization=candidates.normalization,
        )

    scored: list[tuple[float, RankedCandidate]] = []
    for item in candidates.items:
        score = lexical_rerank_score(query, item.text)
        scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].original_rank, pair[1].chunk_id))
    reranked: list[RankedCandidate] = []
    for rank, (score, item) in enumerate(scored, start=1):
        evidence = dict(item.evidence)
        evidence["rank"] = rank
        evidence["score"] = float(score)
        reranked.append(
            RankedCandidate(
                rank=rank,
                original_rank=item.original_rank,
                score=float(score),
                cosine_score=item.cosine_score,
                rerank_score=float(score),
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                evidence=evidence,
            )
        )
    return CandidateSet(
        query_id=candidates.query_id,
        query_text=candidates.query_text,
        candidate_k=candidates.candidate_k,
        reranker_id=reranker_id,
        items=tuple(reranked),
        scored_candidates=candidates.scored_candidates,
        latency_ms=candidates.latency_ms,
        index_id=candidates.index_id,
        source_hash=candidates.source_hash,
        corpus_version=candidates.corpus_version,
        model=candidates.model,
        version=candidates.version,
        metric=candidates.metric,
        normalization=candidates.normalization,
    )


def classify_retrieval_failure(report_fields: Mapping[str, object]) -> str:
    if not report_fields.get("answerable"):
        return FAILURE_UNANSWERABLE
    support_hit = report_fields.get("candidate_support_hit")
    if report_fields.get("support_ids") and support_hit is False:
        return FAILURE_CANDIDATE_MISS
    if report_fields.get("trap_at_1"):
        return FAILURE_TRAP_AT_1
    first_relevant = report_fields.get("first_relevant_rank")
    if first_relevant is None:
        return FAILURE_CANDIDATE_MISS
    if int(first_relevant) > 1:
        return FAILURE_RANKING_MISS
    ndcg = report_fields.get("ndcg_at_k")
    if ndcg is not None and float(ndcg) < 1.0:
        return FAILURE_SUBOPTIMAL
    return FAILURE_OK


def proxy_cost(*, scored_candidates: int, candidate_k: int, reranker_id: str) -> tuple[int, int]:
    rerank_cost = int(candidate_k) if reranker_id != RERANKER_IDENTITY else 0
    return rerank_cost, int(scored_candidates) + rerank_cost


def evaluate_candidates(
    query: FrozenQuery,
    candidates: CandidateSet,
    projected: ProjectedLabels,
    *,
    config: ExperimentConfig,
) -> QueryReport:
    ranked_ids = candidates.ids()
    cosine_ids = tuple(
        item.chunk_id for item in sorted(candidates.items, key=lambda row: (row.original_rank, row.chunk_id))
    )
    grades = projected.grades()
    recall = recall_at_k(ranked_ids, projected.relevant_ids, candidates.candidate_k)
    candidate_recall = recall_at_k(ranked_ids, projected.relevant_ids, len(ranked_ids) or candidates.candidate_k)
    support_recall = recall_at_k(ranked_ids, projected.support_ids, len(ranked_ids) or candidates.candidate_k)
    mrr = mean_reciprocal_rank(ranked_ids, projected.relevant_ids)
    ndcg = ndcg_at_k(ranked_ids, grades, candidates.candidate_k)
    trap_at_1 = bool(ranked_ids) and ranked_ids[0] in set(projected.trap_ids)
    rerank_cost, cost = proxy_cost(
        scored_candidates=candidates.scored_candidates,
        candidate_k=candidates.candidate_k,
        reranker_id=candidates.reranker_id,
    )
    support_hit = None if not query.answerable or not projected.support_ids else bool((support_recall or 0.0) > 0.0)
    fields = {
        "answerable": query.answerable,
        "support_ids": projected.support_ids,
        "candidate_support_hit": support_hit,
        "trap_at_1": trap_at_1,
        "first_relevant_rank": first_rank(ranked_ids, projected.relevant_ids),
        "ndcg_at_k": ndcg,
    }
    return QueryReport(
        query_id=query.query_id,
        query_text=query.text,
        split=query.split,
        answerable=query.answerable,
        experiment_id=config.experiment_id,
        corpus_version=candidates.corpus_version,
        index_id=candidates.index_id,
        source_hash=candidates.source_hash,
        candidate_k=candidates.candidate_k,
        reranker_id=candidates.reranker_id,
        config_identity=config.identity(),
        candidate_ids=cosine_ids,
        ranked_ids=ranked_ids,
        cosine_ids=cosine_ids,
        relevant_ids=projected.relevant_ids,
        support_ids=projected.support_ids,
        trap_ids=projected.trap_ids,
        mixed_ids=projected.mixed_ids,
        recall_at_k=recall,
        mrr=mrr,
        ndcg_at_k=ndcg,
        first_relevant_rank=first_rank(ranked_ids, projected.relevant_ids),
        first_support_rank=first_rank(ranked_ids, projected.support_ids),
        candidate_recall=candidate_recall,
        candidate_support_hit=support_hit,
        trap_at_1=trap_at_1,
        scored_candidates=candidates.scored_candidates,
        rerank_cost=rerank_cost,
        proxy_cost=cost,
        latency_ms=candidates.latency_ms,
        failure_mode=classify_retrieval_failure(fields),
        eval_version=query.eval_version,
    )


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _slice_summary(rows: Sequence[QueryReport]) -> dict[str, object]:
    answerable = [row for row in rows if row.answerable]
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "mean_recall_at_k": _mean([row.recall_at_k for row in answerable]),
        "mean_mrr": _mean([row.mrr for row in answerable]),
        "mean_ndcg_at_k": _mean([row.ndcg_at_k for row in answerable]),
        "mean_candidate_recall": _mean([row.candidate_recall for row in answerable]),
        "trap_at_1_rate": (sum(1 for row in rows if row.trap_at_1) / len(rows)) if rows else 0.0,
        "failure_modes": tuple(row.failure_mode for row in rows),
        "query_ids": tuple(row.query_id for row in rows),
    }


def slice_rows(rows: Sequence[QueryReport]) -> dict[str, tuple[QueryReport, ...]]:
    by_id = {row.query_id: row for row in rows}
    return {
        "all": tuple(rows),
        "dev_answerable": tuple(row for row in rows if row.split == "dev" and row.answerable),
        "holdout_answerable": tuple(row for row in rows if row.split == "holdout" and row.answerable),
        "unanswerable": tuple(row for row in rows if not row.answerable),
        "critical": tuple(by_id[query_id] for query_id in CRITICAL_QUERY_IDS if query_id in by_id),
        "holdout": tuple(row for row in rows if row.split == "holdout"),
    }


def extra_traps_for(query_id: str, payload: Mapping[str, object] | None = None) -> tuple[str, ...]:
    bundle = payload if payload is not None else load_hard_negatives_payload()
    ids = []
    for item in bundle.get("items", ()):
        targets = tuple(str(value) for value in item.get("target_queries", ()))
        if query_id in targets:
            ids.append(str(item["chunk_id"]))
    return tuple(ids)


def inject_hard_negatives(corpus: Corpus, payload: Mapping[str, object] | None = None) -> Corpus:
    bundle = payload if payload is not None else load_hard_negatives_payload()
    extra_docs = []
    for item in bundle.get("items", ()):
        extra_docs.append(
            compose_document(
                str(item["document_id"]),
                str(item.get("title") or item["document_id"]),
                item.get("metadata") or {"topic": "hardneg", "source": "m35", "locale": "en"},
                (
                    {
                        "chunk_id": str(item["chunk_id"]),
                        "local_id": "c0",
                        "text": str(item["text"]),
                        "m28_id": "",
                        "metadata": item.get("metadata") or {"topic": "hardneg", "source": "m35", "locale": "en"},
                    },
                ),
            )
        )
    return Corpus(
        version=str(bundle.get("corpus_version") or "m35.corpus.hardneg.v1"),
        documents=corpus.documents + tuple(extra_docs),
        embedding=corpus.embedding,
        authored_for="M35",
        source_mission="M33",
    )


def leak_eval_phrasing(corpus: Corpus, queries: Sequence[FrozenQuery]) -> Corpus:
    """Copy eval query text into gold support chunks. Original corpus is not mutated."""

    live = corpus
    present = {chunk.chunk_id for chunk in live.chunks()}
    for query in queries:
        if not query.answerable or not query.support_chunk_ids:
            continue
        target = query.support_chunk_ids[0]
        if target not in present:
            continue
        old = live.get_chunk(target).text
        leaked_text = f"{old} {query.text}"
        live = replace_chunk_text(live, target, leaked_text)
        present = {chunk.chunk_id for chunk in live.chunks()}
    return with_corpus_version(live, "m35.corpus.leaked.v1")


def relabel_after_results(
    queries: Sequence[FrozenQuery],
    reports: Sequence[QueryReport],
) -> tuple[FrozenQuery, ...]:
    """Metric gaming: treat the observed top-1 as relevant after seeing ranks."""

    by_id = {row.query_id: row for row in reports}
    gamed: list[FrozenQuery] = []
    for query in queries:
        report = by_id.get(query.query_id)
        if report is None or not query.answerable or not report.ranked_ids:
            gamed.append(query)
            continue
        top = report.ranked_ids[0]
        if top in query.relevant_chunk_ids:
            gamed.append(query)
            continue
        gamed.append(
            replace(
                query,
                relevant_chunk_ids=query.relevant_chunk_ids + (top,),
                notes="relabeled after results",
            )
        )
    return tuple(gamed)


def repair_eval_boundary(
    *,
    broken_corpus: Corpus,
    source_corpus: Corpus,
    gamed_labels: Sequence[FrozenQuery] | None = None,
    frozen_labels: Sequence[FrozenQuery] | None = None,
) -> tuple[Corpus, tuple[FrozenQuery, ...]]:
    """Restore the clean corpus and frozen labels. Broken objects stay broken."""

    del broken_corpus
    labels = tuple(frozen_labels or load_frozen_queries())
    if gamed_labels is not None and label_hash(gamed_labels) == label_hash(labels):
        # still return frozen labels; caller keeps gamed_labels separately
        pass
    return source_corpus, labels


def build_version_index(corpus: Corpus, *, index_id: str | None = None) -> ExactIndex:
    if corpus.version == CANONICAL_CORPUS_VERSION and not any(
        chunk.chunk_id.endswith("::merged") or "::s" in chunk.chunk_id for chunk in corpus.chunks()
    ):
        canonical = load_canonical_index()
        if source_hash(corpus) == canonical.metadata.source_hash:
            return canonical
    used_id = index_id or f"v09-exact-{corpus.version}"
    return build_index(corpus, index_id=used_id)


def materialize(
    config: ExperimentConfig,
    *,
    source_corpus: Corpus | None = None,
    queries: Sequence[FrozenQuery] | None = None,
) -> tuple[Corpus, ExactIndex, ChunkSpec]:
    source = source_corpus or load_canonical_corpus()
    specs = chunk_spec_map()
    if config.corpus_version in specs:
        spec = specs[config.corpus_version]
    else:
        spec = ChunkSpec(
            corpus_version=config.corpus_version,
            mode=config.chunk_mode,
            size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
    corpus = rechunk_corpus(source, spec)
    if config.hard_negatives:
        corpus = inject_hard_negatives(corpus)
    if config.leaked:
        corpus = leak_eval_phrasing(corpus, queries or load_frozen_queries())
    index = build_version_index(corpus)
    return corpus, index, spec


def evaluate_query(
    query: FrozenQuery,
    *,
    config: ExperimentConfig,
    corpus: Corpus,
    index: ExactIndex,
    source_corpus: Corpus,
    candidates: CandidateSet | None = None,
) -> QueryReport:
    used_candidates = candidates or generate_candidates(
        query.text,
        query_id=query.query_id,
        candidate_k=config.candidate_k,
        index=index,
        corpus=corpus,
    )
    ranked = rerank_candidates(used_candidates, reranker_id=config.reranker_id, query_text=query.text)
    extra = extra_traps_for(query.query_id) if config.hard_negatives else ()
    projected = project_labels(query, corpus, source_corpus, extra_trap_ids=extra)
    return evaluate_candidates(query, ranked, projected, config=config)


def evaluate_set(
    *,
    config: ExperimentConfig | None = None,
    queries: Sequence[FrozenQuery] | None = None,
    source_corpus: Corpus | None = None,
    split: str | None = None,
) -> SetReport:
    used_config = config or baseline_config()
    frozen = tuple(queries or load_frozen_queries())
    if used_config.eval_version != EVAL_VERSION:
        raise ValueError("eval labels must remain m34.eval.v1")
    if split is not None:
        frozen = tuple(query for query in frozen if query.split == split)
    source = source_corpus or load_canonical_corpus()
    corpus, index, _spec = materialize(used_config, source_corpus=source, queries=frozen)
    rows = []
    for query in frozen:
        rows.append(
            evaluate_query(
                query,
                config=used_config,
                corpus=corpus,
                index=index,
                source_corpus=source,
            )
        )
    return assemble_set_report(used_config, tuple(rows), corpus=corpus, index=index, queries=frozen)


def assemble_set_report(
    config: ExperimentConfig,
    rows: Sequence[QueryReport],
    *,
    corpus: Corpus,
    index: ExactIndex,
    queries: Sequence[FrozenQuery],
) -> SetReport:
    slices = {name: _slice_summary(items) for name, items in slice_rows(rows).items()}
    answerable = [row for row in rows if row.answerable]
    return SetReport(
        config=config,
        config_identity=config.identity(),
        eval_version=config.eval_version,
        corpus_version=corpus.version,
        index_id=index.metadata.index_id,
        source_hash=index.metadata.source_hash,
        n=len(rows),
        n_answerable=len(answerable),
        mean_recall_at_k=_mean([row.recall_at_k for row in answerable]),
        mean_mrr=_mean([row.mrr for row in answerable]),
        mean_ndcg_at_k=_mean([row.ndcg_at_k for row in answerable]),
        mean_candidate_recall=_mean([row.candidate_recall for row in answerable]),
        trap_at_1_rate=(sum(1 for row in rows if row.trap_at_1) / len(rows)) if rows else 0.0,
        scored_candidates=rows[0].scored_candidates if rows else 0,
        mean_proxy_cost=float(_mean([row.proxy_cost for row in rows]) or 0.0),
        slices=slices,
        rows=tuple(rows),
        label_hash=label_hash(queries),
        questions_sha256=questions_sha256(),
        leaked=config.leaked,
        relabeled=config.relabeled,
    )


def rescore_with_labels(
    rows: Sequence[QueryReport],
    queries: Sequence[FrozenQuery],
    *,
    config: ExperimentConfig,
    corpus: Corpus,
    index: ExactIndex,
    source_corpus: Corpus,
) -> SetReport:
    """Recompute metrics for frozen rankings under a (possibly gamed) label set."""

    query_map = {query.query_id: query for query in queries}
    rescored: list[QueryReport] = []
    extra_fn = extra_traps_for if config.hard_negatives else lambda _query_id: ()
    for row in rows:
        query = query_map[row.query_id]
        projected = project_labels(query, corpus, source_corpus, extra_trap_ids=extra_fn(query.query_id))
        grades = projected.grades()
        recall = recall_at_k(row.ranked_ids, projected.relevant_ids, row.candidate_k)
        candidate_recall = recall_at_k(row.ranked_ids, projected.relevant_ids, len(row.ranked_ids) or row.candidate_k)
        support_recall = recall_at_k(row.ranked_ids, projected.support_ids, len(row.ranked_ids) or row.candidate_k)
        mrr = mean_reciprocal_rank(row.ranked_ids, projected.relevant_ids)
        ndcg = ndcg_at_k(row.ranked_ids, grades, row.candidate_k)
        trap_at_1 = bool(row.ranked_ids) and row.ranked_ids[0] in set(projected.trap_ids)
        support_hit = None if not query.answerable or not projected.support_ids else bool((support_recall or 0.0) > 0.0)
        fields = {
            "answerable": query.answerable,
            "support_ids": projected.support_ids,
            "candidate_support_hit": support_hit,
            "trap_at_1": trap_at_1,
            "first_relevant_rank": first_rank(row.ranked_ids, projected.relevant_ids),
            "ndcg_at_k": ndcg,
        }
        rescored.append(
            replace(
                row,
                relevant_ids=projected.relevant_ids,
                support_ids=projected.support_ids,
                trap_ids=projected.trap_ids,
                mixed_ids=projected.mixed_ids,
                recall_at_k=recall,
                mrr=mrr,
                ndcg_at_k=ndcg,
                first_relevant_rank=first_rank(row.ranked_ids, projected.relevant_ids),
                first_support_rank=first_rank(row.ranked_ids, projected.support_ids),
                candidate_recall=candidate_recall,
                candidate_support_hit=support_hit,
                trap_at_1=trap_at_1,
                failure_mode=classify_retrieval_failure(fields),
            )
        )
    gamed_config = replace(config, relabeled=True, experiment_id=f"{config.experiment_id}-relabeled")
    return assemble_set_report(gamed_config, tuple(rescored), corpus=corpus, index=index, queries=queries)


def worst_queries(rows: Sequence[QueryReport], *, n: int = 5) -> tuple[QueryReport, ...]:
    answerable = [row for row in rows if row.answerable]
    ordered = sorted(
        answerable,
        key=lambda row: (
            row.ndcg_at_k if row.ndcg_at_k is not None else 0.0,
            row.mrr if row.mrr is not None else 0.0,
            row.recall_at_k if row.recall_at_k is not None else 0.0,
            row.query_id,
        ),
    )
    return tuple(ordered[: int(n)])


def round_metric(value: float | None, digits: int = 10) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
