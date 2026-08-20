"""Exact in-memory semantic search for Mission M33.

M33 is the V08 retriever: query text -> M28 embedding contract -> exact
cosine over a versioned chunk index -> ranked evidence (ids, scores,
text spans, provenance). Approximate search, hybrid fusion, reranking,
and generation remain deferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from missions.M28.embedding_core import (  # noqa: E402
    COMPATIBILITY_FIELDS,
    Provenance,
    ProvenanceError,
    as_vector,
    compatibility_report,
    cosine_similarity,
    l2_norm,
    load_encoder,
    operational_score,
    provenance_from_mapping,
)

SEPARATOR = "\n\n"
FILTER_SCHEMA = ("topic", "source", "locale")
TIE_BREAK = ("-score", "chunk_id")
INDEX_BACKEND = "exact-in-memory"
INDEX_ID = "v08-exact-memory"
CORPUS_VERSION = "m33.corpus.v1"
STALE_POLICY = "fail-closed-rebuild-or-reject"
DEFAULT_TOP_K = 5
VECTOR_DIGITS = 10


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M33 requires numpy; install requirements/m33.txt") from exc
    return np


def default_dataset_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M33"
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled search fixtures: {path}")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _vector_tuple(values: Sequence[float], *, name: str, dimensions: int | None = None) -> tuple[float, ...]:
    return as_vector(values, name=name, dimensions=dimensions)


def _rounded_vector(values: Sequence[float]) -> list[float]:
    return [round(float(value), VECTOR_DIGITS) for value in values]


def _meta_tuple(metadata: Mapping[str, object] | Sequence[tuple[str, str]] | None) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    if isinstance(metadata, Mapping):
        items = [(str(key), str(value)) for key, value in metadata.items()]
    else:
        items = [(str(key), str(value)) for key, value in metadata]
    return tuple(sorted(items))


def metadata_map(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    return {key: value for key, value in pairs}


class FilterSchemaError(ValueError):
    """Raised when a query filter uses a key outside the declared schema."""

    def __init__(self, *, unknown: tuple[str, ...], schema: tuple[str, ...]):
        self.unknown = unknown
        self.schema = schema
        detail = ", ".join(unknown) if unknown else "unknown"
        super().__init__(f"filter keys not in schema: {detail}")


class IndexStaleError(ValueError):
    """Raised when the live corpus no longer matches the index source hash."""

    def __init__(
        self,
        *,
        index_id: str,
        index_source_hash: str,
        live_source_hash: str,
        corpus_version: str,
    ):
        self.index_id = index_id
        self.index_source_hash = index_source_hash
        self.live_source_hash = live_source_hash
        self.corpus_version = corpus_version
        super().__init__(
            "stale index: live corpus hash "
            f"{live_source_hash[:12]} != indexed {index_source_hash[:12]}"
        )


class IndexIncompatibleError(ValueError):
    """Raised when query embedding provenance cannot mix with the index."""

    def __init__(self, *, mismatches: tuple[str, ...], query: Provenance, index: Provenance):
        self.mismatches = mismatches
        self.query = query
        self.index = index
        detail = ", ".join(mismatches) if mismatches else "unknown"
        super().__init__(f"incompatible index embedding provenance: {detail}")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    span_start: int
    span_end: int
    m28_id: str
    metadata: tuple[tuple[str, str], ...]

    def metadata_dict(self) -> dict[str, str]:
        return metadata_map(self.metadata)

    def span(self) -> dict[str, int]:
        return {"start": self.span_start, "end": self.span_end}


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    metadata: tuple[tuple[str, str], ...]
    chunks: tuple[Chunk, ...]

    def metadata_dict(self) -> dict[str, str]:
        return metadata_map(self.metadata)


@dataclass(frozen=True)
class Corpus:
    version: str
    documents: tuple[Document, ...]
    embedding: Provenance
    authored_for: str = "M33"
    source_mission: str = "M28"

    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(chunk for document in self.documents for chunk in document.chunks)

    def get_document(self, document_id: str) -> Document:
        for document in self.documents:
            if document.document_id == document_id:
                return document
        raise KeyError(document_id)

    def get_chunk(self, chunk_id: str) -> Chunk:
        for chunk in self.chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(chunk_id)


@dataclass(frozen=True)
class QueryEmbedding:
    query_id: str
    text: str
    vector: tuple[float, ...]
    provenance: Provenance


@dataclass(frozen=True)
class LabeledQuery:
    query_id: str
    text: str
    experiment: str
    relevant_chunk_ids: tuple[str, ...]
    traps: tuple[str, ...]
    hard_neighbor: str | None = None


@dataclass(frozen=True)
class IndexRecord:
    chunk: Chunk
    vector: tuple[float, ...]


@dataclass(frozen=True)
class IndexMetadata:
    index_id: str
    backend: str
    corpus_version: str
    source_hash: str
    vector_hash: str
    embedding: Provenance
    filter_schema: tuple[str, ...]
    tie_break: tuple[str, ...]
    stale_policy: str
    document_count: int
    chunk_count: int
    downloaded: bool
    network_required: bool

    def fingerprint(self) -> dict[str, object]:
        payload = {field: getattr(self.embedding, field) for field in COMPATIBILITY_FIELDS}
        payload.update(
            {
                "index_id": self.index_id,
                "backend": self.backend,
                "corpus_version": self.corpus_version,
                "source_hash": self.source_hash,
                "vector_hash": self.vector_hash,
                "filter_schema": list(self.filter_schema),
                "tie_break": list(self.tie_break),
                "stale_policy": self.stale_policy,
                "downloaded": self.downloaded,
                "network_required": self.network_required,
            }
        )
        return payload


@dataclass(frozen=True)
class ExactIndex:
    records: tuple[IndexRecord, ...]
    metadata: IndexMetadata

    def get(self, chunk_id: str) -> IndexRecord:
        for record in self.records:
            if record.chunk.chunk_id == chunk_id:
                return record
        raise KeyError(chunk_id)

    def matrix(self):
        np = _require_numpy()
        if not self.records:
            raise ValueError("index has no records")
        return np.asarray([record.vector for record in self.records], dtype=float)


@dataclass(frozen=True)
class RankedHit:
    rank: int
    score: float
    document_id: str
    chunk_id: str
    text: str
    span_start: int
    span_end: int
    metadata: tuple[tuple[str, str], ...]
    m28_id: str
    vector: tuple[float, ...]

    def metadata_dict(self) -> dict[str, str]:
        return metadata_map(self.metadata)

    def as_evidence(self) -> dict[str, object]:
        """M34-facing ranked evidence row. No generation or citations."""

        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "text": self.text,
            "span": {"start": self.span_start, "end": self.span_end},
            "metadata": self.metadata_dict(),
            "m28_id": self.m28_id,
        }


@dataclass(frozen=True)
class SearchResponse:
    query_id: str
    query_text: str
    query_vector: tuple[float, ...]
    top_k: int
    filters: tuple[tuple[str, str], ...]
    hits: tuple[RankedHit, ...]
    scored_candidates: int
    latency_ms: float
    index_id: str
    corpus_version: str
    source_hash: str
    embedding: Provenance
    metric: str
    normalization: str
    enforced_freshness: bool
    enforced_provenance: bool
    backend: str

    def ids(self) -> tuple[str, ...]:
        return tuple(hit.chunk_id for hit in self.hits)

    def document_ids(self) -> tuple[str, ...]:
        return tuple(hit.document_id for hit in self.hits)

    @property
    def top_id(self) -> str | None:
        return self.hits[0].chunk_id if self.hits else None

    def evidence(self) -> tuple[dict[str, object], ...]:
        return tuple(hit.as_evidence() for hit in self.hits)


def compose_document(
    document_id: str,
    title: str,
    metadata: Mapping[str, object],
    parts: Sequence[Mapping[str, object]],
) -> Document:
    """Join chunk texts with a separator and assign stable spans."""

    meta = _meta_tuple(metadata)
    texts = [str(part["text"]) for part in parts]
    full = SEPARATOR.join(texts)
    cursor = 0
    chunks: list[Chunk] = []
    for index, part in enumerate(parts):
        if index:
            cursor += len(SEPARATOR)
        text = str(part["text"])
        start = cursor
        end = start + len(text)
        local_id = str(part.get("local_id") or f"c{index}")
        chunk_id = str(part.get("chunk_id") or f"{document_id}::{local_id}")
        chunk_meta = _meta_tuple(part.get("metadata") or metadata)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                span_start=start,
                span_end=end,
                m28_id=str(part.get("m28_id") or ""),
                metadata=chunk_meta,
            )
        )
        cursor = end
    document = Document(
        document_id=document_id,
        title=title,
        text=full,
        metadata=meta,
        chunks=tuple(chunks),
    )
    _assert_spans(document)
    return document


def _assert_spans(document: Document) -> None:
    for chunk in document.chunks:
        if chunk.span_start < 0 or chunk.span_end > len(document.text):
            raise ValueError(f"{chunk.chunk_id} span is outside the document text")
        if document.text[chunk.span_start : chunk.span_end] != chunk.text:
            raise ValueError(f"{chunk.chunk_id} span does not recover chunk text")
        if chunk.document_id != document.document_id:
            raise ValueError(f"{chunk.chunk_id} document_id does not match parent")


def source_hash(corpus: Corpus) -> str:
    payload = []
    for document in corpus.documents:
        payload.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "text": document.text,
                "metadata": document.metadata_dict(),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "span_start": chunk.span_start,
                        "span_end": chunk.span_end,
                        "m28_id": chunk.m28_id,
                        "metadata": chunk.metadata_dict(),
                    }
                    for chunk in document.chunks
                ],
            }
        )
    return hashlib.sha256(_dump_canonical(payload).encode("utf-8")).hexdigest()


def vector_hash(records: Sequence[IndexRecord]) -> str:
    payload = [
        {"chunk_id": record.chunk.chunk_id, "vector": _rounded_vector(record.vector)}
        for record in records
    ]
    return hashlib.sha256(_dump_canonical(payload).encode("utf-8")).hexdigest()


def replace_chunk_text(corpus: Corpus, chunk_id: str, new_text: str) -> Corpus:
    """Return a new corpus with one chunk text replaced and spans rebuilt."""

    documents: list[Document] = []
    found = False
    for document in corpus.documents:
        if all(chunk.chunk_id != chunk_id for chunk in document.chunks):
            documents.append(document)
            continue
        found = True
        parts = []
        for chunk in document.chunks:
            text = new_text if chunk.chunk_id == chunk_id else chunk.text
            parts.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": text,
                    "m28_id": chunk.m28_id,
                    "metadata": chunk.metadata_dict(),
                    "local_id": chunk.chunk_id.split("::", 1)[-1],
                }
            )
        documents.append(
            compose_document(
                document.document_id,
                document.title,
                document.metadata_dict(),
                parts,
            )
        )
    if not found:
        raise KeyError(chunk_id)
    return Corpus(
        version=corpus.version,
        documents=tuple(documents),
        embedding=corpus.embedding,
        authored_for=corpus.authored_for,
        source_mission=corpus.source_mission,
    )


def _validate_filters(filters: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if not filters:
        return ()
    unknown = tuple(sorted(str(key) for key in filters if str(key) not in FILTER_SCHEMA))
    if unknown:
        raise FilterSchemaError(unknown=unknown, schema=FILTER_SCHEMA)
    return tuple(sorted((str(key), str(value)) for key, value in filters.items()))


def _matches(chunk: Chunk, filters: Mapping[str, str] | Sequence[tuple[str, str]] | None) -> bool:
    if not filters:
        return True
    items = filters.items() if isinstance(filters, Mapping) else filters
    meta = chunk.metadata_dict()
    return all(meta.get(key) == value for key, value in items)


def encode_query(
    text: str,
    *,
    query_id: str = "query",
    encoder=None,
) -> QueryEmbedding:
    encoder = encoder or load_encoder()
    vector = encoder.encode(text)
    return QueryEmbedding(
        query_id=query_id,
        text=text,
        vector=_vector_tuple(vector, name="query", dimensions=encoder.provenance.dimensions),
        provenance=encoder.provenance,
    )


def _as_query(
    query: str | QueryEmbedding,
    *,
    query_id: str,
    encoder=None,
) -> QueryEmbedding:
    if isinstance(query, QueryEmbedding):
        return query
    if not isinstance(query, str):
        raise TypeError("query must be text or QueryEmbedding")
    return encode_query(query, query_id=query_id, encoder=encoder)


def exact_cosine_rank(
    query_vector: Sequence[float],
    records: Sequence[IndexRecord],
    *,
    top_k: int,
    filters: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    metric: str = "cosine",
    normalization: str = "l2",
) -> tuple[tuple[IndexRecord, float], ...]:
    """Score every eligible record. top-k only truncates after scoring."""

    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    eligible = [record for record in records if _matches(record.chunk, filters)]
    if not eligible:
        return ()
    query = _vector_tuple(query_vector, name="query")
    scored: list[tuple[IndexRecord, float]] = []
    for record in eligible:
        if str(metric).lower() == "cosine":
            score = cosine_similarity(query, record.vector)
        else:
            score = operational_score(
                query,
                record.vector,
                metric=metric,
                normalization=normalization,
            )
        scored.append((record, float(score)))
    scored.sort(key=lambda item: (-item[1], item[0].chunk.chunk_id))
    return tuple(scored[: int(top_k)])


def build_index(
    corpus: Corpus,
    *,
    vectors: Mapping[str, Sequence[float]] | None = None,
    encoder=None,
    index_id: str = INDEX_ID,
) -> ExactIndex:
    chunks = corpus.chunks()
    if not chunks:
        raise ValueError("cannot index an empty corpus")
    seen: set[str] = set()
    width = corpus.embedding.dimensions
    resolved: dict[str, tuple[float, ...]] = {}
    if vectors is None:
        encoder = encoder or load_encoder()
        if encoder.provenance.downloaded or encoder.provenance.network_required:
            raise RuntimeError("M33 required path must use the bundled offline encoder")
        report = compatibility_report(encoder.provenance, corpus.embedding)
        if not report["compatible"]:
            raise IndexIncompatibleError(
                mismatches=tuple(report["mismatches"]),
                query=encoder.provenance,
                index=corpus.embedding,
            )
        for chunk in chunks:
            resolved[chunk.chunk_id] = _vector_tuple(
                encoder.encode(chunk.text),
                name=chunk.chunk_id,
                dimensions=width,
            )
    else:
        for chunk in chunks:
            if chunk.chunk_id not in vectors:
                raise KeyError(f"missing vector for {chunk.chunk_id}")
            resolved[chunk.chunk_id] = _vector_tuple(
                vectors[chunk.chunk_id],
                name=chunk.chunk_id,
                dimensions=width,
            )
    records: list[IndexRecord] = []
    unit_store = str(corpus.embedding.normalization).lower() == "l2"
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"duplicate chunk id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        vector = resolved[chunk.chunk_id]
        if unit_store and abs(l2_norm(vector) - 1.0) > 1e-6:
            raise ValueError(f"{chunk.chunk_id} is not unit length under L2 policy")
        records.append(IndexRecord(chunk=chunk, vector=vector))
    record_tuple = tuple(records)
    metadata = IndexMetadata(
        index_id=index_id,
        backend=INDEX_BACKEND,
        corpus_version=corpus.version,
        source_hash=source_hash(corpus),
        vector_hash=vector_hash(record_tuple),
        embedding=corpus.embedding,
        filter_schema=FILTER_SCHEMA,
        tie_break=TIE_BREAK,
        stale_policy=STALE_POLICY,
        document_count=len(corpus.documents),
        chunk_count=len(record_tuple),
        downloaded=False,
        network_required=False,
    )
    return ExactIndex(records=record_tuple, metadata=metadata)


def rebuild_index(
    index: ExactIndex,
    live_corpus: Corpus,
    *,
    encoder=None,
) -> ExactIndex:
    """Re-ingest live texts with the declared encoder. Same backend policy."""

    report = compatibility_report(live_corpus.embedding, index.metadata.embedding)
    if not report["compatible"]:
        raise IndexIncompatibleError(
            mismatches=tuple(report["mismatches"]),
            query=live_corpus.embedding,
            index=index.metadata.embedding,
        )
    return build_index(
        live_corpus,
        vectors=None,
        encoder=encoder,
        index_id=index.metadata.index_id,
    )


def assert_fresh(index: ExactIndex, live_corpus: Corpus) -> None:
    live = source_hash(live_corpus)
    if live == index.metadata.source_hash:
        return
    raise IndexStaleError(
        index_id=index.metadata.index_id,
        index_source_hash=index.metadata.source_hash,
        live_source_hash=live,
        corpus_version=index.metadata.corpus_version,
    )


def assert_index_compatible(query: Provenance, index: ExactIndex) -> None:
    report = compatibility_report(query, index.metadata.embedding)
    if report["compatible"]:
        return
    raise IndexIncompatibleError(
        mismatches=tuple(report["mismatches"]),
        query=query,
        index=index.metadata.embedding,
    )


def search(
    index: ExactIndex,
    query: str | QueryEmbedding,
    *,
    top_k: int = DEFAULT_TOP_K,
    filters: Mapping[str, object] | None = None,
    query_id: str = "query",
    live_corpus: Corpus | None = None,
    enforce_freshness: bool = True,
    enforce_provenance: bool = True,
    encoder=None,
) -> SearchResponse:
    """Query -> embed -> exact similarity -> top-k evidence.

    Eligibility filters run *before* scoring so only eligible chunks compete.
    top-k truncates after every eligible row is scored.
    """

    started = time.perf_counter()
    filter_pairs = _validate_filters(filters)
    encoded = _as_query(query, query_id=query_id, encoder=encoder)
    if enforce_provenance:
        assert_index_compatible(encoded.provenance, index)
    if enforce_freshness:
        if live_corpus is None:
            raise ValueError("live_corpus is required when enforce_freshness=True")
        assert_fresh(index, live_corpus)
    ranked = exact_cosine_rank(
        encoded.vector,
        index.records,
        top_k=top_k,
        filters=filter_pairs,
        metric=index.metadata.embedding.metric,
        normalization=index.metadata.embedding.normalization,
    )
    eligible_count = sum(1 for record in index.records if _matches(record.chunk, filter_pairs))
    hits = tuple(
        RankedHit(
            rank=offset + 1,
            score=score,
            document_id=record.chunk.document_id,
            chunk_id=record.chunk.chunk_id,
            text=record.chunk.text,
            span_start=record.chunk.span_start,
            span_end=record.chunk.span_end,
            metadata=record.chunk.metadata,
            m28_id=record.chunk.m28_id,
            vector=record.vector,
        )
        for offset, (record, score) in enumerate(ranked)
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return SearchResponse(
        query_id=encoded.query_id,
        query_text=encoded.text,
        query_vector=encoded.vector,
        top_k=int(top_k),
        filters=filter_pairs,
        hits=hits,
        scored_candidates=int(eligible_count),
        latency_ms=float(latency_ms),
        index_id=index.metadata.index_id,
        corpus_version=index.metadata.corpus_version,
        source_hash=index.metadata.source_hash,
        embedding=index.metadata.embedding,
        metric=index.metadata.embedding.metric,
        normalization=index.metadata.embedding.normalization,
        enforced_freshness=bool(enforce_freshness),
        enforced_provenance=bool(enforce_provenance),
        backend=index.metadata.backend,
    )


def search_unchecked(
    index: ExactIndex,
    query: str | QueryEmbedding,
    **kwargs,
) -> SearchResponse:
    """Controlled defect: score a stale or mixed index without the gate.

    The named change is ``enforce_freshness=False``. Provenance stays
    enforced unless the caller also sets ``enforce_provenance=False``.
    """

    kwargs.setdefault("enforce_freshness", False)
    kwargs.setdefault("live_corpus", None)
    return search(index, query, **kwargs)


def search_report(response: SearchResponse) -> dict[str, object]:
    return {
        "query_id": response.query_id,
        "query_text": response.query_text,
        "top_k": response.top_k,
        "filters": dict(response.filters),
        "scored_candidates": response.scored_candidates,
        "latency_ms": round(response.latency_ms, 3),
        "index_id": response.index_id,
        "corpus_version": response.corpus_version,
        "source_hash": response.source_hash,
        "model": response.embedding.model,
        "version": response.embedding.version,
        "metric": response.metric,
        "normalization": response.normalization,
        "backend": response.backend,
        "enforced_freshness": response.enforced_freshness,
        "enforced_provenance": response.enforced_provenance,
        "top_id": response.top_id,
        "hits": tuple(
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "score": hit.score,
                "span": {"start": hit.span_start, "end": hit.span_end},
                "text": hit.text,
                "m28_id": hit.m28_id,
            }
            for hit in response.hits
        ),
    }


def evaluate_labeled(
    response: SearchResponse,
    labeled: LabeledQuery,
    *,
    k: int | None = None,
) -> dict[str, object]:
    """Compare ranked ids to labels. Score is never treated as relevance."""

    depth = int(k if k is not None else response.top_k)
    retrieved = list(response.ids()[:depth])
    relevant = set(labeled.relevant_chunk_ids)
    retrieved_relevant = [chunk_id for chunk_id in retrieved if chunk_id in relevant]
    trap_hits = [chunk_id for chunk_id in retrieved if chunk_id in set(labeled.traps)]
    recall = (len(set(retrieved_relevant)) / len(relevant)) if relevant else 0.0
    precision = (len(retrieved_relevant) / len(retrieved)) if retrieved else 0.0
    return {
        "query_id": labeled.query_id,
        "k": depth,
        "retrieved": tuple(retrieved),
        "relevant": labeled.relevant_chunk_ids,
        "hit_at_k": bool(retrieved_relevant),
        "recall_at_k": recall,
        "precision_at_k": precision,
        "trap_hits": tuple(trap_hits),
        "scores_are_not_labels": True,
        "top_score": response.hits[0].score if response.hits else None,
        "top_id_labeled_relevant": (response.top_id in relevant) if response.top_id else False,
    }


def _corpus_from_payload(payload: Mapping[str, object], embedding: Provenance) -> Corpus:
    documents = []
    for row in payload["documents"]:
        documents.append(
            compose_document(
                str(row["document_id"]),
                str(row["title"]),
                row.get("metadata") or {},
                row["chunks"],
            )
        )
    return Corpus(
        version=str(payload.get("version") or CORPUS_VERSION),
        documents=tuple(documents),
        embedding=embedding,
        authored_for=str(payload.get("authored_for") or "M33"),
        source_mission=str(payload.get("source_mission") or "M28"),
    )


def _vectors_from_payload(payload: Mapping[str, object]) -> dict[str, tuple[float, ...]]:
    width = int(payload["provenance"]["dimensions"])
    out: dict[str, tuple[float, ...]] = {}
    for row in payload["items"]:
        chunk_id = str(row["chunk_id"])
        out[chunk_id] = _vector_tuple(row["vector"], name=chunk_id, dimensions=width)
    return out


def _labeled_from_payload(payload: Mapping[str, object]) -> tuple[LabeledQuery, ...]:
    queries = []
    for row in payload["queries"]:
        queries.append(
            LabeledQuery(
                query_id=str(row["id"]),
                text=str(row["text"]),
                experiment=str(row.get("experiment") or ""),
                relevant_chunk_ids=tuple(str(item) for item in row.get("relevant", ())),
                traps=tuple(str(item) for item in row.get("traps", ())),
                hard_neighbor=str(row["hard_neighbor"]) if row.get("hard_neighbor") else None,
            )
        )
    return tuple(queries)


@lru_cache(maxsize=4)
def load_corpus_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "corpus.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_vectors_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "vectors.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_queries_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "queries.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_expected_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "expected.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_incompatible_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "incompatible_vectors.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_transfer_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "transfer.json"
    return _read_json(target)


@lru_cache(maxsize=2)
def load_canonical_corpus() -> Corpus:
    corpus_payload = load_corpus_payload()
    vectors_payload = load_vectors_payload()
    embedding = provenance_from_mapping(vectors_payload["provenance"])
    return _corpus_from_payload(corpus_payload, embedding)


@lru_cache(maxsize=2)
def load_canonical_vectors() -> dict[str, tuple[float, ...]]:
    return _vectors_from_payload(load_vectors_payload())


@lru_cache(maxsize=2)
def load_canonical_index() -> ExactIndex:
    corpus = load_canonical_corpus()
    return build_index(corpus, vectors=load_canonical_vectors())


def load_labeled_queries() -> tuple[LabeledQuery, ...]:
    return _labeled_from_payload(load_queries_payload())


def load_query_map() -> dict[str, LabeledQuery]:
    return {query.query_id: query for query in load_labeled_queries()}


def load_incompatible_index() -> ExactIndex:
    corpus = load_canonical_corpus()
    payload = load_incompatible_payload()
    embedding = provenance_from_mapping(payload["provenance"])
    mismatched = Corpus(
        version=corpus.version,
        documents=corpus.documents,
        embedding=embedding,
        authored_for=corpus.authored_for,
        source_mission=corpus.source_mission,
    )
    return build_index(mismatched, vectors=_vectors_from_payload(payload), index_id="v08-exact-memory-incompatible")


def load_transfer_bundle() -> tuple[Corpus, ExactIndex, tuple[LabeledQuery, ...], dict]:
    payload = load_transfer_payload()
    embedding = provenance_from_mapping(payload["provenance"])
    corpus = _corpus_from_payload(payload["corpus"], embedding)
    vectors = {
        str(row["chunk_id"]): _vector_tuple(
            row["vector"],
            name=str(row["chunk_id"]),
            dimensions=embedding.dimensions,
        )
        for row in payload["vectors"]
    }
    index = build_index(corpus, vectors=vectors, index_id="v08-transfer-exact")
    queries = _labeled_from_payload(payload)
    return corpus, index, queries, payload


def search_labeled(
    query_id: str,
    *,
    index: ExactIndex | None = None,
    corpus: Corpus | None = None,
    top_k: int = DEFAULT_TOP_K,
    filters: Mapping[str, object] | None = None,
    enforce_freshness: bool = True,
    enforce_provenance: bool = True,
    encoder=None,
) -> SearchResponse:
    labeled = load_query_map()[query_id]
    used_index = index or load_canonical_index()
    used_corpus = corpus or load_canonical_corpus()
    encoded = encode_query(labeled.text, query_id=labeled.query_id, encoder=encoder)
    return search(
        used_index,
        encoded,
        top_k=top_k,
        filters=filters,
        query_id=labeled.query_id,
        live_corpus=used_corpus,
        enforce_freshness=enforce_freshness,
        enforce_provenance=enforce_provenance,
        encoder=encoder,
    )
