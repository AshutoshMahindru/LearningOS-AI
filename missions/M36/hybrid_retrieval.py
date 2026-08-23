"""Vector-store teaching adapter and hybrid retrieval for Mission M36.

M36 is the V09 infrastructure boundary. M35 exact retrieval is the
correctness oracle. A local graph adapter exposes an ef/effort knob
without implementing a production ANN library. Metadata filters, a
BM25-style sparse channel, declared rank fusion, and
insert/update/delete/rebuild complete the teaching store.

Required tests do not use a managed vector database. Optional live
adapters must fail closed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import heapq
import json
import math
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from missions.M28.embedding_core import cosine_similarity, l2_norm, load_encoder  # noqa: E402
from missions.M33.semantic_search import (  # noqa: E402
    FILTER_SCHEMA,
    Chunk,
    Corpus,
    ExactIndex,
    FilterSchemaError,
    IndexRecord,
    QueryEmbedding,
    RankedHit,
    build_index,
    compose_document,
    encode_query,
    exact_cosine_rank,
    load_canonical_corpus,
    load_canonical_index,
    search,
    source_hash,
)
from missions.M35.retrieval_eval import (  # noqa: E402
    EVAL_VERSION,
    FrozenQuery,
    baseline_config,
    evaluate_set,
    generate_candidates,
    label_hash,
    load_frozen_queries,
    load_query_map,
    mean_reciprocal_rank,
    ndcg_at_k,
    questions_sha256,
    recall_at_k,
    round_metric,
    tokenize,
)

PIPELINE_ID = "v09-hybrid-retrieval"
STORE_ID = "v09-teaching-store"
BACKEND_EXACT = "exact-in-memory"
BACKEND_TEACHING_GRAPH = "teaching-graph"
FUSION_RRF = "rrf"
FUSION_RAW_SUM = "raw-sum"
FILTER_PRE = "pre-filter"
FILTER_LATE = "late-filter"
CHANNEL_EXACT = "exact"
CHANNEL_APPROX = "approx"
CHANNEL_SPARSE = "sparse"
CHANNEL_HYBRID = "hybrid"
CHANNEL_RAW_MIX = "raw-mix"
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 5
DEFAULT_EF = 8
LOW_EF = 1
GRAPH_M = 2
GRAPH_LONG_RANGE = 1
RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
SEED = 3601
ENTRY_ID = "doc-weather::c0"
STALE_POLICY = "fail-closed-rebuild-or-reject"
CANONICAL_CORPUS_VERSION = "m33.corpus.v1"
CANONICAL_INDEX_ID = "v08-exact-memory"
FILTER_DEMO_QUERY_ID = "filter-reset"
FILTER_DEMO_QUERY = "Please reset"
FILTER_DEMO_FILTERS = {"topic": "account"}
FILTER_DEMO_RELEVANT = ("doc-account-access::c1",)
FILTER_DEMO_K = 1
LIFECYCLE_INSERT_ID = "doc-tickets::c-m36-insert"
LEXICAL_SLICE = ("rag-ticket-4412", "rag-h-invoice")
SEMANTIC_SLICE = ("rag-password-procedure",)
HYBRID_SLICE = ("rag-ticket-4412", "rag-h-invoice", "rag-password-procedure")

SYSTEM_MAP = (
    "frozen M35 exact eval --oracle--> ExactIndex / generate_candidates\n"
    "same embeddings --teaching graph (M, long-range, ef)--> approx ids + comparisons\n"
    "payload filters --pre-filter vs late-after-small-k--> eligible ids / misses\n"
    "chunk text --BM25 sparse--> lexical ranks (incomparable to cosine)\n"
    "dense ranks + sparse ranks --declared RRF--> hybrid ids\n"
    "insert/update/delete --dirty generation--> rebuild restores freshness"
)

DEFERRED_BOUNDARY = (
    "M36 chooses retrieval infrastructure on the M35 oracle. "
    "It does not open tool/agent orchestration (M37/M38), "
    "temperature/top-p decoding (M32), or M34 extractive generation."
)


def default_dataset_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M36"
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled hybrid-retrieval fixtures: {path}")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@lru_cache(maxsize=4)
def load_expected_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "expected.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_transfer_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "transfer.json"
    return _read_json(target)


class StoreStaleError(ValueError):
    """Raised when a mutated teaching store is searched before rebuild."""

    def __init__(self, *, store_id: str, generation: int, source_hash: str):
        self.store_id = store_id
        self.generation = generation
        self.source_hash = source_hash
        super().__init__(
            f"stale teaching store {store_id} generation {generation}; rebuild before search"
        )


@dataclass(frozen=True)
class InfraConfig:
    experiment_id: str
    backend: str
    top_k: int = DEFAULT_TOP_K
    candidate_k: int = DEFAULT_CANDIDATE_K
    ef: int | None = None
    fusion: str | None = None
    rrf_k: int = RRF_K
    filter_mode: str = FILTER_PRE
    eval_version: str = EVAL_VERSION
    seed: int = SEED

    def __post_init__(self) -> None:
        if int(self.top_k) < 1:
            raise ValueError("top_k must be positive")
        if int(self.candidate_k) < 1:
            raise ValueError("candidate_k must be positive")
        if self.ef is not None and int(self.ef) < 1:
            raise ValueError("ef must be positive when set")
        if self.backend not in (BACKEND_EXACT, BACKEND_TEACHING_GRAPH):
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.filter_mode not in (FILTER_PRE, FILTER_LATE):
            raise ValueError(f"unsupported filter_mode: {self.filter_mode}")
        if self.fusion not in (None, FUSION_RRF):
            raise ValueError(
                "hybrid_search fusion must be declared RRF; "
                "raw-sum is the controlled-failure path"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "backend": self.backend,
            "top_k": int(self.top_k),
            "candidate_k": int(self.candidate_k),
            "ef": self.ef,
            "fusion": self.fusion,
            "rrf_k": int(self.rrf_k),
            "filter_mode": self.filter_mode,
            "eval_version": self.eval_version,
            "seed": self.seed,
            "pipeline_id": PIPELINE_ID,
        }

    def identity(self) -> str:
        return hashlib.sha256(_dump_canonical(self.as_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoreMetadata:
    store_id: str
    backend: str
    corpus_version: str
    source_hash: str
    embedding_model: str
    embedding_version: str
    filter_schema: tuple[str, ...]
    graph_m: int
    graph_long_range: int
    entry_id: str
    generation: int
    dirty: bool
    stale_policy: str
    chunk_count: int
    document_count: int
    downloaded: bool = False
    network_required: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "store_id": self.store_id,
            "backend": self.backend,
            "corpus_version": self.corpus_version,
            "source_hash": self.source_hash,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "filter_schema": list(self.filter_schema),
            "graph_m": self.graph_m,
            "graph_long_range": self.graph_long_range,
            "entry_id": self.entry_id,
            "generation": self.generation,
            "dirty": self.dirty,
            "stale_policy": self.stale_policy,
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "downloaded": self.downloaded,
            "network_required": self.network_required,
        }


@dataclass(frozen=True)
class Adjacency:
    neighbors: tuple[tuple[str, tuple[str, ...]], ...]
    entry_id: str
    degree_m: int
    long_range: int

    def mapping(self) -> dict[str, tuple[str, ...]]:
        return {chunk_id: nbs for chunk_id, nbs in self.neighbors}

    def edge_count(self) -> int:
        return sum(len(nbs) for _, nbs in self.neighbors)


@dataclass(frozen=True)
class SparseVocabulary:
    postings: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    df: tuple[tuple[str, int], ...]
    avgdl: float
    n_docs: int
    k1: float = BM25_K1
    b: float = BM25_B

    def posting_map(self) -> dict[str, tuple[tuple[str, int], ...]]:
        return {term: rows for term, rows in self.postings}

    def df_map(self) -> dict[str, int]:
        return {term: count for term, count in self.df}

    def posting_count(self) -> int:
        return sum(len(rows) for _, rows in self.postings)


@dataclass(frozen=True)
class TeachingStore:
    corpus: Corpus
    exact: ExactIndex
    adjacency: Adjacency
    sparse: SparseVocabulary
    metadata: StoreMetadata

    def records(self) -> tuple[IndexRecord, ...]:
        return self.exact.records

    def get(self, chunk_id: str) -> IndexRecord:
        return self.exact.get(chunk_id)


@dataclass(frozen=True)
class HybridHit:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    text: str
    metadata: tuple[tuple[str, str], ...]
    channel: str
    index_id: str
    source_hash: str
    corpus_version: str
    backend: str
    comparisons: int
    fusion: str | None = None
    ef: int | None = None
    cosine_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    span_start: int = 0
    span_end: int = 0
    m28_id: str = ""
    filter_mode: str | None = None

    def metadata_dict(self) -> dict[str, str]:
        return {key: value for key, value in self.metadata}

    def as_evidence(self) -> dict[str, object]:
        """M40-facing ranked evidence row. No generation or tool calls."""

        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "text": self.text,
            "span": {"start": self.span_start, "end": self.span_end},
            "metadata": self.metadata_dict(),
            "m28_id": self.m28_id,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "corpus_version": self.corpus_version,
            "backend": self.backend,
            "channel": self.channel,
            "fusion": self.fusion,
            "ef": self.ef,
            "comparisons": self.comparisons,
            "cosine_score": self.cosine_score,
            "sparse_score": self.sparse_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "filter_mode": self.filter_mode,
        }


@dataclass(frozen=True)
class ChannelResult:
    channel: str
    query_id: str
    query_text: str
    top_k: int
    hits: tuple[HybridHit, ...]
    scored_candidates: int
    comparisons: int
    latency_ms: float
    filters: tuple[tuple[str, str], ...]
    filter_mode: str
    ef: int | None
    backend: str
    index_id: str
    source_hash: str
    corpus_version: str
    fusion: str | None = None
    all_scores: tuple[tuple[str, float], ...] = ()

    def ids(self) -> tuple[str, ...]:
        return tuple(hit.chunk_id for hit in self.hits)

    def id_set(self) -> frozenset[str]:
        return frozenset(self.ids())

    def rank_map(self) -> dict[str, int]:
        return {hit.chunk_id: hit.rank for hit in self.hits}

    def score_map(self) -> dict[str, float]:
        if self.all_scores:
            return {chunk_id: float(score) for chunk_id, score in self.all_scores}
        return {hit.chunk_id: hit.score for hit in self.hits}


@dataclass(frozen=True)
class NeighborCompare:
    query_id: str
    query_text: str
    top_k: int
    ef: int
    exact_ids: tuple[str, ...]
    approx_ids: tuple[str, ...]
    neighbor_recall: float
    exact_comparisons: int
    approx_comparisons: int
    top1_match: bool
    exact_index_id: str
    approx_backend: str

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "top_k": self.top_k,
            "ef": self.ef,
            "exact_ids": list(self.exact_ids),
            "approx_ids": list(self.approx_ids),
            "neighbor_recall": self.neighbor_recall,
            "exact_comparisons": self.exact_comparisons,
            "approx_comparisons": self.approx_comparisons,
            "top1_match": self.top1_match,
            "exact_index_id": self.exact_index_id,
            "approx_backend": self.approx_backend,
        }


@dataclass(frozen=True)
class FilterTrace:
    query_id: str
    query_text: str
    filters: tuple[tuple[str, str], ...]
    top_k: int
    relevant_ids: tuple[str, ...]
    eligible_ids: tuple[str, ...]
    unfiltered_ids: tuple[str, ...]
    prefilter_ids: tuple[str, ...]
    late_ids: tuple[str, ...]
    late_missed_relevant: tuple[str, ...]
    prefilter_missed_relevant: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "filters": dict(self.filters),
            "top_k": self.top_k,
            "relevant_ids": list(self.relevant_ids),
            "eligible_ids": list(self.eligible_ids),
            "unfiltered_ids": list(self.unfiltered_ids),
            "prefilter_ids": list(self.prefilter_ids),
            "late_ids": list(self.late_ids),
            "late_missed_relevant": list(self.late_missed_relevant),
            "prefilter_missed_relevant": list(self.prefilter_missed_relevant),
        }


@dataclass(frozen=True)
class MemoryProxy:
    n_vectors: int
    dimensions: int
    graph_edges: int
    sparse_terms: int
    sparse_postings: int
    vector_bytes: int
    graph_bytes: int
    sparse_bytes: int
    total_bytes: int
    backend: str

    def as_dict(self) -> dict[str, object]:
        return {
            "n_vectors": self.n_vectors,
            "dimensions": self.dimensions,
            "graph_edges": self.graph_edges,
            "sparse_terms": self.sparse_terms,
            "sparse_postings": self.sparse_postings,
            "vector_bytes": self.vector_bytes,
            "graph_bytes": self.graph_bytes,
            "sparse_bytes": self.sparse_bytes,
            "total_bytes": self.total_bytes,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class LifecycleEvent:
    op: str
    chunk_id: str
    generation: int
    dirty: bool
    source_hash: str
    chunk_count: int


def _validate_filters(filters: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if not filters:
        return ()
    unknown = tuple(sorted(str(key) for key in filters if str(key) not in FILTER_SCHEMA))
    if unknown:
        raise FilterSchemaError(unknown=unknown, schema=FILTER_SCHEMA)
    return tuple(sorted((str(key), str(value)) for key, value in filters.items()))


def matches_metadata(
    metadata: Mapping[str, str] | Sequence[tuple[str, str]],
    filters: Mapping[str, str] | Sequence[tuple[str, str]] | None,
) -> bool:
    if not filters:
        return True
    items = filters.items() if isinstance(filters, Mapping) else filters
    meta = dict(metadata) if not isinstance(metadata, Mapping) else dict(metadata)
    return all(meta.get(str(key)) == str(value) for key, value in items)


def record_matches(record: IndexRecord, filters: Sequence[tuple[str, str]] | None) -> bool:
    return matches_metadata(record.chunk.metadata_dict(), filters)


def _as_query(
    query: str | QueryEmbedding,
    *,
    query_id: str,
) -> QueryEmbedding:
    if isinstance(query, QueryEmbedding):
        return query
    if not isinstance(query, str):
        raise TypeError("query must be text or QueryEmbedding")
    return encode_query(query, query_id=query_id)


def _unit_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = l2_norm(vector)
    if abs(norm - 1.0) <= 1e-6:
        return vector
    if norm <= 1e-12:
        raise ValueError(f"{name} vector has zero norm")
    return tuple(component / norm for component in vector)


def _hit_from_ranked(
    record: IndexRecord,
    *,
    rank: int,
    score: float,
    channel: str,
    store: TeachingStore,
    comparisons: int,
    ef: int | None = None,
    fusion: str | None = None,
    cosine_score: float | None = None,
    sparse_score: float | None = None,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
    filter_mode: str | None = None,
    backend: str | None = None,
) -> HybridHit:
    chunk = record.chunk
    return HybridHit(
        rank=int(rank),
        score=float(score),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        metadata=chunk.metadata,
        channel=channel,
        index_id=store.metadata.store_id if channel != CHANNEL_EXACT else store.exact.metadata.index_id,
        source_hash=store.metadata.source_hash,
        corpus_version=store.metadata.corpus_version,
        backend=backend or store.metadata.backend,
        comparisons=int(comparisons),
        fusion=fusion,
        ef=ef,
        cosine_score=cosine_score,
        sparse_score=sparse_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        span_start=chunk.span_start,
        span_end=chunk.span_end,
        m28_id=chunk.m28_id,
        filter_mode=filter_mode,
    )


def _result_from_hits(
    hits: Sequence[HybridHit],
    *,
    channel: str,
    query_id: str,
    query_text: str,
    top_k: int,
    scored_candidates: int,
    comparisons: int,
    latency_ms: float,
    filters: tuple[tuple[str, str], ...],
    filter_mode: str,
    ef: int | None,
    backend: str,
    store: TeachingStore,
    fusion: str | None = None,
    all_scores: Sequence[tuple[str, float]] = (),
) -> ChannelResult:
    return ChannelResult(
        channel=channel,
        query_id=query_id,
        query_text=query_text,
        top_k=int(top_k),
        hits=tuple(hits),
        scored_candidates=int(scored_candidates),
        comparisons=int(comparisons),
        latency_ms=float(latency_ms),
        filters=filters,
        filter_mode=filter_mode,
        ef=ef,
        backend=backend,
        index_id=store.exact.metadata.index_id if channel == CHANNEL_EXACT else store.metadata.store_id,
        source_hash=store.metadata.source_hash,
        corpus_version=store.metadata.corpus_version,
        fusion=fusion,
        all_scores=tuple(all_scores),
    )


def neighbor_recall(approx_ids: Sequence[str], exact_ids: Sequence[str]) -> float:
    exact = [chunk_id for chunk_id in exact_ids if chunk_id]
    if not exact:
        raise ValueError("exact_ids must be non-empty")
    return len(set(approx_ids).intersection(exact)) / len(exact)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = RRF_K,
) -> tuple[tuple[str, float], ...]:
    """Declared fusion: sum 1/(rrf_k + rank) over lists. Does not add raw scores."""

    if int(rrf_k) < 0:
        raise ValueError("rrf_k must be >= 0")
    scores: dict[str, float] = {}
    for ranked in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranked, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (int(rrf_k) + rank)
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def build_adjacency(
    records: Sequence[IndexRecord],
    *,
    degree_m: int = GRAPH_M,
    long_range: int = GRAPH_LONG_RANGE,
    entry_id: str | None = None,
) -> Adjacency:
    """Local navigable graph: M nearest plus deterministic long-range edges.

    This is a teaching adapter for search-effort (ef), not a production
    hierarchical index.
    """

    if int(degree_m) < 1:
        raise ValueError("degree_m must be positive")
    if int(long_range) < 0:
        raise ValueError("long_range must be >= 0")
    ids = [record.chunk.chunk_id for record in records]
    if not ids:
        raise ValueError("cannot build a graph over an empty record set")
    vecs = {record.chunk.chunk_id: record.vector for record in records}
    graph: dict[str, list[str]] = {}
    for chunk_id in ids:
        knn = sorted(
            (
                (cosine_similarity(vecs[chunk_id], vecs[other]), other)
                for other in ids
                if other != chunk_id
            ),
            key=lambda item: (-item[0], item[1]),
        )
        neighbors = [other for _, other in knn[: int(degree_m)]]
        if int(long_range) and knn:
            farthest = sorted(knn, key=lambda item: (item[0], item[1]))
            added = 0
            for _, other in farthest:
                if other in neighbors:
                    continue
                neighbors.append(other)
                added += 1
                if added >= int(long_range):
                    break
        graph[chunk_id] = neighbors
    for chunk_id, neighbors in list(graph.items()):
        for other in list(neighbors):
            if chunk_id not in graph[other]:
                graph[other].append(chunk_id)
    frozen = tuple((chunk_id, tuple(dict.fromkeys(graph[chunk_id]))) for chunk_id in ids)
    chosen = entry_id or (ENTRY_ID if ENTRY_ID in vecs else ids[0])
    if chosen not in vecs:
        chosen = ids[0]
    return Adjacency(
        neighbors=frozen,
        entry_id=chosen,
        degree_m=int(degree_m),
        long_range=int(long_range),
    )


def build_sparse(
    records: Sequence[IndexRecord],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> SparseVocabulary:
    tokens = {record.chunk.chunk_id: tokenize(record.chunk.text) for record in records}
    n_docs = len(tokens)
    if n_docs == 0:
        raise ValueError("cannot build sparse postings over an empty record set")
    df_counter: Counter[str] = Counter()
    tf_maps: dict[str, Counter[str]] = {}
    lengths: list[int] = []
    for chunk_id, toks in tokens.items():
        counts = Counter(toks)
        tf_maps[chunk_id] = counts
        df_counter.update(set(toks))
        lengths.append(len(toks))
    avgdl = sum(lengths) / n_docs
    terms = tuple(sorted(df_counter))
    postings = []
    for term in terms:
        rows = tuple(
            (chunk_id, int(tf_maps[chunk_id][term]))
            for chunk_id in sorted(tf_maps)
            if tf_maps[chunk_id][term]
        )
        postings.append((term, rows))
    df_rows = tuple((term, int(df_counter[term])) for term in terms)
    return SparseVocabulary(
        postings=tuple(postings),
        df=df_rows,
        avgdl=float(avgdl),
        n_docs=n_docs,
        k1=float(k1),
        b=float(b),
    )


def memory_proxy(store: TeachingStore) -> MemoryProxy:
    dimensions = int(store.corpus.embedding.dimensions)
    n_vectors = len(store.records())
    graph_edges = store.adjacency.edge_count()
    sparse_terms = len(store.sparse.postings)
    sparse_postings = store.sparse.posting_count()
    vector_bytes = n_vectors * dimensions * 8
    graph_bytes = graph_edges * 8
    sparse_bytes = sparse_postings * 8
    return MemoryProxy(
        n_vectors=n_vectors,
        dimensions=dimensions,
        graph_edges=graph_edges,
        sparse_terms=sparse_terms,
        sparse_postings=sparse_postings,
        vector_bytes=vector_bytes,
        graph_bytes=graph_bytes,
        sparse_bytes=sparse_bytes,
        total_bytes=vector_bytes + graph_bytes + sparse_bytes,
        backend=store.metadata.backend,
    )


def _store_metadata(
    corpus: Corpus,
    exact: ExactIndex,
    adjacency: Adjacency,
    *,
    generation: int,
    dirty: bool,
    store_id: str = STORE_ID,
) -> StoreMetadata:
    return StoreMetadata(
        store_id=store_id,
        backend=BACKEND_TEACHING_GRAPH,
        corpus_version=corpus.version,
        source_hash=source_hash(corpus),
        embedding_model=exact.metadata.embedding.model,
        embedding_version=exact.metadata.embedding.version,
        filter_schema=FILTER_SCHEMA,
        graph_m=adjacency.degree_m,
        graph_long_range=adjacency.long_range,
        entry_id=adjacency.entry_id,
        generation=int(generation),
        dirty=bool(dirty),
        stale_policy=STALE_POLICY,
        chunk_count=len(exact.records),
        document_count=len(corpus.documents),
        downloaded=False,
        network_required=False,
    )


def open_teaching_store(
    *,
    corpus: Corpus | None = None,
    index: ExactIndex | None = None,
    degree_m: int = GRAPH_M,
    long_range: int = GRAPH_LONG_RANGE,
    entry_id: str | None = None,
) -> TeachingStore:
    used_corpus = corpus or load_canonical_corpus()
    used_index = index or load_canonical_index()
    if source_hash(used_corpus) != used_index.metadata.source_hash:
        raise ValueError("corpus source_hash does not match the exact index")
    adjacency = build_adjacency(
        used_index.records,
        degree_m=degree_m,
        long_range=long_range,
        entry_id=entry_id,
    )
    sparse = build_sparse(used_index.records)
    metadata = _store_metadata(
        used_corpus,
        used_index,
        adjacency,
        generation=0,
        dirty=False,
    )
    return TeachingStore(
        corpus=used_corpus,
        exact=used_index,
        adjacency=adjacency,
        sparse=sparse,
        metadata=metadata,
    )


def assert_store_fresh(store: TeachingStore) -> None:
    if store.metadata.dirty:
        raise StoreStaleError(
            store_id=store.metadata.store_id,
            generation=store.metadata.generation,
            source_hash=store.metadata.source_hash,
        )
    live = source_hash(store.corpus)
    if live != store.metadata.source_hash:
        raise StoreStaleError(
            store_id=store.metadata.store_id,
            generation=store.metadata.generation,
            source_hash=store.metadata.source_hash,
        )


def m35_baseline_report(*, queries: Sequence[FrozenQuery] | None = None):
    """M35 exact identity rerank on frozen labels. Correctness reference."""

    frozen = tuple(queries or load_frozen_queries())
    return evaluate_set(config=baseline_config(), queries=frozen)


def exact_search(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    filters: Mapping[str, object] | None = None,
    enforce_freshness: bool = True,
) -> ChannelResult:
    """Exact cosine over eligible rows. Oracle path. Filters run before scoring."""

    if enforce_freshness:
        assert_store_fresh(store)
    started = time.perf_counter()
    encoded = _as_query(query, query_id=query_id)
    filter_pairs = _validate_filters(filters)
    response = search(
        store.exact,
        encoded,
        top_k=int(top_k),
        filters=dict(filter_pairs) if filter_pairs else None,
        query_id=query_id,
        live_corpus=store.corpus,
        enforce_freshness=enforce_freshness,
        enforce_provenance=True,
    )
    ranked = exact_cosine_rank(
        encoded.vector,
        store.exact.records,
        top_k=len(store.exact.records) or 1,
        filters=filter_pairs,
        metric=store.exact.metadata.embedding.metric,
        normalization=store.exact.metadata.embedding.normalization,
    )
    record_map = {record.chunk.chunk_id: record for record, _score in ranked}
    hits = tuple(
        _hit_from_ranked(
            store.exact.get(hit.chunk_id),
            rank=hit.rank,
            score=hit.score,
            channel=CHANNEL_EXACT,
            store=store,
            comparisons=int(response.scored_candidates),
            cosine_score=hit.score,
            filter_mode=FILTER_PRE,
            backend=BACKEND_EXACT,
        )
        for hit in response.hits
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    all_scores = tuple((record.chunk.chunk_id, float(score)) for record, score in ranked)
    del record_map
    return _result_from_hits(
        hits,
        channel=CHANNEL_EXACT,
        query_id=query_id,
        query_text=encoded.text,
        top_k=top_k,
        scored_candidates=int(response.scored_candidates),
        comparisons=int(response.scored_candidates),
        latency_ms=latency_ms,
        filters=filter_pairs,
        filter_mode=FILTER_PRE,
        ef=None,
        backend=BACKEND_EXACT,
        store=store,
        all_scores=all_scores,
    )


def _search_graph(
    query_vector: Sequence[float],
    store: TeachingStore,
    *,
    ef: int,
    eligible: set[str] | None,
) -> tuple[list[tuple[float, str]], int]:
    mapping = store.adjacency.mapping()
    vectors = {record.chunk.chunk_id: record.vector for record in store.records()}
    if eligible is not None:
        mapping = {
            chunk_id: tuple(nb for nb in neighbors if nb in eligible)
            for chunk_id, neighbors in mapping.items()
            if chunk_id in eligible
        }
        if not mapping:
            return [], 0
        entry = store.adjacency.entry_id if store.adjacency.entry_id in mapping else sorted(mapping)[0]
    else:
        entry = store.adjacency.entry_id if store.adjacency.entry_id in mapping else next(iter(mapping))
    score = float(cosine_similarity(query_vector, vectors[entry]))
    comparisons = 1
    visited = {entry: score}
    candidates = [(-score, entry)]
    beam = [(score, entry)]
    while candidates:
        best_score = -candidates[0][0]
        if best_score < beam[0][0]:
            break
        _ignored, current = heapq.heappop(candidates)
        del _ignored
        for neighbor in mapping.get(current, ()):
            if neighbor in visited:
                continue
            nscore = float(cosine_similarity(query_vector, vectors[neighbor]))
            comparisons += 1
            visited[neighbor] = nscore
            if nscore > beam[0][0] or len(beam) < int(ef):
                heapq.heappush(candidates, (-nscore, neighbor))
                heapq.heappush(beam, (nscore, neighbor))
                if len(beam) > int(ef):
                    heapq.heappop(beam)
    ranked = sorted(beam, key=lambda item: (-item[0], item[1]))
    return ranked, comparisons


def approximate_search(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    ef: int = DEFAULT_EF,
    filters: Mapping[str, object] | None = None,
    filter_mode: str = FILTER_PRE,
    enforce_freshness: bool = True,
) -> ChannelResult:
    """Greedy graph search with an ef effort knob. Same embeddings as exact."""

    if int(ef) < 1:
        raise ValueError("ef must be positive")
    if filter_mode not in (FILTER_PRE, FILTER_LATE):
        raise ValueError(f"unsupported filter_mode: {filter_mode}")
    if enforce_freshness:
        assert_store_fresh(store)
    started = time.perf_counter()
    encoded = _as_query(query, query_id=query_id)
    filter_pairs = _validate_filters(filters)
    eligible = None
    if filter_mode == FILTER_PRE and filter_pairs:
        eligible = {
            record.chunk.chunk_id
            for record in store.records()
            if record_matches(record, filter_pairs)
        }
    beam, comparisons = _search_graph(
        encoded.vector,
        store,
        ef=int(ef),
        eligible=eligible,
    )
    selected = beam[: int(top_k)]
    if filter_mode == FILTER_LATE and filter_pairs:
        selected = [
            (score, chunk_id)
            for score, chunk_id in selected
            if record_matches(store.get(chunk_id), filter_pairs)
        ]
    hits = tuple(
        _hit_from_ranked(
            store.get(chunk_id),
            rank=offset,
            score=score,
            channel=CHANNEL_APPROX,
            store=store,
            comparisons=comparisons,
            ef=int(ef),
            cosine_score=score,
            filter_mode=filter_mode,
            backend=BACKEND_TEACHING_GRAPH,
        )
        for offset, (score, chunk_id) in enumerate(selected, start=1)
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return _result_from_hits(
        hits,
        channel=CHANNEL_APPROX,
        query_id=query_id,
        query_text=encoded.text,
        top_k=top_k,
        scored_candidates=comparisons,
        comparisons=comparisons,
        latency_ms=latency_ms,
        filters=filter_pairs,
        filter_mode=filter_mode,
        ef=int(ef),
        backend=BACKEND_TEACHING_GRAPH,
        store=store,
        all_scores=tuple((chunk_id, score) for score, chunk_id in beam),
    )


def bm25_score_map(
    store: TeachingStore,
    query_text: str,
    *,
    filters: Sequence[tuple[str, str]] | None = None,
) -> tuple[tuple[str, float], ...]:
    query_terms = tokenize(query_text)
    df_map = store.sparse.df_map()
    postings = store.sparse.posting_map()
    n_docs = store.sparse.n_docs
    avgdl = store.sparse.avgdl
    k1 = store.sparse.k1
    b = store.sparse.b
    lengths = {record.chunk.chunk_id: len(tokenize(record.chunk.text)) for record in store.records()}
    scores = {record.chunk.chunk_id: 0.0 for record in store.records()}
    for term in query_terms:
        df = df_map.get(term)
        if not df:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        for chunk_id, tf in postings.get(term, ()):
            if filters and not record_matches(store.get(chunk_id), filters):
                continue
            dl = max(lengths.get(chunk_id, 0), 1)
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (k1 + 1.0)) / denom
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if filters:
        ranked = [
            (chunk_id, score)
            for chunk_id, score in ranked
            if record_matches(store.get(chunk_id), filters)
        ]
    return tuple(ranked)


def sparse_search(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    filters: Mapping[str, object] | None = None,
    enforce_freshness: bool = True,
) -> ChannelResult:
    """Lexical BM25 over chunk text. Scores are not on the cosine scale."""

    if enforce_freshness:
        assert_store_fresh(store)
    started = time.perf_counter()
    encoded = _as_query(query, query_id=query_id)
    filter_pairs = _validate_filters(filters)
    ranked = bm25_score_map(store, encoded.text, filters=filter_pairs)
    hits = tuple(
        _hit_from_ranked(
            store.get(chunk_id),
            rank=offset,
            score=score,
            channel=CHANNEL_SPARSE,
            store=store,
            comparisons=len(ranked),
            sparse_score=score,
            filter_mode=FILTER_PRE,
            backend=BACKEND_TEACHING_GRAPH,
        )
        for offset, (chunk_id, score) in enumerate(ranked[: int(top_k)], start=1)
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return _result_from_hits(
        hits,
        channel=CHANNEL_SPARSE,
        query_id=query_id,
        query_text=encoded.text,
        top_k=top_k,
        scored_candidates=len(ranked),
        comparisons=len(ranked),
        latency_ms=latency_ms,
        filters=filter_pairs,
        filter_mode=FILTER_PRE,
        ef=None,
        backend=BACKEND_TEACHING_GRAPH,
        store=store,
        all_scores=ranked,
    )


def compare_to_exact(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    ef: int = DEFAULT_EF,
) -> NeighborCompare:
    exact = exact_search(store, query, query_id=query_id, top_k=top_k)
    approx = approximate_search(store, query, query_id=query_id, top_k=top_k, ef=ef)
    exact_ids = exact.ids()
    approx_ids = approx.ids()
    return NeighborCompare(
        query_id=query_id,
        query_text=exact.query_text,
        top_k=int(top_k),
        ef=int(ef),
        exact_ids=exact_ids,
        approx_ids=approx_ids,
        neighbor_recall=neighbor_recall(approx_ids, exact_ids) if exact_ids else 0.0,
        exact_comparisons=exact.comparisons,
        approx_comparisons=approx.comparisons,
        top1_match=bool(exact_ids) and bool(approx_ids) and exact_ids[0] == approx_ids[0],
        exact_index_id=exact.index_id,
        approx_backend=approx.backend,
    )


def late_filter_hits(
    hits: Sequence[HybridHit],
    filters: Mapping[str, object] | Sequence[tuple[str, str]] | None,
) -> tuple[HybridHit, ...]:
    if not filters:
        return tuple(hits)
    pairs = _validate_filters(dict(filters) if isinstance(filters, Mapping) else dict(filters))
    kept = [hit for hit in hits if matches_metadata(hit.metadata_dict(), pairs)]
    return tuple(
        HybridHit(
            rank=offset,
            score=hit.score,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            text=hit.text,
            metadata=hit.metadata,
            channel=hit.channel,
            index_id=hit.index_id,
            source_hash=hit.source_hash,
            corpus_version=hit.corpus_version,
            backend=hit.backend,
            comparisons=hit.comparisons,
            fusion=hit.fusion,
            ef=hit.ef,
            cosine_score=hit.cosine_score,
            sparse_score=hit.sparse_score,
            dense_rank=hit.dense_rank,
            sparse_rank=hit.sparse_rank,
            span_start=hit.span_start,
            span_end=hit.span_end,
            m28_id=hit.m28_id,
            filter_mode=FILTER_LATE,
        )
        for offset, hit in enumerate(kept, start=1)
    )


def filter_placement_trace(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str,
    filters: Mapping[str, object],
    relevant_ids: Sequence[str],
    top_k: int = FILTER_DEMO_K,
) -> FilterTrace:
    """Compare proper pre-filter with unsafe filter-after-small-top-k."""

    encoded = _as_query(query, query_id=query_id)
    filter_pairs = _validate_filters(filters)
    eligible = tuple(
        record.chunk.chunk_id
        for record in store.records()
        if record_matches(record, filter_pairs)
    )
    unfiltered = exact_search(store, encoded, query_id=query_id, top_k=top_k, filters=None)
    prefilter = exact_search(store, encoded, query_id=query_id, top_k=top_k, filters=filters)
    late_ids = tuple(hit.chunk_id for hit in late_filter_hits(unfiltered.hits, filters))
    relevant = tuple(str(item) for item in relevant_ids)
    eligible_relevant = tuple(chunk_id for chunk_id in relevant if chunk_id in set(eligible))
    return FilterTrace(
        query_id=query_id,
        query_text=encoded.text,
        filters=filter_pairs,
        top_k=int(top_k),
        relevant_ids=relevant,
        eligible_ids=eligible,
        unfiltered_ids=unfiltered.ids(),
        prefilter_ids=prefilter.ids(),
        late_ids=late_ids,
        late_missed_relevant=tuple(
            chunk_id for chunk_id in eligible_relevant if chunk_id not in set(late_ids)
        ),
        prefilter_missed_relevant=tuple(
            chunk_id for chunk_id in eligible_relevant if chunk_id not in set(prefilter.ids())
        ),
    )


def fuse_channels(
    dense: ChannelResult,
    sparse: ChannelResult,
    *,
    method: str = FUSION_RRF,
    rrf_k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
    store: TeachingStore | None = None,
) -> ChannelResult:
    """Fuse two ranked lists with a declared method. RRF ignores raw scores."""

    if method != FUSION_RRF:
        raise ValueError("fuse_channels only accepts declared RRF; use mix_raw_scores for the broken path")
    fused = reciprocal_rank_fusion((dense.ids(), sparse.ids()), rrf_k=rrf_k)[: int(top_k)]
    used_store = store
    dense_ranks = dense.rank_map()
    sparse_ranks = sparse.rank_map()
    dense_scores = dense.score_map()
    sparse_scores = sparse.score_map()
    hits = []
    for offset, (chunk_id, score) in enumerate(fused, start=1):
        record = used_store.get(chunk_id) if used_store is not None else None
        if record is None:
            source = next((hit for hit in (*dense.hits, *sparse.hits) if hit.chunk_id == chunk_id), None)
            if source is None:
                raise KeyError(chunk_id)
            hit = HybridHit(
                rank=offset,
                score=float(score),
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                text=source.text,
                metadata=source.metadata,
                channel=CHANNEL_HYBRID,
                index_id=dense.index_id,
                source_hash=dense.source_hash,
                corpus_version=dense.corpus_version,
                backend=dense.backend,
                comparisons=dense.comparisons + sparse.comparisons,
                fusion=FUSION_RRF,
                cosine_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                span_start=source.span_start,
                span_end=source.span_end,
                m28_id=source.m28_id,
            )
        else:
            hit = _hit_from_ranked(
                record,
                rank=offset,
                score=float(score),
                channel=CHANNEL_HYBRID,
                store=used_store,
                comparisons=dense.comparisons + sparse.comparisons,
                fusion=FUSION_RRF,
                cosine_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                backend=BACKEND_TEACHING_GRAPH,
            )
        hits.append(hit)
    index_id = used_store.metadata.store_id if used_store is not None else dense.index_id
    return ChannelResult(
        channel=CHANNEL_HYBRID,
        query_id=dense.query_id,
        query_text=dense.query_text,
        top_k=int(top_k),
        hits=tuple(hits),
        scored_candidates=len(set(dense.ids()) | set(sparse.ids())),
        comparisons=dense.comparisons + sparse.comparisons,
        latency_ms=dense.latency_ms + sparse.latency_ms,
        filters=dense.filters,
        filter_mode=dense.filter_mode,
        ef=dense.ef,
        backend=BACKEND_TEACHING_GRAPH if used_store is not None else dense.backend,
        index_id=index_id,
        source_hash=dense.source_hash,
        corpus_version=dense.corpus_version,
        fusion=FUSION_RRF,
        all_scores=fused,
    )


def mix_raw_scores(
    dense: ChannelResult,
    sparse: ChannelResult,
    *,
    top_k: int = DEFAULT_TOP_K,
    store: TeachingStore | None = None,
) -> ChannelResult:
    """Broken path: add cosine and BM25 as if they shared a scale."""

    dense_scores = dense.score_map()
    sparse_scores = sparse.score_map()
    union = tuple(dict.fromkeys((*dense_scores, *sparse_scores)))
    mixed = []
    for chunk_id in union:
        mixed.append((float(dense_scores.get(chunk_id, 0.0)) + float(sparse_scores.get(chunk_id, 0.0)), chunk_id))
    mixed.sort(key=lambda item: (-item[0], item[1]))
    selected = mixed[: int(top_k)]
    hits = []
    for offset, (score, chunk_id) in enumerate(selected, start=1):
        source = next((hit for hit in (*dense.hits, *sparse.hits) if hit.chunk_id == chunk_id), None)
        if store is not None:
            record = store.get(chunk_id)
            hits.append(
                _hit_from_ranked(
                    record,
                    rank=offset,
                    score=score,
                    channel=CHANNEL_RAW_MIX,
                    store=store,
                    comparisons=dense.comparisons + sparse.comparisons,
                    fusion=FUSION_RAW_SUM,
                    cosine_score=dense_scores.get(chunk_id),
                    sparse_score=sparse_scores.get(chunk_id),
                    backend=BACKEND_TEACHING_GRAPH,
                )
            )
            continue
        if source is None:
            raise KeyError(chunk_id)
        hits.append(
            HybridHit(
                rank=offset,
                score=score,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                text=source.text,
                metadata=source.metadata,
                channel=CHANNEL_RAW_MIX,
                index_id=store.metadata.store_id if store is not None else dense.index_id,
                source_hash=dense.source_hash,
                corpus_version=dense.corpus_version,
                backend=dense.backend,
                comparisons=dense.comparisons + sparse.comparisons,
                fusion=FUSION_RAW_SUM,
                cosine_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                span_start=source.span_start,
                span_end=source.span_end,
                m28_id=source.m28_id,
            )
        )
    return ChannelResult(
        channel=CHANNEL_RAW_MIX,
        query_id=dense.query_id,
        query_text=dense.query_text,
        top_k=int(top_k),
        hits=tuple(hits),
        scored_candidates=len(union),
        comparisons=dense.comparisons + sparse.comparisons,
        latency_ms=dense.latency_ms + sparse.latency_ms,
        filters=dense.filters,
        filter_mode=dense.filter_mode,
        ef=dense.ef,
        backend=BACKEND_TEACHING_GRAPH if store is not None else dense.backend,
        index_id=store.metadata.store_id if store is not None else dense.index_id,
        source_hash=dense.source_hash,
        corpus_version=dense.corpus_version,
        fusion=FUSION_RAW_SUM,
        all_scores=tuple((chunk_id, score) for score, chunk_id in mixed),
    )


def hybrid_search(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    fusion: str = FUSION_RRF,
    rrf_k: int = RRF_K,
    dense_backend: str = BACKEND_EXACT,
    ef: int = DEFAULT_EF,
    filters: Mapping[str, object] | None = None,
    enforce_freshness: bool = True,
) -> ChannelResult:
    if fusion != FUSION_RRF:
        raise ValueError("hybrid_search fusion must be rrf; raw-sum is mix_raw_scores")
    if dense_backend == BACKEND_EXACT:
        dense = exact_search(
            store,
            query,
            query_id=query_id,
            top_k=candidate_k,
            filters=filters,
            enforce_freshness=enforce_freshness,
        )
    elif dense_backend == BACKEND_TEACHING_GRAPH:
        dense = approximate_search(
            store,
            query,
            query_id=query_id,
            top_k=candidate_k,
            ef=ef,
            filters=filters,
            filter_mode=FILTER_PRE,
            enforce_freshness=enforce_freshness,
        )
    else:
        raise ValueError(f"unsupported dense_backend: {dense_backend}")
    sparse = sparse_search(
        store,
        query,
        query_id=query_id,
        top_k=candidate_k,
        filters=filters,
        enforce_freshness=enforce_freshness,
    )
    return fuse_channels(dense, sparse, method=FUSION_RRF, rrf_k=rrf_k, top_k=top_k, store=store)


def repair_fusion(
    *,
    broken: ChannelResult,
    dense: ChannelResult,
    sparse: ChannelResult,
    store: TeachingStore,
    top_k: int = DEFAULT_TOP_K,
) -> ChannelResult:
    """Rebuild RRF from the same channel objects. Broken mix stays mixed."""

    if broken.fusion != FUSION_RAW_SUM:
        raise ValueError("repair_fusion expects the raw-sum broken object")
    return fuse_channels(dense, sparse, method=FUSION_RRF, top_k=top_k, store=store)


def repair_filter_placement(
    store: TeachingStore,
    query: str | QueryEmbedding,
    *,
    query_id: str,
    filters: Mapping[str, object],
    relevant_ids: Sequence[str],
    top_k: int = FILTER_DEMO_K,
    broken: FilterTrace | None = None,
) -> FilterTrace:
    """Return the pre-filter trace. The late-filter object is not mutated."""

    del broken
    return filter_placement_trace(
        store,
        query,
        query_id=query_id,
        filters=filters,
        relevant_ids=relevant_ids,
        top_k=top_k,
    )


def _replace_document_chunks(
    document,
    *,
    updater,
) -> object:
    parts = []
    changed = False
    for chunk in document.chunks:
        replacement = updater(chunk)
        if replacement is None:
            changed = True
            continue
        if replacement is not chunk:
            changed = True
        parts.append(
            {
                "chunk_id": replacement.chunk_id,
                "text": replacement.text,
                "m28_id": replacement.m28_id,
                "metadata": replacement.metadata_dict(),
                "local_id": replacement.chunk_id.split("::", 1)[-1],
            }
        )
    if not changed:
        return document
    if not parts:
        return None
    return compose_document(
        document.document_id,
        document.title,
        document.metadata_dict(),
        parts,
    )


def _corpus_from_documents(source: Corpus, documents: Sequence, *, version: str | None = None) -> Corpus:
    return Corpus(
        version=version or source.version,
        documents=tuple(documents),
        embedding=source.embedding,
        authored_for=source.authored_for,
        source_mission=source.source_mission,
    )


def _dirty_store(store: TeachingStore, corpus: Corpus, index: ExactIndex, *, op_generation: int) -> TeachingStore:
    metadata = _store_metadata(
        corpus,
        index,
        store.adjacency,
        generation=op_generation,
        dirty=True,
        store_id=store.metadata.store_id,
    )
    return TeachingStore(
        corpus=corpus,
        exact=index,
        adjacency=store.adjacency,
        sparse=store.sparse,
        metadata=metadata,
    )


def insert_chunk(
    store: TeachingStore,
    *,
    document_id: str,
    chunk_id: str,
    text: str,
    metadata: Mapping[str, object],
    title: str | None = None,
    encoder=None,
) -> tuple[TeachingStore, LifecycleEvent]:
    if any(record.chunk.chunk_id == chunk_id for record in store.records()):
        raise ValueError(f"duplicate chunk id: {chunk_id}")
    encoder = encoder or load_encoder()
    vector = _unit_vector(encoder.encode(text), name=chunk_id)
    existing_ids = {document.document_id for document in store.corpus.documents}
    if document_id in existing_ids:
        documents = []
        for document in store.corpus.documents:
            if document.document_id != document_id:
                documents.append(document)
                continue
            parts = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "m28_id": chunk.m28_id,
                    "metadata": chunk.metadata_dict(),
                    "local_id": chunk.chunk_id.split("::", 1)[-1],
                }
                for chunk in document.chunks
            ]
            parts.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "m28_id": "",
                    "metadata": dict(metadata),
                    "local_id": chunk_id.split("::", 1)[-1],
                }
            )
            documents.append(
                compose_document(document.document_id, document.title, document.metadata_dict(), parts)
            )
        corpus = _corpus_from_documents(store.corpus, documents)
    else:
        document = compose_document(
            document_id,
            title or document_id,
            metadata,
            (
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "m28_id": "",
                    "metadata": dict(metadata),
                    "local_id": chunk_id.split("::", 1)[-1],
                },
            ),
        )
        corpus = _corpus_from_documents(
            store.corpus,
            store.corpus.documents + (document,),
        )
    new_chunk = corpus.get_chunk(chunk_id)
    records = store.exact.records + (IndexRecord(chunk=new_chunk, vector=vector),)
    index = ExactIndex(records=records, metadata=store.exact.metadata)
    generation = store.metadata.generation + 1
    new_store = _dirty_store(store, corpus, index, op_generation=generation)
    event = LifecycleEvent(
        op="insert",
        chunk_id=chunk_id,
        generation=generation,
        dirty=True,
        source_hash=new_store.metadata.source_hash,
        chunk_count=len(records),
    )
    return new_store, event


def update_chunk_text(
    store: TeachingStore,
    chunk_id: str,
    new_text: str,
    *,
    encoder=None,
) -> tuple[TeachingStore, LifecycleEvent]:
    found = False

    def updater(chunk: Chunk) -> Chunk | None:
        nonlocal found
        if chunk.chunk_id != chunk_id:
            return chunk
        found = True
        return Chunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=new_text,
            span_start=chunk.span_start,
            span_end=chunk.span_start + len(new_text),
            m28_id=chunk.m28_id,
            metadata=chunk.metadata,
        )

    documents = []
    for document in store.corpus.documents:
        replaced = _replace_document_chunks(document, updater=updater)
        if replaced is None:
            continue
        documents.append(replaced)
    if not found:
        raise KeyError(chunk_id)
    corpus = _corpus_from_documents(store.corpus, documents)
    encoder = encoder or load_encoder()
    vector = _unit_vector(encoder.encode(new_text), name=chunk_id)
    records = []
    for record in store.exact.records:
        if record.chunk.chunk_id != chunk_id:
            records.append(record)
            continue
        records.append(IndexRecord(chunk=corpus.get_chunk(chunk_id), vector=vector))
    index = ExactIndex(records=tuple(records), metadata=store.exact.metadata)
    generation = store.metadata.generation + 1
    new_store = _dirty_store(store, corpus, index, op_generation=generation)
    event = LifecycleEvent(
        op="update",
        chunk_id=chunk_id,
        generation=generation,
        dirty=True,
        source_hash=new_store.metadata.source_hash,
        chunk_count=len(records),
    )
    return new_store, event


def delete_chunk(store: TeachingStore, chunk_id: str) -> tuple[TeachingStore, LifecycleEvent]:
    found = False

    def updater(chunk: Chunk) -> Chunk | None:
        nonlocal found
        if chunk.chunk_id != chunk_id:
            return chunk
        found = True
        return None

    documents = []
    for document in store.corpus.documents:
        replaced = _replace_document_chunks(document, updater=updater)
        if replaced is None:
            continue
        documents.append(replaced)
    if not found:
        raise KeyError(chunk_id)
    corpus = _corpus_from_documents(store.corpus, documents)
    records = tuple(record for record in store.exact.records if record.chunk.chunk_id != chunk_id)
    index = ExactIndex(records=records, metadata=store.exact.metadata)
    generation = store.metadata.generation + 1
    new_store = _dirty_store(store, corpus, index, op_generation=generation)
    event = LifecycleEvent(
        op="delete",
        chunk_id=chunk_id,
        generation=generation,
        dirty=True,
        source_hash=new_store.metadata.source_hash,
        chunk_count=len(records),
    )
    return new_store, event


def rebuild_store(
    store: TeachingStore,
    *,
    degree_m: int | None = None,
    long_range: int | None = None,
) -> tuple[TeachingStore, LifecycleEvent]:
    """Recompute exact identity, graph, and sparse postings from live texts."""

    index = build_index(store.corpus, index_id=store.exact.metadata.index_id)
    adjacency = build_adjacency(
        index.records,
        degree_m=degree_m if degree_m is not None else store.adjacency.degree_m,
        long_range=long_range if long_range is not None else store.adjacency.long_range,
        entry_id=store.adjacency.entry_id if store.adjacency.entry_id in {r.chunk.chunk_id for r in index.records} else None,
    )
    sparse = build_sparse(index.records)
    generation = store.metadata.generation + 1
    metadata = _store_metadata(
        store.corpus,
        index,
        adjacency,
        generation=generation,
        dirty=False,
        store_id=store.metadata.store_id,
    )
    rebuilt = TeachingStore(
        corpus=store.corpus,
        exact=index,
        adjacency=adjacency,
        sparse=sparse,
        metadata=metadata,
    )
    event = LifecycleEvent(
        op="rebuild",
        chunk_id="*",
        generation=generation,
        dirty=False,
        source_hash=rebuilt.metadata.source_hash,
        chunk_count=len(index.records),
    )
    return rebuilt, event


def slice_channel_rows(
    store: TeachingStore,
    queries: Sequence[FrozenQuery],
    *,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> tuple[dict[str, dict[str, object]], ...]:
    rows = []
    for query in queries:
        if not query.support_chunk_ids:
            continue
        dense = exact_search(store, query.text, query_id=query.query_id, top_k=top_k)
        sparse = sparse_search(store, query.text, query_id=query.query_id, top_k=top_k)
        hybrid = hybrid_search(
            store,
            query.text,
            query_id=query.query_id,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        gold = query.support_chunk_ids[0]
        rows.append(
            {
                "query_id": query.query_id,
                "support_id": gold,
                "dense_ids": dense.ids(),
                "sparse_ids": sparse.ids(),
                "hybrid_ids": hybrid.ids(),
                "dense_support_rank": dense.rank_map().get(gold),
                "sparse_support_rank": sparse.rank_map().get(gold),
                "hybrid_support_rank": hybrid.rank_map().get(gold),
                "dense_top": dense.ids()[0] if dense.ids() else None,
                "sparse_top": sparse.ids()[0] if sparse.ids() else None,
                "hybrid_top": hybrid.ids()[0] if hybrid.ids() else None,
            }
        )
    return tuple(rows)


def evidence_from_ranked_hit(hit: RankedHit) -> dict[str, object]:
    return hit.as_evidence()
