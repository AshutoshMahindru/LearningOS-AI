"""Deterministic extractive RAG pipeline for Mission M34.

M34 is the V09 grounding boundary: query -> M33 retrieval -> budgeted
context pack -> local extractive synthesis -> citations -> support
check / abstention. Reranking, chunk retuning, ANN/hybrid indexes,
and decoding labs remain deferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from missions.M33.semantic_search import (  # noqa: E402
    Corpus,
    ExactIndex,
    RankedHit,
    SearchResponse,
    encode_query,
    load_canonical_corpus,
    load_canonical_index,
    search,
)

SEPARATOR = "\n\n"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.ASCII)
PROCEDURE_RE = re.compile(
    r"\b(how do i|what should|how can|what does|why does|which policy|which sentence)\b",
    re.IGNORECASE,
)
ACTION_RE = re.compile(
    r"^(please reset|approve |do not |the service agreement forbids|"
    r"please process|rain is|ticket |the customer paid|the printer needs)",
    re.IGNORECASE,
)

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

PIPELINE_ID = "v09-rag-extractive"
SYNTHESIZER_ID = "extractive-span-v1"
SYNTHESIZER_FAMILY = "local-extractive"
EVAL_VERSION = "m34.eval.v1"
CORPUS_VERSION = "m33.corpus.v1"
INDEX_ID = "v08-exact-memory"
SEED = 3401
DEFAULT_TOP_K = 3
DEFAULT_BUDGET_CHARS = 400
DEFAULT_BUDGET_CHUNKS = 8
MIN_OVERLAP_TOKENS = 2
POLICY_GATED = "support_gated"
POLICY_NAIVE = "naive_top1"
DEFECT_NONE = "none"
DEFECT_UNSUPPORTED = "unsupported_citation"
DEFECT_INVENTED = "invented_support"
DEFECT_CLOSED = "closed_book_guess"
SUPPORTED_POLICIES = (POLICY_GATED, POLICY_NAIVE)
SUPPORTED_DEFECTS = (DEFECT_NONE, DEFECT_UNSUPPORTED, DEFECT_INVENTED, DEFECT_CLOSED)
FAILURE_LAYERS = ("retrieval", "context", "generation", "citation")
ABSTAIN_TEXT = "Insufficient evidence in the packed context to support an answer."

INFERENCE_BOUNDARY = (
    "M34 runs at inference time. The extractive synthesizer does not update "
    "weights. Retrieval changes the context pack, not the training objective. "
    "Token sampling is not opened here."
)

SYSTEM_MAP = (
    "query --normalize--> query text\n"
    "query text --M33 encode/search--> ranked evidence (ids, scores, spans, provenance)\n"
    "ranked evidence --pack(budget)--> context pack\n"
    "context pack --extractive synthesizer--> claim + citations\n"
    "claim + citations --verify_support--> answered | abstain\n"
    "trace retains retrieval, pack, inference config, and support"
)


def default_dataset_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M34"
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled RAG fixtures: {path}")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_query(text: str) -> str:
    """Collapse whitespace. Do not rewrite meaning and do not re-embed here."""

    return " ".join(str(text).strip().split())


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


def normalize_answer(text: str) -> str:
    return " ".join(tokenize(text))


def span_supports_claim(claim: str, span: str) -> bool:
    """A span supports a claim when every claim content stem appears in the span."""

    claim_stems = content_stems(claim)
    if not claim_stems:
        return False
    span_stems = frozenset(stem_token(token) for token in tokenize(span))
    return claim_stems <= span_stems


def answers_match(predicted: str, gold: str | None, aliases: Sequence[str] = ()) -> bool:
    if gold is None:
        return False
    predicted_norm = normalize_answer(predicted)
    options = {normalize_answer(gold), *(normalize_answer(alias) for alias in aliases)}
    return predicted_norm in options


@dataclass(frozen=True)
class ContextBudget:
    max_chars: int = DEFAULT_BUDGET_CHARS
    max_chunks: int = DEFAULT_BUDGET_CHUNKS

    def __post_init__(self) -> None:
        if int(self.max_chars) < 1:
            raise ValueError("max_chars must be positive")
        if int(self.max_chunks) < 1:
            raise ValueError("max_chunks must be positive")


@dataclass(frozen=True)
class InferenceConfig:
    synthesizer_id: str = SYNTHESIZER_ID
    synthesizer_family: str = SYNTHESIZER_FAMILY
    policy: str = POLICY_GATED
    retrieval_enabled: bool = True
    top_k: int = DEFAULT_TOP_K
    budget_chars: int = DEFAULT_BUDGET_CHARS
    budget_chunks: int = DEFAULT_BUDGET_CHUNKS
    weights_updated: bool = False
    decoding: str = "extractive-copy"
    seed: int | None = SEED
    adapter: str = "canonical"
    defect: str = DEFECT_NONE
    pipeline_id: str = PIPELINE_ID

    def as_dict(self) -> dict[str, object]:
        return {
            "synthesizer_id": self.synthesizer_id,
            "synthesizer_family": self.synthesizer_family,
            "policy": self.policy,
            "retrieval_enabled": self.retrieval_enabled,
            "top_k": self.top_k,
            "budget_chars": self.budget_chars,
            "budget_chunks": self.budget_chunks,
            "weights_updated": self.weights_updated,
            "decoding": self.decoding,
            "seed": self.seed,
            "adapter": self.adapter,
            "defect": self.defect,
            "pipeline_id": self.pipeline_id,
            "inference_boundary": INFERENCE_BOUNDARY,
        }


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_id: str
    span_start: int
    span_end: int
    quote: str
    index_id: str
    source_hash: str
    corpus_version: str
    model: str
    version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "span": {"start": self.span_start, "end": self.span_end},
            "quote": self.quote,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "corpus_version": self.corpus_version,
            "model": self.model,
            "version": self.version,
        }


@dataclass(frozen=True)
class PackedItem:
    rank: int
    score: float
    document_id: str
    chunk_id: str
    text: str
    span_start: int
    span_end: int
    metadata: tuple[tuple[str, str], ...]
    m28_id: str
    index_id: str
    source_hash: str
    corpus_version: str
    model: str
    version: str
    metric: str
    normalization: str
    packed_offset: int

    def metadata_dict(self) -> dict[str, str]:
        return {key: value for key, value in self.metadata}

    def as_citation(self) -> Citation:
        return Citation(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            span_start=self.span_start,
            span_end=self.span_end,
            quote=self.text,
            index_id=self.index_id,
            source_hash=self.source_hash,
            corpus_version=self.corpus_version,
            model=self.model,
            version=self.version,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "span": {"start": self.span_start, "end": self.span_end},
            "metadata": self.metadata_dict(),
            "m28_id": self.m28_id,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "corpus_version": self.corpus_version,
            "model": self.model,
            "version": self.version,
            "metric": self.metric,
            "normalization": self.normalization,
            "packed_offset": self.packed_offset,
        }


@dataclass(frozen=True)
class ContextPack:
    query_id: str
    query_text: str
    budget: ContextBudget
    items: tuple[PackedItem, ...]
    dropped: tuple[PackedItem, ...]
    retrieval_top_k: int
    index_id: str
    source_hash: str
    corpus_version: str
    model: str
    version: str
    metric: str
    normalization: str
    scored_candidates: int
    formatted: str
    pack_hash: str

    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.items)

    def dropped_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.dropped)

    def char_count(self) -> int:
        if not self.items:
            return 0
        return sum(len(item.text) for item in self.items) + len(SEPARATOR) * (len(self.items) - 1)

    def truncated(self) -> bool:
        return bool(self.dropped)

    def get(self, chunk_id: str) -> PackedItem:
        for item in self.items:
            if item.chunk_id == chunk_id:
                return item
        raise KeyError(chunk_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "budget": {"max_chars": self.budget.max_chars, "max_chunks": self.budget.max_chunks},
            "items": tuple(item.as_dict() for item in self.items),
            "dropped": tuple(item.as_dict() for item in self.dropped),
            "retrieval_top_k": self.retrieval_top_k,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "corpus_version": self.corpus_version,
            "model": self.model,
            "version": self.version,
            "metric": self.metric,
            "normalization": self.normalization,
            "scored_candidates": self.scored_candidates,
            "formatted": self.formatted,
            "pack_hash": self.pack_hash,
            "char_count": self.char_count(),
            "truncated": self.truncated(),
        }


@dataclass(frozen=True)
class SupportIssue:
    kind: str
    chunk_id: str | None
    detail: str


@dataclass(frozen=True)
class SupportReport:
    ok: bool
    issues: tuple[SupportIssue, ...]
    supported_citations: tuple[str, ...]
    unsupported_citations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "issues": tuple({"kind": issue.kind, "chunk_id": issue.chunk_id, "detail": issue.detail} for issue in self.issues),
            "supported_citations": self.supported_citations,
            "unsupported_citations": self.unsupported_citations,
        }


@dataclass(frozen=True)
class GroundedAnswer:
    query_id: str
    query_text: str
    status: str
    text: str
    claims: tuple[str, ...]
    citations: tuple[Citation, ...]
    abstain_reason: str | None
    support: SupportReport
    policy: str
    defect: str

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    def citation_ids(self) -> tuple[str, ...]:
        return tuple(citation.chunk_id for citation in self.citations)

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "status": self.status,
            "text": self.text,
            "claims": self.claims,
            "citations": tuple(citation.as_dict() for citation in self.citations),
            "abstain_reason": self.abstain_reason,
            "support": self.support.as_dict(),
            "policy": self.policy,
            "defect": self.defect,
        }


@dataclass(frozen=True)
class LabeledRagQuery:
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


@dataclass(frozen=True)
class RagTrace:
    query_id: str
    query_text: str
    inference: InferenceConfig
    retrieval_ids: tuple[str, ...]
    retrieval_scores: tuple[float, ...]
    scored_candidates: int
    latency_ms: float
    index_id: str
    source_hash: str
    corpus_version: str
    model: str
    version: str
    metric: str
    normalization: str
    pack: ContextPack
    answer: GroundedAnswer
    evaluation: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "inference": self.inference.as_dict(),
            "retrieval_ids": self.retrieval_ids,
            "retrieval_scores": self.retrieval_scores,
            "scored_candidates": self.scored_candidates,
            "latency_ms": round(self.latency_ms, 3),
            "index_id": self.index_id,
            "source_hash": self.source_hash,
            "corpus_version": self.corpus_version,
            "model": self.model,
            "version": self.version,
            "metric": self.metric,
            "normalization": self.normalization,
            "pack": self.pack.as_dict(),
            "answer": self.answer.as_dict(),
            "evaluation": self.evaluation,
        }


def _meta_tuple(metadata: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if not metadata:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in metadata.items()))


def evidence_from_hit(hit: RankedHit) -> dict[str, object]:
    """M33 ranked evidence row. Generation is not part of this object."""

    return hit.as_evidence()


def evidence_from_response(response: SearchResponse) -> tuple[dict[str, object], ...]:
    return tuple(evidence_from_hit(hit) for hit in response.hits)


def _as_evidence_row(item: RankedHit | Mapping[str, object]) -> dict[str, object]:
    if isinstance(item, RankedHit):
        return item.as_evidence()
    if isinstance(item, Mapping):
        return dict(item)
    raise TypeError("context items must be RankedHit or as_evidence mappings")


def packed_from_evidence(row: Mapping[str, object], *, packed_offset: int) -> PackedItem:
    span = row.get("span") or {}
    return PackedItem(
        rank=int(row["rank"]),
        score=float(row["score"]),
        document_id=str(row["document_id"]),
        chunk_id=str(row["chunk_id"]),
        text=str(row["text"]),
        span_start=int(span.get("start", 0)),
        span_end=int(span.get("end", 0)),
        metadata=_meta_tuple(row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}),
        m28_id=str(row.get("m28_id") or ""),
        index_id=str(row.get("index_id") or ""),
        source_hash=str(row.get("source_hash") or ""),
        corpus_version=str(row.get("corpus_version") or ""),
        model=str(row.get("model") or ""),
        version=str(row.get("version") or ""),
        metric=str(row.get("metric") or ""),
        normalization=str(row.get("normalization") or ""),
        packed_offset=int(packed_offset),
    )


def format_pack_items(items: Sequence[PackedItem]) -> str:
    blocks = []
    for item in items:
        header = f"[{item.packed_offset + 1}] {item.chunk_id} score={item.score:.3f}"
        blocks.append(f"{header}\n{item.text}")
    return "\n\n".join(blocks)


def pack_hash(items: Sequence[PackedItem]) -> str:
    payload = [{"chunk_id": item.chunk_id, "text": item.text, "rank": item.rank} for item in items]
    return hashlib.sha256(_dump_canonical(payload).encode("utf-8")).hexdigest()


def pack_context(
    evidence: Sequence[RankedHit | Mapping[str, object]],
    *,
    query_id: str,
    query_text: str,
    budget: ContextBudget | None = None,
    retrieval_top_k: int = DEFAULT_TOP_K,
    scored_candidates: int = 0,
    index_id: str = "",
    source_hash: str = "",
    corpus_version: str = "",
    model: str = "",
    version: str = "",
    metric: str = "",
    normalization: str = "",
) -> ContextPack:
    """Pack retrieved evidence in rank order. Never reorder or split chunks."""

    used_budget = budget or ContextBudget()
    rows = [_as_evidence_row(item) for item in evidence]
    kept: list[PackedItem] = []
    dropped_rows: list[Mapping[str, object]] = []
    used_chars = 0
    for row in rows:
        if len(kept) >= used_budget.max_chunks:
            dropped_rows.append(row)
            continue
        extra = len(str(row["text"])) if not kept else len(SEPARATOR) + len(str(row["text"]))
        if used_chars + extra > used_budget.max_chars:
            dropped_rows.append(row)
            continue
        kept.append(packed_from_evidence(row, packed_offset=len(kept)))
        used_chars += extra
    identity_row = rows[0] if rows else {}
    if not index_id:
        if kept:
            index_id = kept[0].index_id
            source_hash = kept[0].source_hash
            corpus_version = kept[0].corpus_version
            model = kept[0].model
            version = kept[0].version
            metric = kept[0].metric
            normalization = kept[0].normalization
        elif identity_row:
            index_id = str(identity_row.get("index_id") or "")
            source_hash = str(identity_row.get("source_hash") or "")
            corpus_version = str(identity_row.get("corpus_version") or "")
            model = str(identity_row.get("model") or "")
            version = str(identity_row.get("version") or "")
            metric = str(identity_row.get("metric") or "")
            normalization = str(identity_row.get("normalization") or "")
    items = tuple(kept)
    dropped_items = tuple(
        packed_from_evidence(row, packed_offset=len(items) + offset)
        for offset, row in enumerate(dropped_rows)
    )
    return ContextPack(
        query_id=query_id,
        query_text=query_text,
        budget=used_budget,
        items=items,
        dropped=dropped_items,
        retrieval_top_k=int(retrieval_top_k),
        index_id=index_id,
        source_hash=source_hash,
        corpus_version=corpus_version,
        model=model,
        version=version,
        metric=metric,
        normalization=normalization,
        scored_candidates=int(scored_candidates),
        formatted=format_pack_items(items),
        pack_hash=pack_hash(items),
    )


def retrieve(
    query_text: str,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    filters: Mapping[str, object] | None = None,
    index: ExactIndex | None = None,
    corpus: Corpus | None = None,
    enforce_freshness: bool = True,
    enforce_provenance: bool = True,
) -> SearchResponse:
    """Query -> M33 search. Returns ranked evidence, not an answer."""

    used_index = index or load_canonical_index()
    used_corpus = corpus or load_canonical_corpus()
    encoded = encode_query(query_text, query_id=query_id)
    return search(
        used_index,
        encoded,
        top_k=int(top_k),
        filters=filters,
        query_id=query_id,
        live_corpus=used_corpus,
        enforce_freshness=enforce_freshness,
        enforce_provenance=enforce_provenance,
    )


def _coverage_stats(query_text: str, item: PackedItem) -> tuple[float, int]:
    q_content = content_stems(query_text)
    span_content = content_stems(item.text)
    overlap = q_content & span_content
    coverage = (len(overlap) / len(q_content)) if q_content else 0.0
    return coverage, len(overlap)


def _eligible_for_query(query_text: str, item: PackedItem) -> bool:
    q_digits = query_digits(query_text)
    span_tokens = frozenset(tokenize(item.text))
    if q_digits and not q_digits <= span_tokens:
        return False
    q_content = content_stems(query_text)
    if not q_content:
        return bool(q_digits)
    overlap = len(q_content & content_stems(item.text))
    needed = min(MIN_OVERLAP_TOKENS, len(q_content))
    if q_digits and q_digits <= span_tokens:
        return True
    return overlap >= needed


def select_source(query_text: str, items: Sequence[PackedItem], *, policy: str) -> PackedItem | None:
    if not items:
        return None
    if policy == POLICY_NAIVE:
        return items[0]
    if policy != POLICY_GATED:
        raise ValueError(f"unsupported policy: {policy}")

    eligible = [item for item in items if _eligible_for_query(query_text, item)]
    procedure = bool(PROCEDURE_RE.search(query_text))
    action_items = [item for item in items if ACTION_RE.match(item.text.strip())]
    if procedure:
        if not action_items:
            return None
        high_docs = {
            item.document_id
            for item in eligible
            if _coverage_stats(query_text, item)[0] >= 0.5
        }
        preferred = [
            item
            for item in action_items
            if item.document_id in high_docs or _eligible_for_query(query_text, item)
        ]
        pool = preferred or [item for item in action_items if _eligible_for_query(query_text, item)]
        if not pool:
            return None
    else:
        pool = eligible
        if not pool:
            return None

    q_neg = "not" in tokenize(query_text) or "never" in tokenize(query_text)
    if q_neg:
        negated = [item for item in pool if "not" in tokenize(item.text) or "never" in tokenize(item.text)]
        if negated:
            pool = negated

    ranked = sorted(
        pool,
        key=lambda item: (
            -_coverage_stats(query_text, item)[0],
            -_coverage_stats(query_text, item)[1],
            item.rank,
        ),
    )
    return ranked[0]


def empty_support() -> SupportReport:
    return SupportReport(ok=True, issues=(), supported_citations=(), unsupported_citations=())


def verify_support(answer: GroundedAnswer, pack: ContextPack) -> SupportReport:
    """Fail when a citation is missing from the pack or does not support the claim."""

    if answer.abstained:
        if answer.citations:
            issue = SupportIssue(
                kind="abstain_with_citation",
                chunk_id=answer.citations[0].chunk_id,
                detail="abstention must not carry an evidence citation",
            )
            return SupportReport(
                ok=False,
                issues=(issue,),
                supported_citations=(),
                unsupported_citations=answer.citation_ids(),
            )
        return empty_support()

    packed_ids = set(pack.chunk_ids())
    issues: list[SupportIssue] = []
    supported: list[str] = []
    unsupported: list[str] = []
    claims = answer.claims or ((answer.text,) if answer.text else ())
    if not answer.citations:
        issues.append(SupportIssue(kind="missing_citation", chunk_id=None, detail="answered without a citation"))
    for citation in answer.citations:
        if citation.chunk_id not in packed_ids:
            unsupported.append(citation.chunk_id)
            issues.append(
                SupportIssue(
                    kind="citation_not_in_pack",
                    chunk_id=citation.chunk_id,
                    detail="citation is not in the packed context",
                )
            )
            continue
        packed = pack.get(citation.chunk_id)
        if citation.quote not in packed.text:
            unsupported.append(citation.chunk_id)
            issues.append(
                SupportIssue(
                    kind="quote_mismatch",
                    chunk_id=citation.chunk_id,
                    detail="quoted span is not in the packed chunk text",
                )
            )
            continue
        claim_ok = all(span_supports_claim(claim, packed.text) for claim in claims)
        if not claim_ok:
            unsupported.append(citation.chunk_id)
            issues.append(
                SupportIssue(
                    kind="unsupported_claim",
                    chunk_id=citation.chunk_id,
                    detail="cited chunk does not contain the claim stems",
                )
            )
            continue
        supported.append(citation.chunk_id)
    ok = not issues
    return SupportReport(
        ok=ok,
        issues=tuple(issues),
        supported_citations=tuple(supported),
        unsupported_citations=tuple(unsupported),
    )


def _with_support(answer: GroundedAnswer, support: SupportReport) -> GroundedAnswer:
    return GroundedAnswer(
        query_id=answer.query_id,
        query_text=answer.query_text,
        status=answer.status,
        text=answer.text,
        claims=answer.claims,
        citations=answer.citations,
        abstain_reason=answer.abstain_reason,
        support=support,
        policy=answer.policy,
        defect=answer.defect,
    )


def abstain(
    query_id: str,
    query_text: str,
    *,
    reason: str,
    policy: str,
    defect: str = DEFECT_NONE,
) -> GroundedAnswer:
    answer = GroundedAnswer(
        query_id=query_id,
        query_text=query_text,
        status="abstained",
        text=ABSTAIN_TEXT,
        claims=(),
        citations=(),
        abstain_reason=reason,
        support=empty_support(),
        policy=policy,
        defect=defect,
    )
    return answer


def _invented_claim(query_text: str) -> str:
    digits = tuple(sorted(query_digits(query_text), key=lambda token: (len(token), token)))
    if digits:
        return f"Ticket {digits[0]} is waiting for inspection."
    return "Valley Services is led by the CEO named in the handbook."


def _answered_from_item(
    query_id: str,
    query_text: str,
    item: PackedItem,
    *,
    policy: str,
    defect: str,
    citation: PackedItem | None = None,
    claim: str | None = None,
) -> GroundedAnswer:
    cited = citation or item
    text = claim if claim is not None else item.text
    return GroundedAnswer(
        query_id=query_id,
        query_text=query_text,
        status="answered",
        text=text,
        claims=(text,),
        citations=(cited.as_citation(),),
        abstain_reason=None,
        support=empty_support(),
        policy=policy,
        defect=defect,
    )


def synthesize(
    query_text: str,
    pack: ContextPack,
    *,
    policy: str = POLICY_GATED,
    defect: str = DEFECT_NONE,
    query_id: str | None = None,
) -> GroundedAnswer:
    """Copy a supported span from the pack, or abstain. No live model."""

    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if defect not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect: {defect}")
    used_id = query_id or pack.query_id
    used_query = query_text or pack.query_text

    if defect == DEFECT_CLOSED:
        fake = PackedItem(
            rank=0,
            score=0.0,
            document_id="doc-device-printer",
            chunk_id="doc-device-printer::c0",
            text="Please reset the printer.",
            span_start=0,
            span_end=25,
            metadata=(),
            m28_id="",
            index_id=pack.index_id,
            source_hash=pack.source_hash,
            corpus_version=pack.corpus_version,
            model=pack.model,
            version=pack.version,
            metric=pack.metric,
            normalization=pack.normalization,
            packed_offset=-1,
        )
        return _answered_from_item(
            used_id,
            used_query,
            fake,
            policy=policy,
            defect=defect,
            claim="Please reset the printer.",
        )

    if not pack.items:
        if defect == DEFECT_INVENTED:
            return GroundedAnswer(
                query_id=used_id,
                query_text=used_query,
                status="answered",
                text=_invented_claim(used_query),
                claims=(_invented_claim(used_query),),
                citations=(),
                abstain_reason=None,
                support=empty_support(),
                policy=policy,
                defect=defect,
            )
        return abstain(used_id, used_query, reason="empty_context", policy=policy, defect=defect)

    source = select_source(used_query, pack.items, policy=policy)

    if defect == DEFECT_INVENTED:
        claim = _invented_claim(used_query)
        cited = pack.items[0]
        return _answered_from_item(
            used_id,
            used_query,
            cited,
            policy=policy,
            defect=defect,
            citation=cited,
            claim=claim,
        )

    if source is None:
        return abstain(
            used_id,
            used_query,
            reason="insufficient_overlap",
            policy=policy,
            defect=defect,
        )

    if defect == DEFECT_UNSUPPORTED:
        claim = source.text
        distractor = next(
            (item for item in pack.items if not span_supports_claim(claim, item.text)),
            None,
        )
        if distractor is None:
            distractor = pack.items[0]
        return _answered_from_item(
            used_id,
            used_query,
            source,
            policy=policy,
            defect=defect,
            citation=distractor,
            claim=claim,
        )

    return _answered_from_item(
        used_id,
        used_query,
        source,
        policy=policy,
        defect=defect,
    )


def repair_grounding(answer: GroundedAnswer, pack: ContextPack) -> GroundedAnswer:
    """Rebind citations to packed spans that support the claim, or abstain."""

    if answer.abstained:
        return _with_support(answer, verify_support(answer, pack))
    claims = answer.claims or ((answer.text,) if answer.text else ())
    supporters: list[PackedItem] = []
    for item in pack.items:
        if all(span_supports_claim(claim, item.text) for claim in claims):
            supporters.append(item)
    if not supporters or not claims:
        repaired = abstain(
            answer.query_id,
            answer.query_text,
            reason="unsupported_claim",
            policy=answer.policy,
            defect=DEFECT_NONE,
        )
        return _with_support(repaired, verify_support(repaired, pack))
    source = supporters[0]
    repaired = _answered_from_item(
        answer.query_id,
        answer.query_text,
        source,
        policy=answer.policy,
        defect=DEFECT_NONE,
        claim=claims[0],
    )
    return _with_support(repaired, verify_support(repaired, pack))


def classify_failure(trace: RagTrace, labeled: LabeledRagQuery) -> dict[str, object]:
    """Separate retrieval, context, generation, and citation failures."""

    retrieved = set(trace.retrieval_ids)
    packed = set(trace.pack.chunk_ids())
    gold_support = set(labeled.support_chunk_ids)
    retrieval_hit = bool(gold_support & retrieved) if labeled.answerable else None
    packed_hit = bool(gold_support & packed) if labeled.answerable else None
    answered = trace.answer.answered
    abstained = trace.answer.abstained
    citations_supported = bool(trace.answer.support.ok) if answered else True
    correct = bool(
        answered
        and answers_match(trace.answer.text, labeled.gold_answer, labeled.gold_aliases)
    )
    layers: list[str] = []
    if labeled.answerable and gold_support:
        if not retrieval_hit:
            layers.append("retrieval")
        elif not packed_hit:
            layers.append("context")
    if answered:
        if not citations_supported:
            layers.append("citation")
        elif labeled.answerable and not correct:
            layers.append("generation")
        elif not labeled.answerable:
            layers.append("generation")
    elif labeled.answerable and packed_hit:
        layers.append("generation")
    primary = layers[0] if layers else None
    return {
        "query_id": labeled.query_id,
        "primary": primary,
        "layers": tuple(layers),
        "retrieval_hit": retrieval_hit,
        "packed_hit": packed_hit,
        "citations_supported": citations_supported,
        "answer_correct": correct,
        "abstained": abstained,
        "answered": answered,
        "answerable": labeled.answerable,
        "high_score_is_not_correctness": True,
    }


def evaluate_answer(trace: RagTrace, labeled: LabeledRagQuery) -> dict[str, object]:
    classification = classify_failure(trace, labeled)
    answered = trace.answer.answered
    abstained = trace.answer.abstained
    support_ok = bool(trace.answer.support.ok)
    correct = bool(classification["answer_correct"])
    grounded = bool(answered and correct and support_ok)
    if labeled.answerable:
        eval_pass = grounded
    else:
        eval_pass = bool(abstained)
    if answered and not support_ok:
        eval_pass = False
    return {
        "query_id": labeled.query_id,
        "split": labeled.split,
        "answerable": labeled.answerable,
        "status": trace.answer.status,
        "predicted": trace.answer.text,
        "gold_answer": labeled.gold_answer,
        "citation_ids": trace.answer.citation_ids(),
        "support_ok": support_ok,
        "eval_pass": eval_pass,
        "grounded": grounded,
        "retrieval_ids": trace.retrieval_ids,
        "packed_ids": trace.pack.chunk_ids(),
        "dropped_ids": trace.pack.dropped_ids(),
        "top_score": trace.retrieval_scores[0] if trace.retrieval_scores else None,
        "index_id": trace.index_id,
        "source_hash": trace.source_hash,
        "model": trace.model,
        "version": trace.version,
        **classification,
    }


def _labeled_from_payload(row: Mapping[str, object]) -> LabeledRagQuery:
    gold = row.get("gold_answer")
    return LabeledRagQuery(
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
    )


@lru_cache(maxsize=4)
def load_questions_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "questions.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_expected_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "expected.json"
    return _read_json(target)


@lru_cache(maxsize=4)
def load_transfer_payload(path: str | None = None) -> dict:
    target = Path(path) if path else default_dataset_dir() / "transfer.json"
    return _read_json(target)


def load_labeled_queries() -> tuple[LabeledRagQuery, ...]:
    return tuple(_labeled_from_payload(row) for row in load_questions_payload()["questions"])


def load_query_map() -> dict[str, LabeledRagQuery]:
    return {query.query_id: query for query in load_labeled_queries()}


def load_split(split: str) -> tuple[LabeledRagQuery, ...]:
    return tuple(query for query in load_labeled_queries() if query.split == split)


def answer_query(
    query_text: str,
    *,
    query_id: str = "query",
    top_k: int = DEFAULT_TOP_K,
    retrieval_enabled: bool = True,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    budget_chunks: int = DEFAULT_BUDGET_CHUNKS,
    policy: str = POLICY_GATED,
    defect: str = DEFECT_NONE,
    filters: Mapping[str, object] | None = None,
    index: ExactIndex | None = None,
    corpus: Corpus | None = None,
    labeled: LabeledRagQuery | None = None,
) -> RagTrace:
    """Useful whole: retrieve, pack, synthesize, verify, optionally evaluate."""

    started = time.perf_counter()
    used_index = index or load_canonical_index()
    used_corpus = corpus or load_canonical_corpus()
    query_text = normalize_query(query_text)
    inference = InferenceConfig(
        policy=policy,
        retrieval_enabled=bool(retrieval_enabled),
        top_k=int(top_k),
        budget_chars=int(budget_chars),
        budget_chunks=int(budget_chunks),
        defect=defect,
    )
    budget = ContextBudget(max_chars=int(budget_chars), max_chunks=int(budget_chunks))
    response = None
    evidence: tuple[dict[str, object], ...] = ()
    scored_candidates = 0
    retrieval_ids: tuple[str, ...] = ()
    retrieval_scores: tuple[float, ...] = ()
    if retrieval_enabled:
        response = retrieve(
            query_text,
            query_id=query_id,
            top_k=top_k,
            filters=filters,
            index=used_index,
            corpus=used_corpus,
        )
        evidence = evidence_from_response(response)
        scored_candidates = int(response.scored_candidates)
        retrieval_ids = response.ids()
        retrieval_scores = tuple(hit.score for hit in response.hits)
    pack = pack_context(
        evidence,
        query_id=query_id,
        query_text=query_text,
        budget=budget,
        retrieval_top_k=int(top_k),
        scored_candidates=scored_candidates,
        index_id=used_index.metadata.index_id,
        source_hash=used_index.metadata.source_hash,
        corpus_version=used_index.metadata.corpus_version,
        model=used_index.metadata.embedding.model,
        version=used_index.metadata.embedding.version,
        metric=used_index.metadata.embedding.metric,
        normalization=used_index.metadata.embedding.normalization,
    )
    raw_answer = synthesize(
        query_text,
        pack,
        policy=policy,
        defect=defect,
        query_id=query_id,
    )
    support = verify_support(raw_answer, pack)
    answer = _with_support(raw_answer, support)
    latency_ms = (time.perf_counter() - started) * 1000.0
    trace = RagTrace(
        query_id=query_id,
        query_text=query_text,
        inference=inference,
        retrieval_ids=retrieval_ids,
        retrieval_scores=retrieval_scores,
        scored_candidates=scored_candidates,
        latency_ms=float(latency_ms),
        index_id=used_index.metadata.index_id,
        source_hash=used_index.metadata.source_hash,
        corpus_version=used_index.metadata.corpus_version,
        model=used_index.metadata.embedding.model,
        version=used_index.metadata.embedding.version,
        metric=used_index.metadata.embedding.metric,
        normalization=used_index.metadata.embedding.normalization,
        pack=pack,
        answer=answer,
    )
    evaluation = evaluate_answer(trace, labeled) if labeled is not None else {}
    return RagTrace(
        query_id=trace.query_id,
        query_text=trace.query_text,
        inference=trace.inference,
        retrieval_ids=trace.retrieval_ids,
        retrieval_scores=trace.retrieval_scores,
        scored_candidates=trace.scored_candidates,
        latency_ms=trace.latency_ms,
        index_id=trace.index_id,
        source_hash=trace.source_hash,
        corpus_version=trace.corpus_version,
        model=trace.model,
        version=trace.version,
        metric=trace.metric,
        normalization=trace.normalization,
        pack=trace.pack,
        answer=trace.answer,
        evaluation=evaluation,
    )


def answer_labeled(
    query_id: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    retrieval_enabled: bool = True,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    budget_chunks: int = DEFAULT_BUDGET_CHUNKS,
    policy: str = POLICY_GATED,
    defect: str = DEFECT_NONE,
    filters: Mapping[str, object] | None = None,
    index: ExactIndex | None = None,
    corpus: Corpus | None = None,
) -> RagTrace:
    labeled = load_query_map()[query_id]
    return answer_query(
        labeled.text,
        query_id=labeled.query_id,
        top_k=top_k,
        retrieval_enabled=retrieval_enabled,
        budget_chars=budget_chars,
        budget_chunks=budget_chunks,
        policy=policy,
        defect=defect,
        filters=filters,
        index=index,
        corpus=corpus,
        labeled=labeled,
    )


def evaluate_set(
    queries: Sequence[LabeledRagQuery] | None = None,
    *,
    split: str | None = "holdout",
    top_k: int = DEFAULT_TOP_K,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    policy: str = POLICY_GATED,
    index: ExactIndex | None = None,
    corpus: Corpus | None = None,
) -> dict[str, object]:
    if queries is None:
        queries = load_split(split) if split else load_labeled_queries()
    rows = []
    for labeled in queries:
        trace = answer_query(
            labeled.text,
            query_id=labeled.query_id,
            top_k=top_k,
            budget_chars=budget_chars,
            policy=policy,
            index=index,
            corpus=corpus,
            labeled=labeled,
        )
        rows.append(trace.evaluation)
    n = len(rows)
    n_pass = sum(1 for row in rows if row["eval_pass"])
    n_abstain = sum(1 for row in rows if row["abstained"])
    n_grounded = sum(1 for row in rows if row["grounded"])
    n_unsupported = sum(1 for row in rows if row["answered"] and not row["support_ok"])
    layer_counts = {layer: 0 for layer in FAILURE_LAYERS}
    for row in rows:
        primary = row.get("primary")
        if primary in layer_counts:
            layer_counts[primary] += 1
    return {
        "eval_version": EVAL_VERSION,
        "split": split,
        "n": n,
        "n_pass": n_pass,
        "pass_rate": (n_pass / n) if n else 0.0,
        "n_abstain": n_abstain,
        "n_grounded": n_grounded,
        "n_unsupported_citation": n_unsupported,
        "layer_counts": layer_counts,
        "policy": policy,
        "top_k": top_k,
        "budget_chars": budget_chars,
        "held_out_untuned": split == "holdout",
        "rows": tuple(rows),
    }


def trace_report(trace: RagTrace) -> dict[str, object]:
    return {
        "query_id": trace.query_id,
        "query_text": trace.query_text,
        "retrieval_enabled": trace.inference.retrieval_enabled,
        "top_k": trace.inference.top_k,
        "policy": trace.inference.policy,
        "defect": trace.inference.defect,
        "retrieval_ids": trace.retrieval_ids,
        "packed_ids": trace.pack.chunk_ids(),
        "dropped_ids": trace.pack.dropped_ids(),
        "status": trace.answer.status,
        "answer": trace.answer.text,
        "citations": trace.answer.citation_ids(),
        "support_ok": trace.answer.support.ok,
        "abstain_reason": trace.answer.abstain_reason,
        "index_id": trace.index_id,
        "source_hash": trace.source_hash,
        "model": trace.model,
        "version": trace.version,
        "weights_updated": trace.inference.weights_updated,
        "decoding": trace.inference.decoding,
        "pack_hash": trace.pack.pack_hash,
        "primary_failure": (trace.evaluation or {}).get("primary"),
    }
