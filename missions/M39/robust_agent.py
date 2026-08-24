"""Memory, routing, and bounded fallbacks around the M38 workflow.

M39 consumes M38's explicit AgentState machine (nodes, checkpoints,
approval, loop bounds, tool execution) and opens a teaching robustness
layer:

working vs persistent memory → provenance/expiry/scope retrieval
  → explicit route selection → wrap M38 → bounded fallback ladder
  → circuit / attempt bound → complete | degraded | no_match | circuit_open.

Canonical path: local deterministic fixtures, not a live model.
A live / LangGraph adapter, if requested, is optional and fail-closed.
Formal eval stays M40; RAG/ANN stay M34–M36; sampling labs stay M32.

Named sources (not SDKs): langgraph-docs, anthropic-agents.
Content bundle: tool-using-agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping

SEED = 3901
ROBUST_VERSION = "v10-teaching-robust-1"
MAX_ATTEMPTS = 3
CIRCUIT_THRESHOLD = 2
ABSOLUTE_CEILING = 8
DEFAULT_NOW = 1000
CATALOG_SKU = "SKU-7"
CATALOG_PRICE = 42.0
STALE_PRICE = 99.0
IRRELEVANT_BIN = "BIN-4"
IRRELEVANT_QTY = 4
IRRELEVANT_SKU = "SKU-9"
IRRELEVANT_SKU_PRICE = 15.5
STALE_WRITTEN_AT = 100
STALE_EXPIRES_AT = 200
FRESH_WRITTEN_AT = 800
FRESH_EXPIRES_AT = 2000
DEFAULT_TASK = "Purchase SKU-7 and post the catalog price to the cash ledger."
LOOKUP_TASK = "Look up the catalog price of SKU-7."
NO_MATCH_TASK = "Compose a haiku about warehouse bins."

MEM_SKU7_PRICE = "mem-sku7-price"
MEM_BIN4_QTY = "mem-bin4-qty"
MEM_SKU9_PRICE = "mem-sku9-price"
MEM_SKU7_STALE = "mem-sku7-price-stale"
MEM_SKU7_FRESH = "mem-sku7-price-fresh"
MEM_SKU7_SUPERSEDED = "mem-sku7-price-superseded"

WORKING_EPHEMERAL = (
    "node",
    "step",
    "pending_action",
    "last_tool_result",
    "approval",
    "history",
)
PERSISTENT_CANDIDATES = ("catalog_price_fact",)

FIELD_CLASSIFICATION = {
    "working_ephemeral": WORKING_EPHEMERAL,
    "persistent_candidates": PERSISTENT_CANDIDATES,
    "note": (
        "M38 node/step/pending_action/last_tool_result/approval/history are "
        "working/ephemeral. Facts like catalog price can become persistent "
        "memory with provenance, expiry, and scope."
    ),
}

ROUTE_NAMES = ("catalog_purchase", "catalog_lookup", "no_match")
LADDER = ("primary", "lookup_only")
SUPPORTED_DEFECTS = ("none", "stale_memory_trusted", "fallback_oscillation")
TERMINALS = (
    "complete",
    "degraded",
    "no_match",
    "failed",
    "circuit_open",
)
TRACE_FIELDS = (
    "retrieved_ids",
    "excluded",
    "used_memory_ids",
    "route",
    "attempt",
    "rung",
    "circuit_open",
    "degraded",
    "terminal",
    "posted_amount",
    "provenance",
    "expiry",
)

SKU_RE = re.compile(r"SKU-[A-Z0-9]+", re.IGNORECASE)

# Fresh no-AI numbers. Do not print derived conclusions in the gate.
TRANSFER_SKU = "SKU-21"
TRANSFER_STALE_PRICE = 13.0
TRANSFER_CURRENT_PRICE = 27.0
TRANSFER_WRITTEN_AT = 5
TRANSFER_EXPIRES_AT = 8
TRANSFER_NOW = 40
TRANSFER_IRRELEVANT_BIN = "BIN-8"
TRANSFER_IRRELEVANT_QTY = 4
TRANSFER_MAX_ATTEMPTS = 2
TRANSFER_CIRCUIT = 2
TRANSFER_CASES = (
    "Refund order 4401 for SKU-21.",
    "Look up the catalog price of SKU-21.",
    "Schedule a picnic on the loading dock.",
)

ROUTE_CASES = (
    (DEFAULT_TASK, "catalog_purchase"),
    (LOOKUP_TASK, "catalog_lookup"),
    (NO_MATCH_TASK, "no_match"),
)

SCALE_LIMIT = (
    "Teaching scale: one local memory store, explicit route predicates, "
    "a two-rung fallback ladder, CPU stdlib orchestration wrapping the M38 "
    "reference machine, no paid API, no LangGraph/Anthropic SDK, no RAG pack, "
    "no Qdrant, no sampling lab, no M40 eval harness. Degraded success is "
    "explicit and is not silent incorrect success. This fixture is not a "
    "production agent runtime."
)

SYSTEM_MAP = (
    "goal\n"
    "  -> classify M38 AgentState (working vs persistent candidates)\n"
    "  -> retrieve relevant memory (provenance, expiry, scope)\n"
    "  -> select route (catalog_purchase | catalog_lookup | no_match)\n"
    "  -> primary: wrap M38 workflow for that route\n"
    "  -> on primary failure: bounded fallback (lookup_only)\n"
    "  -> circuit / attempt bound\n"
    "  terminals: complete | degraded | no_match | failed | circuit_open\n"
    "  degraded success is labeled; stale memory must not look like complete"
)

HANDOFF = (
    "M40 receives a robust agent with declared memory, route, fallback, "
    "and trace surfaces (retrieved ids, excluded reasons, used memory ids, "
    "route name, attempt count, circuit state, degraded flag, provenance/"
    "expiry annotations) that can be evaluated systematically. M39 does not "
    "open a formal eval harness."
)


class LiveAdapterUnavailable(RuntimeError):
    """Raised when a live model/store is requested; canonical tests must not need one."""


class OptionalLangGraphUnavailable(RuntimeError):
    """Raised when a LangGraph store adapter is requested; canonical tests must not need one."""


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M39 optional NumPy path requires requirements/m39.txt") from exc
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


def _load_m38():
    """Load M38's workflow. Prefer the package so class identity matches.

    File-load is only a fallback and still registers the package name.
    """

    packaged_name = "missions.M38.agent_workflow"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(packaged_name)
    except ImportError:
        return _load_sibling("M38", "agent_workflow.py", packaged_name)


def _normalize_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none"}:
        return "none"
    if key not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return key


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return json.loads(json.dumps(dict(value)))


def extract_sku(text: str) -> str | None:
    match = SKU_RE.search(text or "")
    if match is None:
        return None
    return match.group(0).upper()


@dataclass(frozen=True)
class Provenance:
    source: str
    writer: str
    written_at: int
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "writer": self.writer,
            "written_at": self.written_at,
            "method": self.method,
        }


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    key: str
    value: Any
    scope: str
    provenance: Provenance
    sku: str | None = None
    expires_at: int | None = None
    superseded_by: str | None = None
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "value": json.loads(json.dumps(self.value)),
            "scope": self.scope,
            "sku": self.sku,
            "expires_at": self.expires_at,
            "superseded_by": self.superseded_by,
            "tags": list(self.tags),
            "provenance": self.provenance.as_dict(),
        }


def entry_from_dict(payload: Mapping[str, Any]) -> MemoryEntry:
    prov = payload.get("provenance") or {}
    return MemoryEntry(
        entry_id=str(payload["entry_id"]),
        key=str(payload["key"]),
        value=payload.get("value"),
        scope=str(payload.get("scope") or "catalog"),
        sku=payload.get("sku"),
        provenance=Provenance(
            source=str(prov.get("source") or "unknown"),
            writer=str(prov.get("writer") or "unknown"),
            written_at=int(prov.get("written_at") or 0),
            method=str(prov.get("method") or "asserted"),
        ),
        expires_at=None if payload.get("expires_at") is None else int(payload["expires_at"]),
        superseded_by=payload.get("superseded_by"),
        tags=tuple(str(tag) for tag in (payload.get("tags") or ())),
    )


@dataclass(frozen=True)
class MemoryStore:
    entries: tuple[MemoryEntry, ...] = ()

    def put(self, entry: MemoryEntry) -> MemoryStore:
        others = tuple(item for item in self.entries if item.entry_id != entry.entry_id)
        return MemoryStore(others + (entry,))

    def get(self, entry_id: str) -> MemoryEntry | None:
        for item in self.entries:
            if item.entry_id == entry_id:
                return item
        return None

    def as_dicts(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.entries]


def restore_store(rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> MemoryStore:
    store = MemoryStore()
    for row in rows:
        store = store.put(entry_from_dict(row))
    return store


def lifecycle(entry: MemoryEntry, now: int) -> str:
    if entry.superseded_by:
        return "superseded"
    if entry.expires_at is not None and int(now) >= int(entry.expires_at):
        return "expired"
    return "active"


def write_memory(
    store: MemoryStore,
    *,
    entry_id: str,
    key: str,
    value: Any,
    scope: str,
    now: int,
    source: str,
    writer: str = "m39",
    method: str = "asserted",
    sku: str | None = None,
    ttl: int | None = None,
    expires_at: int | None = None,
    tags: tuple[str, ...] = (),
) -> MemoryStore:
    """Append or replace a provenance-bearing persistent entry."""

    expiry = expires_at
    if expiry is None and ttl is not None:
        expiry = int(now) + int(ttl)
    entry = MemoryEntry(
        entry_id=entry_id,
        key=key,
        value=value,
        scope=scope,
        sku=sku,
        provenance=Provenance(
            source=source,
            writer=writer,
            written_at=int(now),
            method=method,
        ),
        expires_at=expiry,
        tags=tags,
    )
    return store.put(entry)


def supersede_memory(
    store: MemoryStore,
    old_id: str,
    new_entry: MemoryEntry,
) -> MemoryStore:
    old = store.get(old_id)
    if old is None:
        raise KeyError(old_id)
    marked = replace(old, superseded_by=new_entry.entry_id)
    return store.put(marked).put(new_entry)


@dataclass(frozen=True)
class RetrievalResult:
    included: tuple[MemoryEntry, ...]
    excluded: tuple[tuple[MemoryEntry, str], ...]
    query: dict[str, Any]
    now: int

    @property
    def included_ids(self) -> tuple[str, ...]:
        return tuple(item.entry_id for item in self.included)

    def brief(self) -> dict[str, Any]:
        return {
            "included_ids": list(self.included_ids),
            "excluded": [
                {"id": item.entry_id, "reason": reason} for item, reason in self.excluded
            ],
            "now": self.now,
            "query_sku": self.query.get("sku"),
            "query_scope": self.query.get("scope"),
        }


def memory_query_from_task(task: str, *, route: str | None = None) -> dict[str, Any]:
    sku = extract_sku(task)
    if route == "no_match":
        scope = None
    else:
        scope = "catalog"
    return {"sku": sku, "scope": scope, "route": route, "text": task}


def _relevance_reason(
    entry: MemoryEntry,
    query: Mapping[str, Any],
    now: int,
    *,
    skip_expiry: bool,
) -> str | None:
    """Return an exclusion reason, or None if the entry is relevant."""

    scope = query.get("scope")
    sku = query.get("sku")
    if not scope:
        return "no_retrieval_scope"
    if entry.scope != scope:
        return "scope_mismatch"
    if sku:
        if entry.sku and entry.sku != sku:
            return "sku_mismatch"
        if not entry.sku:
            return "unscoped_entry"
    elif entry.sku:
        return "unscoped_query"
    status = lifecycle(entry, now)
    if status == "superseded":
        return "superseded"
    if status == "expired" and not skip_expiry:
        return "expired"
    return None


def retrieve_memory(
    store: MemoryStore,
    query: Mapping[str, Any],
    *,
    now: int = DEFAULT_NOW,
    skip_expiry: bool = False,
) -> RetrievalResult:
    """Return only in-scope, unexpired, unsuperseded entries for this query.

    Does not dump values; callers that need a price read included entries
    deliberately. Expired or superseded rows are flagged in excluded.
    """

    included: list[MemoryEntry] = []
    excluded: list[tuple[MemoryEntry, str]] = []
    for entry in store.entries:
        reason = _relevance_reason(entry, query, now, skip_expiry=skip_expiry)
        if reason is None:
            included.append(entry)
        else:
            excluded.append((entry, reason))
    return RetrievalResult(
        included=tuple(included),
        excluded=tuple(excluded),
        query=dict(query),
        now=int(now),
    )


def _pred_purchase(text: str) -> bool:
    return ("purchase" in text or "post" in text) and "sku" in text


def _pred_lookup(text: str) -> bool:
    return (
        "look up" in text or "lookup" in text or "price of" in text
    ) and "sku" in text


ROUTE_PREDICATES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("catalog_purchase", _pred_purchase),
    ("catalog_lookup", _pred_lookup),
)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    matched_predicate: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "matched_predicate": self.matched_predicate,
        }


def select_route(
    task: str,
    *,
    policy: tuple[tuple[str, Callable[[str], bool]], ...] | None = None,
) -> RouteDecision:
    """Apply frozen predicates in precedence order. No match is explicit."""

    text = (task or "").casefold()
    rules = policy if policy is not None else ROUTE_PREDICATES
    for name, predicate in rules:
        if predicate(text):
            return RouteDecision(
                route=name,
                reason=f"matched {name}",
                matched_predicate=name,
            )
    return RouteDecision(
        route="no_match",
        reason="no predicate matched",
        matched_predicate=None,
    )


@dataclass(frozen=True)
class CircuitBreaker:
    threshold: int = CIRCUIT_THRESHOLD
    consecutive_failures: int = 0
    state: str = "closed"

    @property
    def open(self) -> bool:
        return self.state == "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record(self, ok: bool) -> CircuitBreaker:
        if ok:
            return CircuitBreaker(threshold=self.threshold, consecutive_failures=0, state="closed")
        failures = self.consecutive_failures + 1
        state = "open" if failures >= self.threshold else "closed"
        return CircuitBreaker(
            threshold=self.threshold,
            consecutive_failures=failures,
            state=state,
        )


class PostedFromMemoryPolicy:
    """Post a remembered amount without looking up. Defect path only."""

    name = "posted_from_memory"

    def __init__(self, price: float) -> None:
        self.price = float(price)

    def propose(self, state: Any) -> dict[str, Any]:
        m38 = _load_m38()
        if m38.LEDGER_KEY not in state.completed_effect_keys:
            return {
                "tool": "post_ledger_entry",
                "arguments": {
                    "account": m38.LEDGER_ACCOUNT,
                    "amount": self.price,
                    "memo": m38.LEDGER_MEMO,
                    "idempotency_key": m38.LEDGER_KEY,
                },
            }
        return {"done": True}


class ForcedFailurePolicy:
    """Propose an unregistered tool so the wrapped M38 machine fails closed."""

    name = "forced_failure"

    def propose(self, state: Any) -> dict[str, Any]:
        del state
        return {"tool": "not_registered", "arguments": {"sku": CATALOG_SKU}}


def _price_entry_for(retrieved: RetrievalResult, sku: str) -> MemoryEntry | None:
    for entry in retrieved.included:
        if entry.sku == sku and str(entry.key).endswith(":price") and _is_number(entry.value):
            return entry
    return None


def _wrap_m38(
    kind: str,
    *,
    approval: str | None,
    fail: bool = False,
    price_from_memory: float | None = None,
) -> Any:
    m38 = _load_m38()
    if fail:
        return m38.run_workflow(ForcedFailurePolicy(), approval=approval, max_steps=4)
    if kind == "purchase" and price_from_memory is not None:
        return m38.run_workflow(
            PostedFromMemoryPolicy(price_from_memory),
            approval=approval,
        )
    if kind == "purchase":
        return m38.run_workflow("purchase_sku7", approval=approval)
    if kind == "lookup":
        return m38.run_workflow(
            "purchase_sku7",
            approval=approval,
            interrupt_when="after_lookup",
        )
    raise ValueError(f"unknown wrap kind {kind!r}")


def _rung_kind(rung: str, route: str) -> str:
    if rung == "lookup_only":
        return "lookup"
    if route == "catalog_lookup":
        return "lookup"
    return "purchase"


def _rung_should_fail(rung: str, inject: str | None) -> bool:
    if inject == "all_failures":
        return True
    if inject == "primary_failure" and rung == "primary":
        return True
    return False


@dataclass(frozen=True)
class LadderOutcome:
    workflow: Any | None
    attempts: int
    trace: tuple[dict[str, Any], ...]
    fallbacks_used: tuple[str, ...]
    circuit: CircuitBreaker
    degraded: bool
    terminal: str
    used_memory_ids: tuple[str, ...]
    aborted_ceiling: bool
    claim: str
    posted_amount: float | None
    last_tool_result: dict[str, Any] | None
    effect_count: int


def _posted_amount(workflow: Any | None) -> float | None:
    if workflow is None:
        return None
    result = workflow.state.last_tool_result or {}
    if _is_number(result.get("amount")):
        return float(result["amount"])
    if _is_number(result.get("price")):
        return float(result["price"])
    return None


def _lookup_ok(workflow: Any) -> bool:
    result = {} if workflow is None else (workflow.state.last_tool_result or {})
    return result.get("sku") == CATALOG_SKU and _is_number(result.get("price"))


def run_fallback_ladder(
    *,
    route: str,
    retrieved: RetrievalResult,
    sku: str,
    approval: str | None = "granted",
    inject: str | None = None,
    defect: str = "none",
    max_attempts: int = MAX_ATTEMPTS,
    circuit_threshold: int = CIRCUIT_THRESHOLD,
    safety_ceiling: int = ABSOLUTE_CEILING,
) -> LadderOutcome:
    """Climb a frozen ladder. Bounds are orchestration policy, not a prompt."""

    defect_key = _normalize_defect(defect)
    oscillating = defect_key == "fallback_oscillation"
    trust_stale = defect_key == "stale_memory_trusted"
    circuit = CircuitBreaker(threshold=circuit_threshold)
    rungs = list(LADDER)
    if oscillating:
        rungs = ["primary", "lookup_only"] * (safety_ceiling + 1)
    attempts = 0
    trace: list[dict[str, Any]] = []
    fallbacks: list[str] = []
    workflow = None
    used_memory_ids: tuple[str, ...] = ()
    aborted = False
    last_ok = False
    last_degraded = False
    last_claim = "not_started"
    last_terminal = "failed"

    while True:
        if not oscillating:
            if attempts >= max_attempts:
                break
            if not circuit.allow():
                break
        elif attempts >= safety_ceiling:
            aborted = True
            break
        rung = rungs[attempts if oscillating else min(attempts, len(LADDER) - 1)]
        if not oscillating and attempts >= len(LADDER):
            break
        attempts += 1
        kind = _rung_kind(rung, route)
        fail = _rung_should_fail(rung, inject)
        price = None
        memory_ids: tuple[str, ...] = ()
        if trust_stale and kind == "purchase" and not fail:
            chosen = _price_entry_for(retrieved, sku)
            if chosen is not None:
                price = float(chosen.value)
                memory_ids = (chosen.entry_id,)
        workflow = _wrap_m38(
            kind,
            approval=approval,
            fail=fail,
            price_from_memory=price,
        )
        if kind == "lookup":
            ok = (not fail) and _lookup_ok(workflow)
            degraded = ok and route == "catalog_purchase"
            terminal = "degraded" if degraded else ("complete" if ok else "failed")
            claim = "lookup_without_post" if degraded else ("lookup" if ok else "lookup_failed")
        else:
            ok = (not fail) and workflow is not None and workflow.state.terminal == "complete"
            degraded = False
            terminal = "complete" if ok else "failed"
            claim = "purchase" if ok else "purchase_failed"
            if ok and memory_ids:
                claim = "posted_from_memory"
        if rung != "primary" and ok:
            fallbacks.append(rung)
        event = {
            "attempt": attempts,
            "rung": rung,
            "kind": kind,
            "ok": ok,
            "degraded": degraded,
            "terminal": terminal,
            "claim": claim,
            "used_memory_ids": list(memory_ids),
            "m38_terminal": None if workflow is None else workflow.state.terminal,
            "m38_node": None if workflow is None else workflow.state.node,
            "posted_amount": _posted_amount(workflow),
            "interrupted": False if workflow is None else workflow.interrupted,
        }
        trace.append(event)
        last_ok = ok
        last_degraded = degraded
        last_claim = claim
        last_terminal = terminal
        used_memory_ids = memory_ids
        if not oscillating:
            circuit = circuit.record(ok)
        if ok:
            break

    if aborted:
        last_terminal = "failed"
        last_claim = "oscillation_unbounded"
        last_degraded = False
    elif not last_ok:
        if circuit.open:
            last_terminal = "circuit_open"
            last_claim = "circuit_open"
        elif attempts >= max_attempts:
            last_terminal = "failed"
            last_claim = "attempt_bound"
        last_degraded = False

    effect_count = 0 if workflow is None else int(workflow.effect_count)
    return LadderOutcome(
        workflow=workflow,
        attempts=attempts,
        trace=tuple(trace),
        fallbacks_used=tuple(fallbacks),
        circuit=circuit,
        degraded=last_degraded,
        terminal=last_terminal,
        used_memory_ids=used_memory_ids,
        aborted_ceiling=aborted,
        claim=last_claim,
        posted_amount=_posted_amount(workflow),
        last_tool_result=None if workflow is None else _copy_mapping(workflow.state.last_tool_result),
        effect_count=effect_count,
    )


@dataclass(frozen=True)
class RobustResult:
    task: str
    route: str
    route_reason: str
    retrieved_ids: tuple[str, ...]
    excluded: tuple[dict[str, str], ...]
    used_memory_ids: tuple[str, ...]
    attempts: int
    ladder: tuple[str, ...]
    fallbacks_used: tuple[str, ...]
    circuit_open: bool
    circuit_failures: int
    degraded: bool
    terminal: str
    status: str
    claim: str
    last_tool_result: dict[str, Any] | None
    effect_count: int
    posted_amount: float | None
    workflow: Any | None
    trace: tuple[dict[str, Any], ...]
    retrieval: RetrievalResult
    defect: str
    version: str = ROBUST_VERSION
    aborted_ceiling: bool = False
    now: int = DEFAULT_NOW

    @property
    def retrieved_id_list(self) -> tuple[str, ...]:
        return self.retrieved_ids


@dataclass(frozen=True)
class FailureTrace:
    defect: str
    claim: str
    result: RobustResult
    initial_store: tuple[dict[str, Any], ...]
    initial_task: str
    initial_now: int
    inject: str | None
    audit: dict[str, Any] = field(default_factory=dict)
    version: str = ROBUST_VERSION

    @property
    def attempts(self) -> int:
        return self.result.attempts

    @property
    def terminal(self) -> str:
        return self.result.terminal

    @property
    def degraded(self) -> bool:
        return self.result.degraded

    @property
    def circuit_open(self) -> bool:
        return self.result.circuit_open

    @property
    def posted_amount(self) -> float | None:
        return self.result.posted_amount

    @property
    def used_memory_ids(self) -> tuple[str, ...]:
        return self.result.used_memory_ids

    @property
    def effect_count(self) -> int:
        return self.result.effect_count


def run_robust_task(
    task: str = DEFAULT_TASK,
    *,
    store: MemoryStore | None = None,
    now: int = DEFAULT_NOW,
    approval: str | None = "granted",
    inject: str | None = None,
    defect: str | None = "none",
    max_attempts: int = MAX_ATTEMPTS,
    circuit_threshold: int = CIRCUIT_THRESHOLD,
    safety_ceiling: int = ABSOLUTE_CEILING,
    sku: str | None = None,
) -> RobustResult:
    """Retrieve, route, wrap M38, then climb a bounded fallback ladder."""

    defect_key = _normalize_defect(defect)
    mem = store if store is not None else MemoryStore()
    decision = select_route(task)
    query = memory_query_from_task(task, route=decision.route)
    skip_expiry = defect_key == "stale_memory_trusted"
    retrieved = retrieve_memory(mem, query, now=now, skip_expiry=skip_expiry)
    target_sku = sku or extract_sku(task) or CATALOG_SKU

    if decision.route == "no_match":
        return RobustResult(
            task=task,
            route=decision.route,
            route_reason=decision.reason,
            retrieved_ids=retrieved.included_ids,
            excluded=tuple(
                {"id": item.entry_id, "reason": reason} for item, reason in retrieved.excluded
            ),
            used_memory_ids=(),
            attempts=0,
            ladder=LADDER,
            fallbacks_used=(),
            circuit_open=False,
            circuit_failures=0,
            degraded=False,
            terminal="no_match",
            status="no_match",
            claim="refused_unrouted",
            last_tool_result=None,
            effect_count=0,
            posted_amount=None,
            workflow=None,
            trace=(),
            retrieval=retrieved,
            defect=defect_key,
            aborted_ceiling=False,
            now=int(now),
        )

    outcome = run_fallback_ladder(
        route=decision.route,
        retrieved=retrieved,
        sku=target_sku,
        approval=approval,
        inject=inject,
        defect=defect_key,
        max_attempts=max_attempts,
        circuit_threshold=circuit_threshold,
        safety_ceiling=safety_ceiling,
    )
    status = outcome.terminal
    return RobustResult(
        task=task,
        route=decision.route,
        route_reason=decision.reason,
        retrieved_ids=retrieved.included_ids,
        excluded=tuple(
            {"id": item.entry_id, "reason": reason} for item, reason in retrieved.excluded
        ),
        used_memory_ids=outcome.used_memory_ids,
        attempts=outcome.attempts,
        ladder=LADDER,
        fallbacks_used=outcome.fallbacks_used,
        circuit_open=outcome.circuit.open,
        circuit_failures=outcome.circuit.consecutive_failures,
        degraded=outcome.degraded,
        terminal=outcome.terminal,
        status=status,
        claim=outcome.claim,
        last_tool_result=outcome.last_tool_result,
        effect_count=outcome.effect_count,
        posted_amount=outcome.posted_amount,
        workflow=outcome.workflow,
        trace=outcome.trace,
        retrieval=retrieved,
        defect=defect_key,
        aborted_ceiling=outcome.aborted_ceiling,
        now=int(now),
    )


def catalog_price_entry(*, now: int = DEFAULT_NOW) -> MemoryEntry:
    return MemoryEntry(
        entry_id=MEM_SKU7_PRICE,
        key="sku:SKU-7:price",
        value=CATALOG_PRICE,
        scope="catalog",
        sku=CATALOG_SKU,
        provenance=Provenance(
            source="catalog_fixture",
            writer="m39",
            written_at=int(now),
            method="imported",
        ),
        expires_at=int(now) + 5000,
        tags=("price", "catalog"),
    )


def demo_store_relevant_and_irrelevant(*, now: int = DEFAULT_NOW) -> MemoryStore:
    store = MemoryStore().put(catalog_price_entry(now=now))
    store = store.put(
        MemoryEntry(
            entry_id=MEM_BIN4_QTY,
            key="bin:BIN-4:qty",
            value=IRRELEVANT_QTY,
            scope="warehouse",
            sku=None,
            provenance=Provenance(
                source="ops_fixture",
                writer="m39",
                written_at=int(now),
                method="asserted",
            ),
            expires_at=int(now) + 5000,
            tags=("qty", "warehouse"),
        )
    )
    store = store.put(
        MemoryEntry(
            entry_id=MEM_SKU9_PRICE,
            key="sku:SKU-9:price",
            value=IRRELEVANT_SKU_PRICE,
            scope="catalog",
            sku=IRRELEVANT_SKU,
            provenance=Provenance(
                source="catalog_fixture",
                writer="m39",
                written_at=int(now),
                method="imported",
            ),
            expires_at=int(now) + 5000,
            tags=("price", "catalog"),
        )
    )
    return store


def demo_store_stale_and_fresh() -> MemoryStore:
    stale = MemoryEntry(
        entry_id=MEM_SKU7_STALE,
        key="sku:SKU-7:price",
        value=STALE_PRICE,
        scope="catalog",
        sku=CATALOG_SKU,
        provenance=Provenance(
            source="old_catalog",
            writer="m39",
            written_at=STALE_WRITTEN_AT,
            method="asserted",
        ),
        expires_at=STALE_EXPIRES_AT,
        tags=("price", "catalog"),
    )
    fresh = MemoryEntry(
        entry_id=MEM_SKU7_FRESH,
        key="sku:SKU-7:price",
        value=CATALOG_PRICE,
        scope="catalog",
        sku=CATALOG_SKU,
        provenance=Provenance(
            source="catalog_fixture",
            writer="m39",
            written_at=FRESH_WRITTEN_AT,
            method="imported",
        ),
        expires_at=FRESH_EXPIRES_AT,
        tags=("price", "catalog"),
    )
    return MemoryStore((stale, fresh))


def demo_store_superseded() -> MemoryStore:
    old = MemoryEntry(
        entry_id=MEM_SKU7_SUPERSEDED,
        key="sku:SKU-7:price",
        value=STALE_PRICE,
        scope="catalog",
        sku=CATALOG_SKU,
        provenance=Provenance(
            source="old_catalog",
            writer="m39",
            written_at=STALE_WRITTEN_AT,
            method="asserted",
        ),
        expires_at=FRESH_EXPIRES_AT,
        superseded_by=MEM_SKU7_FRESH,
        tags=("price", "catalog"),
    )
    fresh = MemoryEntry(
        entry_id=MEM_SKU7_FRESH,
        key="sku:SKU-7:price",
        value=CATALOG_PRICE,
        scope="catalog",
        sku=CATALOG_SKU,
        provenance=Provenance(
            source="catalog_fixture",
            writer="m39",
            written_at=FRESH_WRITTEN_AT,
            method="imported",
        ),
        expires_at=FRESH_EXPIRES_AT,
        tags=("price", "catalog"),
    )
    return MemoryStore((old, fresh))


def graph_public() -> dict[str, Any]:
    m38 = _load_m38()
    return {
        "version": ROBUST_VERSION,
        "workflow_version": m38.WORKFLOW_VERSION,
        "routes": list(ROUTE_NAMES),
        "ladder": list(LADDER),
        "terminals": list(TERMINALS),
        "max_attempts": MAX_ATTEMPTS,
        "circuit_threshold": CIRCUIT_THRESHOLD,
        "working_ephemeral": list(WORKING_EPHEMERAL),
        "persistent_candidates": list(PERSISTENT_CANDIDATES),
        "trace_fields": list(TRACE_FIELDS),
        "m38_state_fields": list(m38.STATE_FIELDS),
    }


def graph_fingerprint() -> str:
    payload = json.dumps(
        {
            "routes": ROUTE_NAMES,
            "ladder": LADDER,
            "terminals": TERMINALS,
            "max_attempts": MAX_ATTEMPTS,
            "circuit_threshold": CIRCUIT_THRESHOLD,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pipeline_with_defect(
    *,
    defect: str,
    now: int = DEFAULT_NOW,
    safety_ceiling: int = ABSOLUTE_CEILING,
) -> FailureTrace:
    """Controlled-failure entry: one named memory or fallback defect."""

    key = _normalize_defect(defect)
    if key == "none":
        raise ValueError("pipeline_with_defect requires a named defect")

    if key == "stale_memory_trusted":
        store = demo_store_stale_and_fresh()
        snapshot = tuple(store.as_dicts())
        result = run_robust_task(
            DEFAULT_TASK,
            store=store,
            now=now,
            defect=key,
        )
        return FailureTrace(
            defect=key,
            claim="stale_memory_posted_as_complete",
            result=result,
            initial_store=snapshot,
            initial_task=DEFAULT_TASK,
            initial_now=now,
            inject=None,
            audit={
                "posted_amount": result.posted_amount,
                "catalog_price": CATALOG_PRICE,
                "stale_price": STALE_PRICE,
                "used_memory_ids": list(result.used_memory_ids),
                "retrieved_ids": list(result.retrieved_ids),
                "degraded": result.degraded,
                "terminal": result.terminal,
                "executions": [] if result.workflow is None else list(result.workflow.session.executions),
            },
        )

    if key == "fallback_oscillation":
        store = MemoryStore().put(catalog_price_entry(now=now))
        snapshot = tuple(store.as_dicts())
        result = run_robust_task(
            DEFAULT_TASK,
            store=store,
            now=now,
            inject="all_failures",
            defect=key,
            safety_ceiling=safety_ceiling,
        )
        return FailureTrace(
            defect=key,
            claim="fallback_loop_unbounded",
            result=result,
            initial_store=snapshot,
            initial_task=DEFAULT_TASK,
            initial_now=now,
            inject="all_failures",
            audit={
                "attempts": result.attempts,
                "max_attempts": MAX_ATTEMPTS,
                "circuit_threshold": CIRCUIT_THRESHOLD,
                "circuit_open": result.circuit_open,
                "aborted_ceiling": result.aborted_ceiling,
                "safety_ceiling": safety_ceiling,
                "terminal": result.terminal,
                "rungs": [event["rung"] for event in result.trace],
            },
        )

    raise ValueError(f"unsupported defect {defect!r}")


def repair_run(trace: FailureTrace) -> FailureTrace:
    """Recompute from the defective object's initial store and task.

    Does not start a second unrelated happy-path run from module defaults.
    """

    if trace.defect not in {"stale_memory_trusted", "fallback_oscillation"}:
        raise ValueError(f"repair_run expects a named defect, not {trace.defect!r}")
    store = restore_store(trace.initial_store)
    inject = None if trace.defect == "stale_memory_trusted" else "all_failures"
    repaired = run_robust_task(
        trace.initial_task,
        store=store,
        now=trace.initial_now,
        inject=inject,
        defect="none",
        safety_ceiling=int(trace.audit.get("safety_ceiling") or ABSOLUTE_CEILING),
    )
    claim = (
        "expiry_enforced"
        if trace.defect == "stale_memory_trusted"
        else "circuit_and_attempt_bound_restored"
    )
    return FailureTrace(
        defect="none",
        claim=claim,
        result=repaired,
        initial_store=trace.initial_store,
        initial_task=trace.initial_task,
        initial_now=trace.initial_now,
        inject=inject,
        audit={
            "from_defect": trace.defect,
            "terminal": repaired.terminal,
            "attempts": repaired.attempts,
            "degraded": repaired.degraded,
            "posted_amount": repaired.posted_amount,
            "circuit_open": repaired.circuit_open,
            "used_memory_ids": list(repaired.used_memory_ids),
        },
    )


def observability_report(result: RobustResult) -> dict[str, Any]:
    inference = {}
    if result.workflow is not None:
        inference = dict(result.workflow.state.inference or {})
    return {
        "version": ROBUST_VERSION,
        "route": result.route,
        "terminal": result.terminal,
        "status": result.status,
        "attempts": result.attempts,
        "degraded": result.degraded,
        "circuit_open": result.circuit_open,
        "retrieved_ids": list(result.retrieved_ids),
        "excluded": list(result.excluded),
        "used_memory_ids": list(result.used_memory_ids),
        "fallbacks_used": list(result.fallbacks_used),
        "posted_amount": result.posted_amount,
        "effect_count": result.effect_count,
        "claim": result.claim,
        "trace": list(result.trace),
        "trace_fields": list(TRACE_FIELDS),
        "weights_updated": inference.get("weights_updated", False),
        "training_time": inference.get("training_time", False),
        "scale_limit": SCALE_LIMIT,
        "handoff": HANDOFF,
        "field_classification": FIELD_CLASSIFICATION,
    }


def handoff_contract() -> dict[str, Any]:
    m38 = _load_m38()
    return {
        "version": ROBUST_VERSION,
        "workflow_version": m38.WORKFLOW_VERSION,
        "memory": {
            "working_ephemeral": list(WORKING_EPHEMERAL),
            "persistent_candidates": list(PERSISTENT_CANDIDATES),
            "entry_fields": [
                "entry_id",
                "key",
                "value",
                "scope",
                "sku",
                "provenance",
                "expires_at",
                "superseded_by",
            ],
        },
        "routes": list(ROUTE_NAMES),
        "ladder": list(LADDER),
        "max_attempts": MAX_ATTEMPTS,
        "circuit_threshold": CIRCUIT_THRESHOLD,
        "terminals": list(TERMINALS),
        "trace_fields": list(TRACE_FIELDS),
        "eval_harness": "deferred to M40",
        "handoff": HANDOFF,
    }


def numpy_terminal_counts(results: list[RobustResult] | tuple[RobustResult, ...]):
    """Optional NumPy parity helper. Required tests use stdlib counts."""

    np = _require_numpy()
    labels = list(TERMINALS)
    observed = [item.terminal for item in results]
    counts = np.array([observed.count(name) for name in labels], dtype=int)
    return labels, counts


def optional_live_retrieve(*args: Any, **kwargs: Any) -> Any:
    """Optional live-model memory adapter. Canonical validation uses local fixtures."""

    del args, kwargs
    raise LiveAdapterUnavailable(
        "M39 canonical path uses a local provenance-bearing memory store; "
        "live models are optional and not required for validation"
    )


def optional_langgraph_store(*args: Any, **kwargs: Any) -> Any:
    """Optional LangGraph store adapter. Canonical validation must not load that SDK.

    Named source: langgraph-docs. A LangGraph adapter is optional and not required.
    """

    del args, kwargs
    raise OptionalLangGraphUnavailable(
        "M39 canonical path is the local memory/router/fallback layer wrapping "
        "M38; a LangGraph store adapter is optional and not required for validation"
    )
