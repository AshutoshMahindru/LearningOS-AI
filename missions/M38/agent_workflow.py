"""Explicit stateful workflow around M37 model-call fixtures and tools.

M38 consumes M37's validated tool runtime (registry, schema errors,
idempotency, staged traces) and opens a teaching state machine:

explicit AgentState → nodes → allowed transitions → terminals
  → checkpoint / resume → human approval → loop bound.

Canonical proposals are local deterministic fixtures, not a live model.
The required path is this reference machine. A LangGraph adapter, if
present, is optional and fail-closed. Persistent memory, routing, and
fallback ladders stay M39; eval stays M40; RAG/ANN stay M34–M36;
sampling labs stay M32.

Named sources (not SDKs): langgraph-docs, anthropic-agents.
Content bundle: tool-using-agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SEED = 3801
WORKFLOW_VERSION = "v10-teaching-workflow-1"
GRAPH_ID = "v10-teaching-graph-1"
MAX_STEPS = 8
ABSOLUTE_CEILING = 32
CATALOG_SKU = "SKU-7"
CATALOG_PRICE = 42.0
LEDGER_KEY = "sku7-price-post"
LEDGER_ACCOUNT = "cash"
LEDGER_MEMO = "sku-7-price"
DEFAULT_TASK_ID = "sku7-purchase"
DEFAULT_GOAL = "Look up SKU-7 and post its catalog price to the cash ledger."

# Fresh no-AI numbers. Do not print derived conclusions in the gate.
TRANSFER_BIN = "BIN-4"
TRANSFER_QTY = 9
TRANSFER_MAX_STEPS = 5
TRANSFER_KEY = "reserve-bin-4"

SUPPORTED_DEFECTS = ("none", "infinite_loop", "replayed_side_effect")

NODES = (
    "start",
    "decide",
    "validate",
    "approve",
    "execute",
    "assimilate",
    "complete",
    "denied",
    "failed",
    "loop_exhausted",
)
TERMINALS = frozenset({"complete", "denied", "failed", "loop_exhausted"})
ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "start": ("decide",),
    "decide": ("validate", "complete", "loop_exhausted", "failed"),
    "validate": ("approve", "execute", "failed"),
    "approve": ("execute", "denied"),
    "execute": ("assimilate", "failed"),
    "assimilate": ("decide",),
}
TERMINAL_STATUS = {
    "complete": "succeeded",
    "denied": "denied",
    "failed": "failed",
    "loop_exhausted": "loop_exhausted",
}

STATE_FIELDS = (
    "task_id",
    "goal",
    "node",
    "step",
    "max_steps",
    "model_turn",
    "pending_action",
    "last_tool_result",
    "observations",
    "approval",
    "completed_effect_keys",
    "ledger_entries",
    "executions",
    "terminal",
    "status",
    "history",
    "checkpoint_seq",
    "defect",
    "version",
    "inference",
)

SCALE_LIMIT = (
    "Teaching scale: one local state machine, deterministic model-call "
    "fixtures, M37 tools, CPU stdlib orchestration, no paid API, no "
    "LangGraph/Anthropic SDK, no memory store, no router, no fallback "
    "ladder, no RAG pack, no Qdrant, no sampling lab. Loop bounds and "
    "invalid transitions are explicit. This fixture is not a production "
    "agent runtime."
)

SYSTEM_MAP = (
    "goal\n"
    "  -> explicit AgentState (serializable; not chat history)\n"
    "  -> start\n"
    "  -> decide   (deterministic model fixture proposes next action)\n"
    "  -> validate (M37 schema; invalid -> failed)\n"
    "  -> approve  (human gate for side-effecting tools)\n"
    "  -> execute  (M37 run_tool_call; idempotency composed)\n"
    "  -> assimilate (last_tool_result + completed_effect_keys)\n"
    "  -> decide   or complete\n"
    "  terminals: complete | denied | failed | loop_exhausted\n"
    "  checkpoint after a finished node, never mid-execute"
)

HANDOFF = (
    "M39 receives a stateful workflow with an explicit serializable "
    "state schema and failure/termination semantics (nodes, allowed "
    "transitions, loop bounds, approval, checkpoint/resume without "
    "replaying completed side effects). Persistent memory, "
    "cross-strategy routing, and fallback ladders stay M39. M38 does not "
    "open them."
)


class InvalidTransition(ValueError):
    """A destination is not an allowed edge. State must not mutate."""

    def __init__(self, src: str, dest: str, allowed: tuple[str, ...]) -> None:
        super().__init__(
            f"invalid transition {src!r} -> {dest!r}; allowed={allowed}"
        )
        self.src = src
        self.dest = dest
        self.allowed = allowed


class OptionalLangGraphUnavailable(RuntimeError):
    """Raised when a LangGraph adapter is requested; canonical tests must not need one."""


class LiveAdapterUnavailable(RuntimeError):
    """Raised when a live model is requested; canonical tests must not need one."""


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M38 optional NumPy path requires requirements/m38.txt") from exc
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


def _load_m37():
    """Load M37's tool runtime. Prefer the package so class identity matches.

    File-load is only a fallback and still registers the package name.
    """

    packaged_name = "missions.M37.tool_runtime"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    try:
        return __import__(packaged_name, fromlist=["run_tool_call"])
    except ImportError:
        return _load_sibling("M37", "tool_runtime.py", packaged_name)


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


@dataclass(frozen=True)
class Transition:
    src: str
    dest: str
    step: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dest": self.dest, "step": self.step, "reason": self.reason}


@dataclass(frozen=True)
class AgentState:
    """Serializable workflow state. Design this before adding edges."""

    task_id: str
    goal: str
    node: str = "start"
    step: int = 0
    max_steps: int = MAX_STEPS
    model_turn: int = 0
    pending_action: dict[str, Any] | None = None
    last_tool_result: dict[str, Any] | None = None
    observations: tuple[dict[str, Any], ...] = ()
    approval: str | None = None
    completed_effect_keys: tuple[str, ...] = ()
    ledger_entries: tuple[Any, ...] = ()
    executions: tuple[str, ...] = ()
    terminal: str | None = None
    status: str = "running"
    history: tuple[Transition, ...] = ()
    checkpoint_seq: int = 0
    defect: str = "none"
    version: str = WORKFLOW_VERSION
    inference: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "node": self.node,
            "step": self.step,
            "max_steps": self.max_steps,
            "model_turn": self.model_turn,
            "pending_action": _copy_mapping(self.pending_action),
            "last_tool_result": _copy_mapping(self.last_tool_result),
            "observations": [_copy_mapping(item) for item in self.observations],
            "approval": self.approval,
            "completed_effect_keys": list(self.completed_effect_keys),
            "ledger_entries": [_entry_dict(entry) for entry in self.ledger_entries],
            "executions": list(self.executions),
            "terminal": self.terminal,
            "status": self.status,
            "history": [item.as_dict() for item in self.history],
            "checkpoint_seq": self.checkpoint_seq,
            "defect": self.defect,
            "version": self.version,
            "inference": dict(self.inference),
        }


@dataclass(frozen=True)
class WorkflowResult:
    state: AgentState
    session: Any
    interrupted: bool = False
    aborted_ceiling: bool = False
    proposals: tuple[dict[str, Any], ...] = ()
    checkpoints: tuple[dict[str, Any], ...] = ()

    @property
    def effect_count(self) -> int:
        return int(self.session.ledger.effect_count)

    @property
    def current_node(self) -> str:
        return self.state.node


@dataclass(frozen=True)
class FailureTrace:
    defect: str
    claim: str
    state: AgentState
    initial_checkpoint: dict[str, Any]
    effect_count: int
    model_turn: int
    terminal: str | None
    node: str
    last_tool_result: dict[str, Any] | None
    loop_bound_enforced: bool
    idempotency_consulted: bool
    last_tool_result_preserved: bool
    session: Any = None
    audit: dict[str, Any] = field(default_factory=dict)
    version: str = WORKFLOW_VERSION


class _Context:
    def __init__(
        self,
        *,
        policy: Any,
        approval: str | None,
        defect: str,
        session: Any,
        registry: Any,
        m37: Any,
    ) -> None:
        self.policy = policy
        self.approval = approval
        self.defect = defect
        self.session = session
        self.registry = registry
        self.m37 = m37
        self.proposals: list[dict[str, Any]] = []


class PurchaseSku7Policy:
    """Lookup SKU-7, then post the assimilated price. Model fixture, not an LLM."""

    name = "purchase_sku7"

    def propose(self, state: AgentState) -> dict[str, Any]:
        price = _lookup_price(state)
        if price is None:
            return {"tool": "lookup_catalog_price", "arguments": {"sku": CATALOG_SKU}}
        if LEDGER_KEY not in state.completed_effect_keys:
            return {
                "tool": "post_ledger_entry",
                "arguments": {
                    "account": LEDGER_ACCOUNT,
                    "amount": price,
                    "memo": LEDGER_MEMO,
                    "idempotency_key": LEDGER_KEY,
                },
            }
        return {"done": True}


class UnresolvedLookupPolicy:
    """Always requests another catalog lookup. Used to exercise the loop bound."""

    name = "unresolved_lookup"

    def propose(self, state: AgentState) -> dict[str, Any]:
        del state
        return {"tool": "lookup_catalog_price", "arguments": {"sku": CATALOG_SKU}}


class ReplayPolicy:
    """Replay recorded proposals. Deterministic components only; no model."""

    name = "replay"

    def __init__(self, proposals: tuple[Mapping[str, Any], ...]) -> None:
        self._proposals = [dict(item) for item in proposals]
        self._index = 0

    def propose(self, state: AgentState) -> dict[str, Any]:
        del state
        if self._index >= len(self._proposals):
            return {"done": True}
        item = dict(self._proposals[self._index])
        self._index += 1
        return item


POLICIES = {
    "purchase_sku7": PurchaseSku7Policy,
    "unresolved_lookup": UnresolvedLookupPolicy,
}

RECORDED_PURCHASE_PROPOSALS: tuple[dict[str, Any], ...] = (
    {"tool": "lookup_catalog_price", "arguments": {"sku": CATALOG_SKU}},
    {
        "tool": "post_ledger_entry",
        "arguments": {
            "account": LEDGER_ACCOUNT,
            "amount": CATALOG_PRICE,
            "memo": LEDGER_MEMO,
            "idempotency_key": LEDGER_KEY,
        },
    },
    {"done": True},
)


def _coerce_policy(policy: str | Any) -> Any:
    if not isinstance(policy, str):
        return policy
    if policy == "recorded_purchase":
        return ReplayPolicy(RECORDED_PURCHASE_PROPOSALS)
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    return POLICIES[policy]()


def _lookup_price(state: AgentState) -> float | None:
    result = state.last_tool_result or {}
    if result.get("sku") == CATALOG_SKU and _is_number(result.get("price")):
        return float(result["price"])
    for obs in state.observations:
        output = obs.get("output") or {}
        if (
            obs.get("tool") == "lookup_catalog_price"
            and obs.get("status") == "success"
            and output.get("sku") == CATALOG_SKU
            and _is_number(output.get("price"))
        ):
            return float(output["price"])
    return None


def _entry_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        return {
            "entry_id": entry.get("entry_id"),
            "account": entry.get("account"),
            "amount": entry.get("amount"),
            "memo": entry.get("memo"),
            "idempotency_key": entry.get("idempotency_key"),
        }
    return {
        "entry_id": getattr(entry, "entry_id", None),
        "account": getattr(entry, "account", None),
        "amount": getattr(entry, "amount", None),
        "memo": getattr(entry, "memo", None),
        "idempotency_key": getattr(entry, "idempotency_key", None),
    }


def _entries_from_payload(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    m37 = _load_m37()
    rows = payload.get("ledger_entries") or ()
    entries = []
    for row in rows:
        if isinstance(row, m37.LedgerEntry):
            entries.append(row)
            continue
        data = _entry_dict(row)
        entries.append(
            m37.LedgerEntry(
                entry_id=int(data["entry_id"]),
                account=data["account"],
                amount=data["amount"],
                memo=data["memo"],
                idempotency_key=data["idempotency_key"],
            )
        )
    return tuple(entries)


def graph_public() -> dict[str, Any]:
    return {
        "graph_id": GRAPH_ID,
        "version": WORKFLOW_VERSION,
        "nodes": NODES,
        "terminals": tuple(sorted(TERMINALS)),
        "edges": {src: list(dests) for src, dests in ALLOWED_TRANSITIONS.items()},
        "state_fields": STATE_FIELDS,
        "max_steps": MAX_STEPS,
    }


def graph_fingerprint() -> str:
    payload = json.dumps(
        {
            "nodes": NODES,
            "terminals": sorted(TERMINALS),
            "edges": ALLOWED_TRANSITIONS,
            "state_fields": STATE_FIELDS,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_transition(state: AgentState, dest: str, *, reason: str = "") -> AgentState:
    """Move along an allowed edge. Illegal destinations raise and do not mutate."""

    allowed = ALLOWED_TRANSITIONS.get(state.node, ())
    if dest not in allowed:
        raise InvalidTransition(state.node, dest, allowed)
    terminal = dest if dest in TERMINALS else None
    status = TERMINAL_STATUS[dest] if terminal else "running"
    record = Transition(src=state.node, dest=dest, step=state.step + 1, reason=reason)
    return replace(
        state,
        node=dest,
        step=state.step + 1,
        terminal=terminal,
        status=status,
        history=state.history + (record,),
    )


def initial_state(
    *,
    task_id: str = DEFAULT_TASK_ID,
    goal: str = DEFAULT_GOAL,
    max_steps: int = MAX_STEPS,
    defect: str | None = "none",
) -> AgentState:
    m37 = _load_m37()
    inference = m37.attach_inference_evidence()
    return AgentState(
        task_id=task_id,
        goal=goal,
        max_steps=max(1, int(max_steps)),
        defect=_normalize_defect(defect),
        inference=dict(inference),
    )


def make_session(ledger_entries: tuple[Any, ...] = ()):
    m37 = _load_m37()
    ledger = m37.LedgerState.from_snapshot(ledger_entries)
    return m37.RuntimeSession(ledger)


def checkpoint(state: AgentState, session: Any | None = None) -> dict[str, Any]:
    """JSON-serializable snapshot at a node boundary.

    Safe boundary: after assimilate (side effects and last_tool_result recorded).
    Resume must restore last_tool_result, completed_effect_keys, and ledger.
    """

    payload = state.as_dict()
    payload["schema_version"] = WORKFLOW_VERSION
    payload["graph_id"] = GRAPH_ID
    payload["graph_fingerprint"] = graph_fingerprint()
    if session is not None:
        payload["ledger_entries"] = [_entry_dict(entry) for entry in session.ledger.snapshot()]
        payload["executions"] = list(session.executions)
        payload["effect_count"] = int(session.ledger.effect_count)
    json.dumps(payload)
    return payload


def restore(payload: Mapping[str, Any], *, session: Any | None = None) -> tuple[AgentState, Any]:
    entries = _entries_from_payload(payload)
    sess = session if session is not None else make_session(entries)
    if session is None:
        sess.executions = list(payload.get("executions") or [])
    history = tuple(
        Transition(
            src=str(item["src"]),
            dest=str(item["dest"]),
            step=int(item["step"]),
            reason=str(item.get("reason") or ""),
        )
        for item in (payload.get("history") or ())
    )
    observations = tuple(dict(item) for item in (payload.get("observations") or ()))
    state = AgentState(
        task_id=str(payload.get("task_id") or DEFAULT_TASK_ID),
        goal=str(payload.get("goal") or DEFAULT_GOAL),
        node=str(payload.get("node") or "start"),
        step=int(payload.get("step") or 0),
        max_steps=max(1, int(payload.get("max_steps") or MAX_STEPS)),
        model_turn=int(payload.get("model_turn") or 0),
        pending_action=_copy_mapping(payload.get("pending_action")),
        last_tool_result=_copy_mapping(payload.get("last_tool_result")),
        observations=observations,
        approval=payload.get("approval"),
        completed_effect_keys=tuple(str(key) for key in (payload.get("completed_effect_keys") or ())),
        ledger_entries=entries,
        executions=tuple(str(item) for item in (payload.get("executions") or ())),
        terminal=payload.get("terminal"),
        status=str(payload.get("status") or "running"),
        history=history,
        checkpoint_seq=int(payload.get("checkpoint_seq") or 0),
        defect=_normalize_defect(payload.get("defect")),
        version=str(payload.get("version") or WORKFLOW_VERSION),
        inference=dict(payload.get("inference") or {}),
    )
    return state, sess


def grant_approval(state: AgentState) -> AgentState:
    if state.node != "approve":
        raise InvalidTransition(state.node, "execute", ALLOWED_TRANSITIONS.get(state.node, ()))
    return apply_transition(replace(state, approval="granted"), "execute", reason="approved")


def deny_approval(state: AgentState) -> AgentState:
    if state.node != "approve":
        raise InvalidTransition(state.node, "denied", ALLOWED_TRANSITIONS.get(state.node, ()))
    return apply_transition(replace(state, approval="denied"), "denied", reason="denied")


def _requires_approval(action: Mapping[str, Any] | None, registry: Any) -> bool:
    name = None if action is None else action.get("tool")
    if not name:
        return False
    try:
        return bool(registry.get(str(name)).requires_approval)
    except KeyError:
        return False


def _sync_session(state: AgentState, ctx: _Context) -> AgentState:
    return replace(
        state,
        executions=tuple(ctx.session.executions),
        ledger_entries=ctx.session.ledger.snapshot(),
    )


def _node_start(state: AgentState, ctx: _Context) -> AgentState:
    del ctx
    return apply_transition(state, "decide", reason="start")


def _node_decide(state: AgentState, ctx: _Context) -> AgentState:
    if ctx.defect != "infinite_loop" and state.model_turn >= state.max_steps:
        return apply_transition(state, "loop_exhausted", reason="max_steps")
    action = dict(ctx.policy.propose(state))
    ctx.proposals.append(dict(action))
    new = replace(state, pending_action=action, model_turn=state.model_turn + 1)
    if action.get("done") is True:
        return apply_transition(new, "complete", reason="model_done")
    if not action.get("tool"):
        return apply_transition(new, "failed", reason="no_action")
    return apply_transition(new, "validate", reason="model_proposed_tool")


def _node_validate(state: AgentState, ctx: _Context) -> AgentState:
    m37 = ctx.m37
    try:
        proposal = m37.parse_proposal(state.pending_action)
    except m37.SchemaError:
        return apply_transition(state, "failed", reason="parse")
    validation = m37.validate_proposal(proposal, ctx.registry)
    if not validation.ok:
        return apply_transition(state, "failed", reason="schema")
    if _requires_approval(state.pending_action, ctx.registry) and state.approval != "granted":
        return apply_transition(state, "approve", reason="needs_approval")
    return apply_transition(state, "execute", reason="validated")


def _node_approve(state: AgentState, ctx: _Context) -> AgentState:
    if ctx.approval == "granted":
        return grant_approval(state)
    if ctx.approval == "denied":
        return deny_approval(state)
    return state


def _node_execute(state: AgentState, ctx: _Context) -> AgentState:
    m37 = ctx.m37
    action = dict(state.pending_action or {})
    arguments = dict(action.get("arguments") or {})
    key = arguments.get("idempotency_key")
    skip_idemp = ctx.defect == "replayed_side_effect"
    if (
        not skip_idemp
        and key is not None
        and str(key) in state.completed_effect_keys
    ):
        synced = _sync_session(state, ctx)
        return apply_transition(synced, "assimilate", reason="skipped_completed_effect")
    spec_needs_approval = _requires_approval(action, ctx.registry)
    approved = (state.approval == "granted") or not spec_needs_approval
    trace = m37.run_tool_call(
        action,
        registry=ctx.registry,
        session=ctx.session,
        approved=approved,
        skip_idempotency=skip_idemp,
        defect="none",
    )
    output = None if trace.result.output is None else dict(trace.result.output)
    new = replace(
        _sync_session(state, ctx),
        last_tool_result=output,
    )
    if trace.result.status == "success":
        return apply_transition(new, "assimilate", reason="executed")
    return apply_transition(new, "failed", reason=f"tool_{trace.result.error_kind or trace.result.status}")


def _node_assimilate(state: AgentState, ctx: _Context) -> AgentState:
    del ctx
    action = dict(state.pending_action or {})
    key = (action.get("arguments") or {}).get("idempotency_key")
    keys = state.completed_effect_keys
    if key is not None and str(key) not in keys:
        keys = keys + (str(key),)
    observation = {
        "tool": action.get("tool"),
        "status": "success",
        "output": _copy_mapping(state.last_tool_result) or {},
    }
    new = replace(
        state,
        observations=state.observations + (observation,),
        completed_effect_keys=keys,
        pending_action=None,
    )
    return apply_transition(new, "decide", reason="assimilated")


DISPATCH = {
    "start": _node_start,
    "decide": _node_decide,
    "validate": _node_validate,
    "approve": _node_approve,
    "execute": _node_execute,
    "assimilate": _node_assimilate,
}


def _interrupt_before(state: AgentState, interrupt_when: str | None) -> bool:
    if interrupt_when == "at_post_execute":
        return state.node == "execute" and (state.pending_action or {}).get("tool") == "post_ledger_entry"
    if interrupt_when == "at_approve":
        return state.node == "approve"
    return False


def _interrupt_after(before: str, state: AgentState, interrupt_when: str | None) -> bool:
    if interrupt_when == "after_lookup" and before == "assimilate":
        tools = [obs.get("tool") for obs in state.observations]
        return tools == ["lookup_catalog_price"]
    if interrupt_when == "after_post" and before == "assimilate":
        return LEDGER_KEY in state.completed_effect_keys
    return False


def run_workflow(
    policy: str | Any = "purchase_sku7",
    *,
    approval: str | None = "granted",
    max_steps: int = MAX_STEPS,
    interrupt_when: str | None = None,
    session: Any | None = None,
    state: AgentState | None = None,
    defect: str | None = "none",
    safety_ceiling: int | None = None,
    registry: Any | None = None,
) -> WorkflowResult:
    """Drive the reference machine. Model fixtures propose; orchestration decides.

    Invalid calls never reach M37 execute unless a named defect rewinds an
    execute node. Live adapters and LangGraph fail closed.
    """

    defect_key = _normalize_defect(defect)
    m37 = _load_m37()
    current = state if state is not None else initial_state(max_steps=max_steps, defect=defect_key)
    if state is None:
        current = replace(current, max_steps=max(1, int(max_steps)), defect=defect_key)
    sess = session if session is not None else m37.RuntimeSession()
    ctx = _Context(
        policy=_coerce_policy(policy),
        approval=approval,
        defect=defect_key,
        session=sess,
        registry=registry if registry is not None else m37.default_registry(),
        m37=m37,
    )
    ceiling = (
        int(safety_ceiling)
        if safety_ceiling is not None
        else max(ABSOLUTE_CEILING, int(current.max_steps) * 8)
    )
    interrupted = False
    aborted = False
    visits = 0
    snapshots: list[dict[str, Any]] = []
    while current.node not in TERMINALS:
        visits += 1
        if visits > ceiling:
            aborted = True
            break
        if _interrupt_before(current, interrupt_when):
            interrupted = True
            break
        handler = DISPATCH.get(current.node)
        if handler is None:
            raise InvalidTransition(current.node, "decide", ())
        before = current.node
        nxt = handler(current, ctx)
        if nxt.node == before and before == "approve":
            interrupted = True
            current = nxt
            break
        current = nxt
        if _interrupt_after(before, current, interrupt_when):
            interrupted = True
            break
    if interrupted or aborted:
        payload = checkpoint(current, ctx.session)
        payload["checkpoint_seq"] = current.checkpoint_seq + 1
        current = replace(current, checkpoint_seq=current.checkpoint_seq + 1)
        snapshots.append(payload)
    return WorkflowResult(
        state=current,
        session=ctx.session,
        interrupted=interrupted,
        aborted_ceiling=aborted,
        proposals=tuple(ctx.proposals),
        checkpoints=tuple(snapshots),
    )


def resume(
    payload: Mapping[str, Any],
    *,
    approval: str | None = "granted",
    policy: str | Any = "purchase_sku7",
    defect: str | None = "none",
    interrupt_when: str | None = None,
    safety_ceiling: int | None = None,
) -> WorkflowResult:
    """Continue from a checkpoint without replaying completed side effects."""

    state, session = restore(payload)
    return run_workflow(
        policy=policy,
        approval=approval,
        max_steps=state.max_steps,
        interrupt_when=interrupt_when,
        session=session,
        state=state,
        defect=defect,
        safety_ceiling=safety_ceiling,
    )


def replay_trace(
    proposals: tuple[Mapping[str, Any], ...] | None = None,
    *,
    approval: str | None = "granted",
    max_steps: int = MAX_STEPS,
) -> WorkflowResult:
    """Replay a recorded proposal list through the deterministic machine."""

    recorded = RECORDED_PURCHASE_PROPOSALS if proposals is None else tuple(dict(item) for item in proposals)
    return run_workflow(
        ReplayPolicy(recorded),
        approval=approval,
        max_steps=max_steps,
    )


def optional_langgraph_compile(*args: Any, **kwargs: Any) -> Any:
    """Optional LangGraph adapter. Canonical validation must not load that SDK.

    Named source: langgraph-docs. A LangGraph adapter is optional and not required.
    """

    del args, kwargs
    raise OptionalLangGraphUnavailable(
        "M38 canonical path is the local deterministic state machine; "
        "a LangGraph adapter is optional and not required for validation"
    )


def optional_live_propose(intent: str, config: Any = None) -> dict[str, Any]:
    """Optional live-model adapter. Canonical validation uses local fixtures."""

    del intent, config
    raise LiveAdapterUnavailable(
        "M38 canonical path uses local model-call fixtures; live models are "
        "optional and not required for validation"
    )


def numpy_node_counts(history: tuple[Transition, ...] | list[Transition]):
    """Optional NumPy parity helper. Required tests use stdlib counts."""

    np = _require_numpy()
    sources = [item.src for item in history]
    counts = np.array([sources.count(node) for node in NODES], dtype=int)
    return NODES, counts


def pipeline_with_defect(
    *,
    defect: str,
    max_steps: int = 3,
    safety_ceiling: int = 20,
) -> FailureTrace:
    """Controlled-failure entry: one named orchestration defect."""

    key = _normalize_defect(defect)
    if key == "none":
        raise ValueError("pipeline_with_defect requires a named defect")
    m37 = _load_m37()
    init = initial_state(max_steps=max_steps, defect=key)
    session = m37.RuntimeSession()
    initial_payload = checkpoint(init, session)

    if key == "infinite_loop":
        result = run_workflow(
            "unresolved_lookup",
            approval="granted",
            max_steps=max_steps,
            session=session,
            state=init,
            defect=key,
            safety_ceiling=safety_ceiling,
        )
        return FailureTrace(
            defect=key,
            claim="loop_bound_not_enforced",
            state=result.state,
            initial_checkpoint=initial_payload,
            effect_count=result.effect_count,
            model_turn=result.state.model_turn,
            terminal=result.state.terminal,
            node=result.state.node,
            last_tool_result=_copy_mapping(result.state.last_tool_result),
            loop_bound_enforced=False,
            idempotency_consulted=True,
            last_tool_result_preserved=result.state.last_tool_result is not None,
            session=result.session,
            audit={
                "max_steps": max_steps,
                "safety_ceiling": safety_ceiling,
                "aborted_ceiling": result.aborted_ceiling,
                "step": result.state.step,
                "terminal_is_loop_exhausted": result.state.terminal == "loop_exhausted",
                "policy": "unresolved_lookup",
            },
        )

    if key == "replayed_side_effect":
        driven = run_workflow(
            "purchase_sku7",
            approval="granted",
            max_steps=MAX_STEPS,
            session=session,
            state=init,
            defect="none",
            interrupt_when="at_post_execute",
        )
        ctx = _Context(
            policy=PurchaseSku7Policy(),
            approval="granted",
            defect=key,
            session=driven.session,
            registry=m37.default_registry(),
            m37=m37,
        )
        first = _node_execute(driven.state, ctx)
        first_id = None if first.last_tool_result is None else first.last_tool_result.get("entry_id")
        rewound = replace(first, node="execute", terminal=None, status="running")
        second = _node_execute(rewound, ctx)
        second_id = None if second.last_tool_result is None else second.last_tool_result.get("entry_id")
        return FailureTrace(
            defect=key,
            claim="execute_node_replayed_without_idempotency",
            state=second,
            initial_checkpoint=initial_payload,
            effect_count=driven.session.ledger.effect_count,
            model_turn=second.model_turn,
            terminal=second.terminal,
            node=second.node,
            last_tool_result=_copy_mapping(second.last_tool_result),
            loop_bound_enforced=True,
            idempotency_consulted=False,
            last_tool_result_preserved=True,
            session=driven.session,
            audit={
                "first_entry_id": first_id,
                "second_entry_id": second_id,
                "rewound_from": first.node,
                "rewound_to": "execute",
                "driven_node": driven.state.node,
                "effect_count": driven.session.ledger.effect_count,
            },
        )

    raise ValueError(f"unsupported defect {defect!r}")


def repair_run(trace: FailureTrace) -> FailureTrace:
    """Recompute from the defective object's initial checkpoint.

    Does not start a second unrelated happy-path run from module defaults.
    """

    if trace.defect == "infinite_loop":
        state, session = restore(trace.initial_checkpoint)
        max_steps = int(trace.audit.get("max_steps") or state.max_steps)
        repaired = run_workflow(
            "unresolved_lookup",
            approval="granted",
            max_steps=max_steps,
            session=session,
            state=replace(state, defect="none", max_steps=max_steps),
            defect="none",
        )
        return FailureTrace(
            defect="none",
            claim="loop_bound_restored",
            state=repaired.state,
            initial_checkpoint=trace.initial_checkpoint,
            effect_count=repaired.effect_count,
            model_turn=repaired.state.model_turn,
            terminal=repaired.state.terminal,
            node=repaired.state.node,
            last_tool_result=_copy_mapping(repaired.state.last_tool_result),
            loop_bound_enforced=True,
            idempotency_consulted=True,
            last_tool_result_preserved=True,
            session=repaired.session,
            audit={
                "from_defect": trace.defect,
                "max_steps": max_steps,
                "terminal": repaired.state.terminal,
                "aborted_ceiling": repaired.aborted_ceiling,
            },
        )
    if trace.defect == "replayed_side_effect":
        state, session = restore(trace.initial_checkpoint)
        repaired = run_workflow(
            "purchase_sku7",
            approval="granted",
            max_steps=state.max_steps,
            session=session,
            state=replace(state, defect="none"),
            defect="none",
        )
        return FailureTrace(
            defect="none",
            claim="idempotent_resume_restored",
            state=repaired.state,
            initial_checkpoint=trace.initial_checkpoint,
            effect_count=repaired.effect_count,
            model_turn=repaired.state.model_turn,
            terminal=repaired.state.terminal,
            node=repaired.state.node,
            last_tool_result=_copy_mapping(repaired.state.last_tool_result),
            loop_bound_enforced=True,
            idempotency_consulted=True,
            last_tool_result_preserved=repaired.state.last_tool_result is not None,
            session=repaired.session,
            audit={
                "from_defect": trace.defect,
                "effect_count": repaired.effect_count,
                "terminal": repaired.state.terminal,
                "completed_keys": list(repaired.state.completed_effect_keys),
            },
        )
    raise ValueError(f"repair_run expects a named defect, not {trace.defect!r}")


def observability_report(result: WorkflowResult) -> dict[str, Any]:
    return {
        "version": WORKFLOW_VERSION,
        "graph_id": GRAPH_ID,
        "node": result.state.node,
        "terminal": result.state.terminal,
        "status": result.state.status,
        "model_turn": result.state.model_turn,
        "max_steps": result.state.max_steps,
        "step": result.state.step,
        "effect_count": result.effect_count,
        "executions": list(result.session.executions),
        "last_tool_result": _copy_mapping(result.state.last_tool_result),
        "completed_effect_keys": list(result.state.completed_effect_keys),
        "history": [item.as_dict() for item in result.state.history],
        "interrupted": result.interrupted,
        "aborted_ceiling": result.aborted_ceiling,
        "weights_updated": False,
        "training_time": result.state.inference.get("training_time", False),
        "state_fields": STATE_FIELDS,
        "scale_limit": SCALE_LIMIT,
        "handoff": HANDOFF,
    }


def handoff_contract() -> dict[str, Any]:
    return {
        "graph_id": GRAPH_ID,
        "workflow_version": WORKFLOW_VERSION,
        "nodes": NODES,
        "terminals": tuple(sorted(TERMINALS)),
        "state_fields": STATE_FIELDS,
        "max_steps": MAX_STEPS,
        "tools": _load_m37().default_registry().names(),
        "checkpoint": (
            "serialize node, last_tool_result, completed_effect_keys, "
            "ledger snapshot, model_turn; resume without replaying completed effects"
        ),
        "approval": "side-effecting tools pause at the approve node",
        "loop_limit": MAX_STEPS,
        "memory_routing": "deferred to M39",
        "eval_harness": "deferred to M40",
        "handoff": HANDOFF,
    }
