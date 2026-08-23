"""Tool schemas, validation, and bounded execution for Mission M37.

M37 consumes M32's inference-provider contract (``InferenceConfig``,
``training_time=False``, ``weights_updated=False``, fail-closed live
adapter) and opens a teaching tool runtime:

declared intent → model-call fixture → parse → select → validate →
permission / idempotency → execute → structured result + staged trace.

Canonical proposals are local deterministic fixtures, not a live model.
Persistent multi-step state machines stay M38; memory/routing stay M39;
systematic eval stays M40; RAG/ANN stay M34–M36; sampling labs stay M32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping

SEED = 3701
RUNTIME_VERSION = "v10-teaching-tools-1"
REGISTRY_ID = "v10-teaching-registry-1"
MAX_ATTEMPTS = 3
VAT_AMOUNT = 80.0
VAT_RATE = 0.25
VAT_TAX = 20.0
VAT_TOTAL = 100.0
LEDGER_KEY = "vat-80-0.25"
LEDGER_ACCOUNT = "cash"

# Fresh no-AI numbers (exact in binary). Do not print the products in the gate.
TRANSFER_DUTY_VALUE = 40.0
TRANSFER_DUTY_RATE = 0.125

CATALOG: dict[str, dict[str, Any]] = {
    "SKU-7": {"name": "lab notebook", "price": 42.0},
    "SKU-9": {"name": "pencil set", "price": 15.5},
    "SKU-12": {"name": "reference binder", "price": 80.0},
}

SKU_PATTERN = r"^SKU-[A-Z0-9]+$"
ACCOUNT_PATTERN = r"^[a-z][a-z0-9_]*$"
KEY_PATTERN = r"^[A-Za-z0-9:._-]+$"

SUPPORTED_DEFECTS = ("none", "malformed_reaches_side_effect", "duplicate_side_effect")
TRACE_STAGES = ("selection", "validation", "execution", "result")
ERROR_KINDS = (
    "parse",
    "schema",
    "permission",
    "tool",
    "retry_exhausted",
    "no_tool",
)
TRACE_FIELDS = (
    "selected_tool",
    "arguments",
    "validation_ok",
    "issues",
    "execution_reached",
    "replayed",
    "error_kind",
    "attempt_index",
    "idempotency_key",
    "approved",
    "status",
    "weights_updated",
    "registry_id",
    "runtime_version",
)

SCALE_LIMIT = (
    "Teaching scale: three local tools, deterministic model-call fixtures, "
    "CPU stdlib validation, no paid API, no LangGraph/Anthropic SDK, no RAG "
    "pack, no Qdrant, no sampling lab. Schema failures must not reach "
    "execution. This fixture is not a production tool gateway."
)

SYSTEM_MAP = (
    "declared intent\n"
    "  -> model-call fixture (not a live LLM)\n"
    "  -> parse proposal\n"
    "  -> select among registered tools (or no-tool)\n"
    "  -> validate schema (strict, additionalProperties false)\n"
    "  -> permission / approval\n"
    "  -> idempotency key\n"
    "  -> execute\n"
    "  -> structured result\n"
    "  -> ToolTrace stages: selection | validation | execution | result"
)

HANDOFF = (
    "M38 receives a validated tool registry/executor and a structured "
    "call/result/error trace (selection, validation, execution, result; "
    "typed schema vs tool errors; idempotency keys). M38 may add a "
    "serializable multi-step state machine. M37 does not."
)


class SchemaError(ValueError):
    """Arguments failed the tool schema. The tool must not run."""

    error_kind = "schema"

    def __init__(self, message: str, issues: tuple["SchemaIssue", ...] = ()) -> None:
        super().__init__(message)
        self.issues = tuple(issues)


class ToolError(RuntimeError):
    """The tool ran and failed. Distinct from a schema failure."""

    error_kind = "tool"


class PermissionDenied(RuntimeError):
    """Side-effecting tool required approval that was not granted."""

    error_kind = "permission"


class RetryExhausted(RuntimeError):
    """Repairable schema errors repeated until the attempt bound."""

    error_kind = "retry_exhausted"


class LiveAdapterUnavailable(RuntimeError):
    """Raised when a live model is requested; canonical tests must not need one."""


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M37 optional NumPy path requires requirements/m37.txt") from exc
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


def _load_m32():
    """Load M32's provider contract. Prefer the package so class identity matches.

    File-load is only a fallback and still registers the package name.
    """

    packaged_name = "missions.M32.inference_adaptation"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    try:
        return __import__(packaged_name, fromlist=["InferenceConfig"])
    except ImportError:
        return _load_sibling("M32", "inference_adaptation.py", packaged_name)


def _normalize_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none"}:
        return "none"
    if key not in SUPPORTED_DEFECTS:
        raise ValueError(f"unsupported defect {defect!r}; use one of {SUPPORTED_DEFECTS}")
    return key


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


@dataclass(frozen=True)
class SchemaIssue:
    field: str
    kind: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: tuple[SchemaIssue, ...] = ()
    normalized: dict[str, Any] = field(default_factory=dict)
    repairable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "repairable": self.repairable,
            "issues": [issue.__dict__ for issue in self.issues],
            "normalized": dict(self.normalized),
        }


@dataclass(frozen=True)
class ToolProposal:
    tool_name: str | None
    arguments: dict[str, Any]
    raw: Any = None
    intent_id: str | None = None

    def args_dict(self) -> dict[str, Any]:
        return dict(self.arguments)

    def idempotency_key(self) -> str | None:
        key = self.arguments.get("idempotency_key")
        return None if key is None else str(key)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    side_effecting: bool = False
    requires_approval: bool = False

    def schema_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "side_effecting": self.side_effecting,
            "requires_approval": self.requires_approval,
            "additionalProperties": False,
        }


@dataclass(frozen=True)
class LedgerEntry:
    entry_id: int
    account: Any
    amount: Any
    memo: Any
    idempotency_key: Any


class LedgerState:
    """Mutable teaching mock. Replay with the same key applies at most once."""

    def __init__(self, entries: tuple[LedgerEntry, ...] = ()) -> None:
        self.entries: list[LedgerEntry] = list(entries)
        self.by_key: dict[str, dict[str, Any]] = {}

    @property
    def effect_count(self) -> int:
        return len(self.entries)

    def snapshot(self) -> tuple[LedgerEntry, ...]:
        return tuple(self.entries)

    def lookup(self, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        return self.by_key.get(str(key))

    def remember(self, key: str | None, output: dict[str, Any]) -> None:
        if key:
            self.by_key[str(key)] = dict(output)

    def append(self, *, account: Any, amount: Any, memo: Any, idempotency_key: Any) -> LedgerEntry:
        entry = LedgerEntry(
            entry_id=len(self.entries) + 1,
            account=account,
            amount=amount,
            memo=memo,
            idempotency_key=idempotency_key,
        )
        self.entries.append(entry)
        return entry

    @classmethod
    def from_snapshot(cls, entries: tuple[LedgerEntry, ...] = ()) -> "LedgerState":
        state = cls(entries)
        for entry in entries:
            if entry.idempotency_key:
                state.by_key[str(entry.idempotency_key)] = {
                    "posted": True,
                    "entry_id": entry.entry_id,
                    "account": entry.account,
                    "amount": entry.amount,
                    "memo": entry.memo,
                    "idempotency_key": entry.idempotency_key,
                }
        return state


class RuntimeSession:
    def __init__(self, ledger: LedgerState | None = None) -> None:
        self.ledger = ledger if ledger is not None else LedgerState()
        self.executions: list[str] = []

    def record_execution(self, tool_name: str) -> None:
        self.executions.append(str(tool_name))

    @property
    def execution_count(self) -> int:
        return len(self.executions)


@dataclass(frozen=True)
class ToolResult:
    status: str
    output: dict[str, Any] | None
    error_kind: str | None
    error_type: str | None
    message: str | None
    replayed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": None if self.output is None else dict(self.output),
            "error_kind": self.error_kind,
            "error_type": self.error_type,
            "message": self.message,
            "replayed": self.replayed,
        }


@dataclass(frozen=True)
class AttemptTrace:
    index: int
    tool_name: str | None
    validation: ValidationResult
    execution_reached: bool
    error_kind: str | None
    status: str


@dataclass(frozen=True)
class ToolTrace:
    defect: str
    proposal: ToolProposal
    selected_tool: str | None
    validation: ValidationResult
    execution_reached: bool
    executed_tool: str | None
    result: ToolResult
    attempts: tuple[AttemptTrace, ...]
    stages: tuple[str, ...]
    error_kind: str | None
    replayed: bool
    approved: bool
    ledger_effect_count: int
    session_execution_count: int
    retry_budget_remaining: int
    inference: dict[str, Any]
    weights_updated: bool = False
    registry_id: str = REGISTRY_ID
    version: str = RUNTIME_VERSION
    intent_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "defect": self.defect,
            "intent_id": self.intent_id,
            "selected_tool": self.selected_tool,
            "arguments": self.proposal.args_dict(),
            "validation_ok": self.validation.ok,
            "issues": [issue.__dict__ for issue in self.validation.issues],
            "execution_reached": self.execution_reached,
            "executed_tool": self.executed_tool,
            "replayed": self.replayed,
            "error_kind": self.error_kind,
            "status": self.result.status,
            "output": self.result.output,
            "approved": self.approved,
            "idempotency_key": self.proposal.idempotency_key(),
            "attempt_index": len(self.attempts),
            "retry_budget_remaining": self.retry_budget_remaining,
            "ledger_effect_count": self.ledger_effect_count,
            "session_execution_count": self.session_execution_count,
            "stages": self.stages,
            "weights_updated": self.weights_updated,
            "registry_id": self.registry_id,
            "runtime_version": self.version,
            "inference": dict(self.inference),
        }


@dataclass(frozen=True)
class FailureTrace:
    defect: str
    claim: str
    proposal: ToolProposal
    result_trace: ToolTrace | None
    initial_entries: tuple[LedgerEntry, ...]
    effect_count: int
    execution_reached: bool
    validation_bypassed: bool
    idempotency_consulted: bool
    session_execution_count: int
    audit: dict[str, Any] = field(default_factory=dict)
    version: str = RUNTIME_VERSION


class ToolRegistry:
    def __init__(self, tools: Mapping[str, ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = dict(tools or {})

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name!r}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools[name] for name in self.names())

    def public_schemas(self) -> dict[str, Any]:
        return {spec.name: spec.schema_public() for spec in self.specs()}

    def fingerprint(self) -> str:
        payload = json.dumps(self.public_schemas(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _handle_compute_vat(arguments: Mapping[str, Any], *, session: RuntimeSession) -> dict[str, Any]:
    del session
    amount = float(arguments["amount"])
    rate = float(arguments["rate"])
    tax = amount * rate
    return {"amount": amount, "rate": rate, "tax": tax, "total": amount + tax}


def _handle_lookup_catalog_price(arguments: Mapping[str, Any], *, session: RuntimeSession) -> dict[str, Any]:
    del session
    sku = str(arguments["sku"])
    item = CATALOG.get(sku)
    if item is None:
        raise ToolError(f"unknown sku {sku}")
    return {"sku": sku, "name": item["name"], "price": float(item["price"])}


def _handle_post_ledger_entry(arguments: Mapping[str, Any], *, session: RuntimeSession) -> dict[str, Any]:
    entry = session.ledger.append(
        account=arguments.get("account"),
        amount=arguments.get("amount"),
        memo=arguments.get("memo"),
        idempotency_key=arguments.get("idempotency_key"),
    )
    return {
        "posted": True,
        "entry_id": entry.entry_id,
        "account": entry.account,
        "amount": entry.amount,
        "memo": entry.memo,
        "idempotency_key": entry.idempotency_key,
    }


VAT_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["amount", "rate"],
    "properties": {
        "amount": {"type": "number", "exclusiveMinimum": 0},
        "rate": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

LOOKUP_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sku"],
    "properties": {
        "sku": {"type": "string", "pattern": SKU_PATTERN},
    },
}

LEDGER_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["account", "amount", "memo", "idempotency_key"],
    "properties": {
        "account": {"type": "string", "minLength": 1, "pattern": ACCOUNT_PATTERN},
        "amount": {"type": "number", "exclusiveMinimum": 0},
        "memo": {"type": "string", "minLength": 1},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 64, "pattern": KEY_PATTERN},
    },
}


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="compute_vat",
            description="Compute VAT tax and total from a taxable amount and a rate in [0, 1].",
            parameters=VAT_PARAMETERS,
            handler=_handle_compute_vat,
            side_effecting=False,
            requires_approval=False,
        )
    )
    registry.register(
        ToolSpec(
            name="lookup_catalog_price",
            description="Look up the unit price of a catalog SKU. Does not post a ledger entry.",
            parameters=LOOKUP_PARAMETERS,
            handler=_handle_lookup_catalog_price,
            side_effecting=False,
            requires_approval=False,
        )
    )
    registry.register(
        ToolSpec(
            name="post_ledger_entry",
            description="Post one ledger debit. Requires approval and an idempotency key.",
            parameters=LEDGER_PARAMETERS,
            handler=_handle_post_ledger_entry,
            side_effecting=True,
            requires_approval=True,
        )
    )
    return registry


INTENT_FIXTURES: dict[str, dict[str, Any]] = {
    "vat_on_80_at_025": {
        "intent": "Compute VAT on amount 80 at rate 0.25",
        "expected_tool": "compute_vat",
        "proposal": {"tool": "compute_vat", "arguments": {"amount": VAT_AMOUNT, "rate": VAT_RATE}},
    },
    "price_of_sku_7": {
        "intent": "What is the catalog price of SKU-7?",
        "expected_tool": "lookup_catalog_price",
        "proposal": {"tool": "lookup_catalog_price", "arguments": {"sku": "SKU-7"}},
    },
    "write_a_haiku": {
        "intent": "Write a haiku about autumn rain.",
        "expected_tool": None,
        "proposal": {"tool": None, "arguments": {}},
    },
    "post_vat_to_ledger": {
        "intent": "Post the VAT tax of 20 to the cash ledger",
        "expected_tool": "post_ledger_entry",
        "proposal": {
            "tool": "post_ledger_entry",
            "arguments": {
                "account": LEDGER_ACCOUNT,
                "amount": VAT_TAX,
                "memo": "vat-tax",
                "idempotency_key": LEDGER_KEY,
            },
        },
    },
}

SELECTION_CASES = (
    "vat_on_80_at_025",
    "price_of_sku_7",
    "write_a_haiku",
)

INVALID_FIXTURES: dict[str, Any] = {
    "missing_rate": {"tool": "compute_vat", "arguments": {"amount": VAT_AMOUNT}},
    "wrong_type": {"tool": "compute_vat", "arguments": {"amount": "eighty", "rate": VAT_RATE}},
    "extra_field": {
        "tool": "compute_vat",
        "arguments": {"amount": VAT_AMOUNT, "rate": VAT_RATE, "currency": "USD"},
    },
    "unsafe_rate": {"tool": "compute_vat", "arguments": {"amount": VAT_AMOUNT, "rate": 1.5}},
    "negative_amount": {"tool": "compute_vat", "arguments": {"amount": -80.0, "rate": VAT_RATE}},
    "bool_amount": {"tool": "compute_vat", "arguments": {"amount": True, "rate": VAT_RATE}},
    "unknown_tool": {"tool": "launch_missiles", "arguments": {}},
    "malformed_json": "{not json",
    "unknown_sku": {"tool": "lookup_catalog_price", "arguments": {"sku": "SKU-ZZ"}},
    "bad_sku_pattern": {"tool": "lookup_catalog_price", "arguments": {"sku": "sku-7"}},
    "malformed_ledger_amount": {
        "tool": "post_ledger_entry",
        "arguments": {
            "account": LEDGER_ACCOUNT,
            "amount": "sixteen",
            "memo": "vat-tax",
            "idempotency_key": "malformed-1",
        },
    },
}


def parse_proposal(raw: Any, *, intent_id: str | None = None) -> ToolProposal:
    """Parse a model-call fixture. Malformed JSON never reaches a tool."""

    if isinstance(raw, ToolProposal):
        return raw
    payload = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(
                f"proposal is not parseable JSON: {exc.msg}",
                issues=(SchemaIssue(field="$", kind="parse", message=exc.msg),),
            ) from exc
    if not isinstance(payload, Mapping):
        raise SchemaError(
            "proposal must be an object",
            issues=(SchemaIssue(field="$", kind="wrong_type", message="proposal must be an object"),),
        )
    tool = payload.get("tool", payload.get("name"))
    if tool is not None and not isinstance(tool, str):
        raise SchemaError(
            "tool name must be a string or null",
            issues=(SchemaIssue(field="tool", kind="wrong_type", message="tool must be a string or null"),),
        )
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise SchemaError(
            "arguments must be an object",
            issues=(SchemaIssue(field="arguments", kind="wrong_type", message="arguments must be an object"),),
        )
    return ToolProposal(
        tool_name=None if tool in {None, ""} else str(tool),
        arguments=dict(arguments),
        raw=raw,
        intent_id=intent_id,
    )


def propose_for_intent(intent_id: str) -> ToolProposal:
    if intent_id not in INTENT_FIXTURES:
        raise ValueError(f"unknown intent {intent_id!r}")
    spec = INTENT_FIXTURES[intent_id]
    return parse_proposal(spec["proposal"], intent_id=intent_id)


def _check_type(expected: str, value: Any) -> bool:
    if expected == "number":
        return _is_number(value)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return _is_string(value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    return False


def _constraint_issues(field_name: str, schema: Mapping[str, Any], value: Any) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []
    if "exclusiveMinimum" in schema and _is_number(value) and not (float(value) > float(schema["exclusiveMinimum"])):
        issues.append(
            SchemaIssue(
                field=field_name,
                kind="constraint",
                message=f"{field_name} must be > {schema['exclusiveMinimum']}",
            )
        )
    if "minimum" in schema and _is_number(value) and float(value) < float(schema["minimum"]):
        issues.append(
            SchemaIssue(
                field=field_name,
                kind="constraint",
                message=f"{field_name} must be >= {schema['minimum']}",
            )
        )
    if "maximum" in schema and _is_number(value) and float(value) > float(schema["maximum"]):
        issues.append(
            SchemaIssue(
                field=field_name,
                kind="constraint",
                message=f"{field_name} must be <= {schema['maximum']}",
            )
        )
    if "minLength" in schema and _is_string(value) and len(value) < int(schema["minLength"]):
        issues.append(
            SchemaIssue(
                field=field_name,
                kind="constraint",
                message=f"{field_name} is shorter than {schema['minLength']}",
            )
        )
    if "maxLength" in schema and _is_string(value) and len(value) > int(schema["maxLength"]):
        issues.append(
            SchemaIssue(
                field=field_name,
                kind="constraint",
                message=f"{field_name} is longer than {schema['maxLength']}",
            )
        )
    if "pattern" in schema and _is_string(value) and re.fullmatch(str(schema["pattern"]), value) is None:
        issues.append(
            SchemaIssue(
                field=field_name,
                kind="constraint",
                message=f"{field_name} does not match {schema['pattern']}",
            )
        )
    return issues


def validate_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> ValidationResult:
    """Strict object schema. Extra keys, wrong types, and bool-as-number fail closed."""

    issues: list[SchemaIssue] = []
    if not isinstance(arguments, Mapping):
        return ValidationResult(
            ok=False,
            issues=(SchemaIssue(field="$", kind="wrong_type", message="arguments must be an object"),),
            repairable=False,
        )
    schema = spec.parameters
    properties = dict(schema.get("properties") or {})
    required = list(schema.get("required") or [])
    additional = schema.get("additionalProperties", False)
    incoming = dict(arguments)

    if additional is False:
        for key in incoming:
            if key not in properties:
                issues.append(SchemaIssue(field=str(key), kind="extra", message=f"unexpected field {key}"))

    for key in required:
        if key not in incoming:
            issues.append(SchemaIssue(field=str(key), kind="missing", message=f"missing required field {key}"))

    normalized: dict[str, Any] = {}
    for key, value in incoming.items():
        if key not in properties:
            continue
        field_schema = properties[key]
        expected = str(field_schema.get("type", "string"))
        if not _check_type(expected, value):
            issues.append(
                SchemaIssue(
                    field=str(key),
                    kind="wrong_type",
                    message=f"{key} should be {expected}",
                )
            )
            continue
        issues.extend(_constraint_issues(str(key), field_schema, value))
        if expected == "number":
            normalized[key] = float(value)
        else:
            normalized[key] = value

    repairable = bool(issues) and all(issue.kind in {"missing", "extra"} for issue in issues)
    return ValidationResult(
        ok=not issues,
        issues=tuple(issues),
        normalized=normalized if not issues else {},
        repairable=repairable,
    )


def validate_proposal(proposal: ToolProposal, registry: ToolRegistry) -> ValidationResult:
    if proposal.tool_name is None:
        return ValidationResult(ok=True, normalized={}, repairable=False)
    if proposal.tool_name not in registry.names():
        return ValidationResult(
            ok=False,
            issues=(
                SchemaIssue(
                    field="tool",
                    kind="unknown_tool",
                    message=f"unknown tool {proposal.tool_name}",
                ),
            ),
            repairable=False,
        )
    return validate_arguments(registry.get(proposal.tool_name), proposal.arguments)


def execute_tool(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
    session: RuntimeSession,
    *,
    approved: bool = False,
    skip_idempotency: bool = False,
    skip_permission: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Run a tool only after the caller has validated arguments.

    Idempotency lives here, one call after validation, not as a substring
    of the handler.
    """

    if spec.requires_approval and not approved and not skip_permission:
        raise PermissionDenied(f"{spec.name} requires approval")
    key = arguments.get("idempotency_key") if spec.side_effecting else None
    if spec.side_effecting and not skip_idempotency:
        cached = session.ledger.lookup(None if key is None else str(key))
        if cached is not None:
            return dict(cached), True
    session.record_execution(spec.name)
    output = spec.handler(dict(arguments), session=session)
    if spec.side_effecting and not skip_idempotency:
        session.ledger.remember(None if key is None else str(key), output)
    return dict(output), False


def attach_inference_evidence(config=None) -> dict[str, Any]:
    """Record M32 provider identity. This is not a sampling lab."""

    m32 = _load_m32()
    cfg = m32.make_config() if config is None else config
    if bool(getattr(cfg, "training_time", False)):
        raise ValueError("tool runtime consumes inference-time config; training_time must be False")
    evidence = m32.config_as_evidence(cfg)
    return {
        "checkpoint_id": evidence["checkpoint_id"],
        "fingerprint": evidence["fingerprint"],
        "training_time": bool(evidence["training_time"]),
        "weights_updated": False,
        "inference_version": evidence["version"],
        "adaptation_stage": evidence["adaptation_stage"],
        "do_sample": evidence["do_sample"],
    }


def make_tool_config():
    m32 = _load_m32()
    cfg = m32.make_config()
    if cfg.training_time:
        raise ValueError("inference config cannot have training_time=True")
    return cfg


def result_as_json(result: ToolResult) -> str:
    return json.dumps(result.as_dict(), sort_keys=True, default=str)


def _success_result(output: dict[str, Any], *, replayed: bool) -> ToolResult:
    return ToolResult(
        status="success",
        output=output,
        error_kind=None,
        error_type=None,
        message=None,
        replayed=replayed,
    )


def _error_result(status: str, error_kind: str, error_type: str, message: str) -> ToolResult:
    return ToolResult(
        status=status,
        output=None,
        error_kind=error_kind,
        error_type=error_type,
        message=message,
        replayed=False,
    )


def _stages_for(*, selected: bool, validated: bool, executed: bool) -> tuple[str, ...]:
    stages = ["selection"]
    if selected:
        stages.append("validation")
    if executed:
        stages.append("execution")
    stages.append("result")
    return tuple(stages)


def fill_missing_repairer(
    proposal: ToolProposal,
    issues: tuple[SchemaIssue, ...],
    *,
    fill: Mapping[str, Any],
) -> ToolProposal:
    """Drop extras and fill missing keys. Does not coerce wrong types."""

    del issues
    allowed = {
        "compute_vat": set(VAT_PARAMETERS["properties"]),
        "lookup_catalog_price": set(LOOKUP_PARAMETERS["properties"]),
        "post_ledger_entry": set(LEDGER_PARAMETERS["properties"]),
    }.get(proposal.tool_name or "", None)
    arguments = dict(proposal.arguments)
    if allowed is not None:
        arguments = {key: value for key, value in arguments.items() if key in allowed}
    for key, value in fill.items():
        arguments.setdefault(key, value)
    return ToolProposal(
        tool_name=proposal.tool_name,
        arguments=arguments,
        raw=proposal.raw,
        intent_id=proposal.intent_id,
    )


def sticky_invalid_repairer(proposal: ToolProposal, issues: tuple[SchemaIssue, ...]) -> ToolProposal:
    del issues
    return proposal


def vat_fill_repairer(proposal: ToolProposal, issues: tuple[SchemaIssue, ...]) -> ToolProposal:
    return fill_missing_repairer(
        proposal,
        issues,
        fill={"amount": VAT_AMOUNT, "rate": VAT_RATE},
    )


def _trace(
    *,
    proposal: ToolProposal,
    selected_tool: str | None,
    validation: ValidationResult,
    execution_reached: bool,
    executed_tool: str | None,
    result: ToolResult,
    attempts: list[AttemptTrace],
    approved: bool,
    session: RuntimeSession,
    max_attempts: int,
    inference: dict[str, Any],
    defect: str,
) -> ToolTrace:
    return ToolTrace(
        defect=defect,
        proposal=proposal,
        selected_tool=selected_tool,
        validation=validation,
        execution_reached=execution_reached,
        executed_tool=executed_tool,
        result=result,
        attempts=tuple(attempts),
        stages=_stages_for(
            selected=selected_tool is not None or not validation.ok,
            validated=selected_tool is not None,
            executed=execution_reached,
        ),
        error_kind=result.error_kind,
        replayed=result.replayed,
        approved=approved,
        ledger_effect_count=session.ledger.effect_count,
        session_execution_count=session.execution_count,
        retry_budget_remaining=max(0, int(max_attempts) - len(attempts)),
        inference=inference,
        weights_updated=False,
        intent_id=proposal.intent_id,
    )


def run_tool_call(
    source: Any,
    *,
    registry: ToolRegistry | None = None,
    session: RuntimeSession | None = None,
    approved: bool = False,
    max_attempts: int = MAX_ATTEMPTS,
    repairer: Callable[[ToolProposal, tuple[SchemaIssue, ...]], ToolProposal] | None = None,
    inference_config=None,
    defect: str | None = "none",
    skip_validation: bool = False,
    skip_idempotency: bool = False,
    skip_permission: bool = False,
) -> ToolTrace:
    """Fixture → parse → select → validate → permission → idempotency → execute.

    Invalid calls never reach the tool unless a named defect bypasses
    validation. Live adapters fail closed. This wrapper does not open a
    state machine, a RAG pack, or a sampling lab.
    """

    defect_key = _normalize_defect(defect)
    registry = default_registry() if registry is None else registry
    session = RuntimeSession() if session is None else session
    inference = attach_inference_evidence(inference_config)
    attempts: list[AttemptTrace] = []

    if isinstance(source, str) and source in INTENT_FIXTURES:
        current = propose_for_intent(source)
    else:
        try:
            current = parse_proposal(source)
        except SchemaError as exc:
            validation = ValidationResult(ok=False, issues=exc.issues, repairable=False)
            attempts.append(
                AttemptTrace(
                    index=1,
                    tool_name=None,
                    validation=validation,
                    execution_reached=False,
                    error_kind="parse",
                    status="schema_error",
                )
            )
            result = _error_result("schema_error", "parse", "SchemaError", str(exc))
            return _trace(
                proposal=ToolProposal(tool_name=None, arguments={}, raw=source),
                selected_tool=None,
                validation=validation,
                execution_reached=False,
                executed_tool=None,
                result=result,
                attempts=attempts,
                approved=approved,
                session=session,
                max_attempts=max_attempts,
                inference=inference,
                defect=defect_key,
            )

    bound = max(1, int(max_attempts))
    last_validation = ValidationResult(ok=False)
    for attempt_index in range(1, bound + 1):
        selected = current.tool_name
        if selected is None:
            validation = ValidationResult(ok=True, normalized={}, repairable=False)
            attempts.append(
                AttemptTrace(
                    index=attempt_index,
                    tool_name=None,
                    validation=validation,
                    execution_reached=False,
                    error_kind="no_tool",
                    status="no_tool",
                )
            )
            result = _error_result("no_tool", "no_tool", None, "no tool selected")
            return _trace(
                proposal=current,
                selected_tool=None,
                validation=validation,
                execution_reached=False,
                executed_tool=None,
                result=result,
                attempts=attempts,
                approved=approved,
                session=session,
                max_attempts=bound,
                inference=inference,
                defect=defect_key,
            )

        validation = (
            ValidationResult(ok=True, normalized=dict(current.arguments), repairable=False)
            if skip_validation
            else validate_proposal(current, registry)
        )
        last_validation = validation
        if not validation.ok:
            attempts.append(
                AttemptTrace(
                    index=attempt_index,
                    tool_name=selected,
                    validation=validation,
                    execution_reached=False,
                    error_kind="schema",
                    status="schema_error",
                )
            )
            can_retry = repairer is not None and attempt_index < bound
            if can_retry:
                current = repairer(current, validation.issues)
                continue
            status = "retry_exhausted" if repairer is not None and attempt_index >= bound else "schema_error"
            error_kind = "retry_exhausted" if status == "retry_exhausted" else "schema"
            error_type = "RetryExhausted" if error_kind == "retry_exhausted" else "SchemaError"
            message = "; ".join(issue.message for issue in validation.issues) or "schema validation failed"
            if error_kind == "retry_exhausted":
                message = f"retry bound {bound} exhausted: {message}"
            result = _error_result(status, error_kind, error_type, message)
            return _trace(
                proposal=current,
                selected_tool=selected,
                validation=validation,
                execution_reached=False,
                executed_tool=None,
                result=result,
                attempts=attempts,
                approved=approved,
                session=session,
                max_attempts=bound,
                inference=inference,
                defect=defect_key,
            )

        spec = registry.get(selected)
        try:
            output, replayed = execute_tool(
                spec,
                validation.normalized if not skip_validation else dict(current.arguments),
                session,
                approved=approved,
                skip_idempotency=skip_idempotency,
                skip_permission=skip_permission,
            )
        except PermissionDenied as exc:
            attempts.append(
                AttemptTrace(
                    index=attempt_index,
                    tool_name=selected,
                    validation=validation,
                    execution_reached=False,
                    error_kind="permission",
                    status="permission_denied",
                )
            )
            result = _error_result("permission_denied", "permission", "PermissionDenied", str(exc))
            return _trace(
                proposal=current,
                selected_tool=selected,
                validation=validation,
                execution_reached=False,
                executed_tool=None,
                result=result,
                attempts=attempts,
                approved=approved,
                session=session,
                max_attempts=bound,
                inference=inference,
                defect=defect_key,
            )
        except ToolError as exc:
            attempts.append(
                AttemptTrace(
                    index=attempt_index,
                    tool_name=selected,
                    validation=validation,
                    execution_reached=True,
                    error_kind="tool",
                    status="tool_error",
                )
            )
            result = _error_result("tool_error", "tool", "ToolError", str(exc))
            return _trace(
                proposal=current,
                selected_tool=selected,
                validation=validation,
                execution_reached=True,
                executed_tool=selected,
                result=result,
                attempts=attempts,
                approved=approved,
                session=session,
                max_attempts=bound,
                inference=inference,
                defect=defect_key,
            )

        attempts.append(
            AttemptTrace(
                index=attempt_index,
                tool_name=selected,
                validation=validation,
                execution_reached=True,
                error_kind=None,
                status="success",
            )
        )
        result = _success_result(output, replayed=replayed)
        return _trace(
            proposal=current,
            selected_tool=selected,
            validation=validation,
            execution_reached=True,
            executed_tool=selected,
            result=result,
            attempts=attempts,
            approved=approved,
            session=session,
            max_attempts=bound,
            inference=inference,
            defect=defect_key,
        )

    message = "; ".join(issue.message for issue in last_validation.issues) or "schema validation failed"
    result = _error_result(
        "retry_exhausted",
        "retry_exhausted",
        "RetryExhausted",
        f"retry bound {bound} exhausted: {message}",
    )
    return _trace(
        proposal=current,
        selected_tool=current.tool_name,
        validation=last_validation,
        execution_reached=False,
        executed_tool=None,
        result=result,
        attempts=attempts,
        approved=approved,
        session=session,
        max_attempts=bound,
        inference=inference,
        defect=defect_key,
    )


def evaluate_selection(
    case_ids: tuple[str, ...] = SELECTION_CASES,
    *,
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    registry = default_registry() if registry is None else registry
    rows = []
    for case_id in case_ids:
        expected = INTENT_FIXTURES[case_id]["expected_tool"]
        proposal = propose_for_intent(case_id)
        trace = run_tool_call(proposal, registry=registry, approved=True)
        rows.append(
            {
                "case_id": case_id,
                "expected": expected,
                "selected": proposal.tool_name,
                "status": trace.result.status,
                "correct": proposal.tool_name == expected,
            }
        )
    return {
        "n": len(rows),
        "n_correct": sum(1 for row in rows if row["correct"]),
        "rows": rows,
    }


def numpy_vat_table(amounts, rate: float):
    """Optional NumPy parity helper. Required tests use stdlib arithmetic."""

    np = _require_numpy()
    values = np.asarray(tuple(float(amount) for amount in amounts), dtype=float)
    tax = values * float(rate)
    return tax, values + tax


def optional_live_propose(intent: str, config=None) -> ToolProposal:
    """Optional live-model adapter. Canonical validation must not call a network.

    The function exists so later missions can see where a provider would plug
    in. Live models are not required. It always fails closed here.
    """

    del intent, config
    raise LiveAdapterUnavailable(
        "M37 canonical path uses local model-call fixtures; live models are "
        "optional and not required for validation"
    )


def pipeline_with_defect(
    *,
    defect: str,
    session: RuntimeSession | None = None,
    proposal: ToolProposal | Mapping | None = None,
) -> FailureTrace:
    """Controlled-failure entry: one named defect, same teaching registry."""

    key = _normalize_defect(defect)
    if key == "none":
        raise ValueError("pipeline_with_defect requires a named defect")
    session = RuntimeSession() if session is None else session
    initial = session.ledger.snapshot()
    registry = default_registry()

    if key == "malformed_reaches_side_effect":
        raw = proposal or INVALID_FIXTURES["malformed_ledger_amount"]
        parsed = parse_proposal(raw)
        # Missing validation: malformed amount reaches the side-effecting tool.
        trace = run_tool_call(
            parsed,
            registry=registry,
            session=session,
            approved=True,
            defect=key,
            skip_validation=True,
        )
        return FailureTrace(
            defect=key,
            claim="malformed_arguments_executed",
            proposal=parsed,
            result_trace=trace,
            initial_entries=initial,
            effect_count=session.ledger.effect_count,
            execution_reached=True,
            validation_bypassed=True,
            idempotency_consulted=True,
            session_execution_count=session.execution_count,
            audit={
                "amount": parsed.arguments.get("amount"),
                "amount_type": type(parsed.arguments.get("amount")).__name__,
                "healthy_validation_ok": validate_proposal(parsed, registry).ok,
            },
        )

    if key == "duplicate_side_effect":
        raw = proposal or INTENT_FIXTURES["post_vat_to_ledger"]["proposal"]
        parsed = parse_proposal(raw)
        # Simulated timeout then retry without consulting the idempotency store.
        first = run_tool_call(
            parsed,
            registry=registry,
            session=session,
            approved=True,
            defect=key,
            skip_idempotency=True,
        )
        second = run_tool_call(
            parsed,
            registry=registry,
            session=session,
            approved=True,
            defect=key,
            skip_idempotency=True,
        )
        return FailureTrace(
            defect=key,
            claim="timeout_retry_double_post",
            proposal=parsed,
            result_trace=second,
            initial_entries=initial,
            effect_count=session.ledger.effect_count,
            execution_reached=True,
            validation_bypassed=False,
            idempotency_consulted=False,
            session_execution_count=session.execution_count,
            audit={
                "first_entry_id": None if first.result.output is None else first.result.output.get("entry_id"),
                "second_entry_id": None if second.result.output is None else second.result.output.get("entry_id"),
                "simulated": "timeout_then_retry",
            },
        )

    raise ValueError(f"unsupported defect {defect!r}")


def repair_run(trace: FailureTrace) -> FailureTrace:
    """Recompute from the defective object's proposal and initial ledger snapshot.

    Does not start a second unrelated happy-path run from module defaults.
    """

    if trace.defect == "malformed_reaches_side_effect":
        session = RuntimeSession(LedgerState.from_snapshot(trace.initial_entries))
        repaired = run_tool_call(
            trace.proposal,
            session=session,
            approved=True,
            defect="none",
        )
        return FailureTrace(
            defect="none",
            claim="validation_restored",
            proposal=trace.proposal,
            result_trace=repaired,
            initial_entries=trace.initial_entries,
            effect_count=session.ledger.effect_count,
            execution_reached=repaired.execution_reached,
            validation_bypassed=False,
            idempotency_consulted=True,
            session_execution_count=session.execution_count,
            audit={
                "from_defect": trace.defect,
                "reused_amount": trace.proposal.arguments.get("amount"),
                "status": repaired.result.status,
            },
        )
    if trace.defect == "duplicate_side_effect":
        session = RuntimeSession(LedgerState.from_snapshot(trace.initial_entries))
        first = run_tool_call(trace.proposal, session=session, approved=True, defect="none")
        second = run_tool_call(trace.proposal, session=session, approved=True, defect="none")
        return FailureTrace(
            defect="none",
            claim="idempotency_restored",
            proposal=trace.proposal,
            result_trace=second,
            initial_entries=trace.initial_entries,
            effect_count=session.ledger.effect_count,
            execution_reached=True,
            validation_bypassed=False,
            idempotency_consulted=True,
            session_execution_count=session.execution_count,
            audit={
                "from_defect": trace.defect,
                "first_replayed": first.replayed,
                "second_replayed": second.replayed,
                "reused_key": trace.proposal.idempotency_key(),
            },
        )
    raise ValueError(f"repair_run expects a named defect, not {trace.defect!r}")


def observability_report(trace: ToolTrace) -> dict[str, Any]:
    return {
        "version": RUNTIME_VERSION,
        "registry_id": REGISTRY_ID,
        "defect": trace.defect,
        "stages": trace.stages,
        "trace": trace.as_dict(),
        "trace_fields": TRACE_FIELDS,
        "error_kinds": ERROR_KINDS,
        "weights_updated": False,
        "training_time": trace.inference.get("training_time", False),
        "scale_limit": SCALE_LIMIT,
        "handoff": HANDOFF,
    }


def handoff_contract() -> dict[str, Any]:
    return {
        "registry_id": REGISTRY_ID,
        "runtime_version": RUNTIME_VERSION,
        "tools": default_registry().names(),
        "trace_stages": TRACE_STAGES,
        "trace_fields": TRACE_FIELDS,
        "error_kinds": ERROR_KINDS,
        "idempotency": (
            "side-effecting tools require idempotency_key; replay applies the effect at most once"
        ),
        "permissions": "side-effecting tools require approved=True before execution",
        "retry_limit": MAX_ATTEMPTS,
        "state_machines": "deferred to M38",
        "memory_routing": "deferred to M39",
        "eval_harness": "deferred to M40",
        "handoff": HANDOFF,
    }
