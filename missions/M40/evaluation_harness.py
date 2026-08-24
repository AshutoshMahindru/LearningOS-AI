"""Versioned evaluation harness for Mission M40.

M40 is the V11 evaluation and observability boundary. Freeze a scenario
pack spanning M34 RAG traces and M39 agent traces *before* optimizing
either system. Grade objective invariants with deterministic checkers.
Keep rubric / LLM-as-judge at a calibrated boundary. Slice by severity.
Decide release gates on critical failures, not averages.

Named sources (not SDKs): anthropic-evals, anthropic-agents.
Content bundle: ai-system-evals.

Do not import paid eval SDKs. Optional LLM-as-judge is fail-closed and
cannot be the sole required grader. M41 owns integrated architecture;
this module does not draw that diagram or retune M34/M39 models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SEED = 4001
HARNESS_VERSION = "v11-eval-harness-1"
EVAL_VERSION = "m40.eval.v1"
CONTAMINATED_VERSION = "m40.eval.tuned-dev"
PIPELINE_ID = "v11-eval-harness"
CONTENT_BUNDLE = "ai-system-evals"
CANONICAL_SOURCES = ("anthropic-evals", "anthropic-agents")
M34_PACKAGE = "missions.M34.rag_pipeline"
M39_PACKAGE = "missions.M39.robust_agent"
M37_PACKAGE = "missions.M37.tool_runtime"

DETERMINISTIC_GRADERS = (
    "tool_schema",
    "citation_support",
    "state_termination",
    "idempotency",
)
FAMILIES = ("rag", "agent", "tool")
SEVERITIES = ("critical", "major", "minor")
SCENARIO_TAXONOMY = {
    "rag": ("grounded_answer", "citation_support", "abstention"),
    "agent": (
        "state_termination",
        "memory_scope",
        "fallback_bound",
        "degraded_path",
    ),
    "tool": ("tool_schema", "idempotency"),
}
CRITICAL_GRADERS = DETERMINISTIC_GRADERS
OBSERVABILITY_FIELDS = (
    "case_id",
    "family",
    "eval_version",
    "retrieval_ids",
    "citation_ids",
    "support_ok",
    "abstain_reason",
    "route",
    "terminal",
    "degraded",
    "used_memory_ids",
    "excluded",
    "attempts",
    "circuit_open",
    "effect_count",
    "posted_amount",
    "tool_name",
    "schema_ok",
    "replayed",
    "execution_reached",
    "step_count",
    "cost_proxy",
    "index_id",
    "source_hash",
)
ABLATION_DIAGNOSES = {
    "used_memory_ids": ("stale_memory_posted_as_complete",),
    "citation_ids": ("unsupported_citation_localization",),
    "support_ok": ("citation_vs_generation_split",),
    "excluded": ("memory_scope_or_expiry_reason",),
    "circuit_open": ("unbounded_fallback_vs_intended_stop",),
    "attempts": ("attempt_bound_diagnosis",),
    "effect_count": ("idempotent_replay_vs_double_post",),
    "schema_ok": ("invalid_tool_args_reached_execution",),
}

SYSTEM_INJECTABLE = (
    "unsupported_citation",
    "duplicate_side_effect",
    "stale_memory_trusted",
)
GOVERNANCE_DEFECTS = ("hidden_critical", "contaminated_pack")
SUPPORTED_DEFECTS = SYSTEM_INJECTABLE + GOVERNANCE_DEFECTS + ("none",)

HIDDEN_CRITICAL_CASE = "rag-grounded-reset"
HIDDEN_CRITICAL_GRADER = "citation_support"
HIDDEN_CRITICAL_INJECT = "unsupported_citation"

SCALE_LIMIT = (
    "Teaching scale: one frozen local eval pack, deterministic graders for "
    "schema/citation/termination/idempotency, a calibrated rubric boundary, "
    "CPU stdlib orchestration wrapping M34 and M39 fixtures, no paid eval "
    "SDK, no required LLM-as-judge, no M41 architecture diagram. Averages "
    "are not release evidence. This fixture is not a production eval platform."
)

SYSTEM_MAP = (
    "frozen eval pack (versioned cases, RAG + agent + tool)\n"
    "  -> invoke_case (M34 answer_labeled | M39 run_robust_task | M37 run_tool_call)\n"
    "  -> deterministic graders (tool_schema, citation_support, state_termination, idempotency)\n"
    "  -> rubric/judge only where deterministic is insufficient (calibrated; fail-closed)\n"
    "  -> aggregate outcome success vs component slices vs critical rate\n"
    "  -> cost/step proxies\n"
    "  -> release gate (version + contamination + slice thresholds)\n"
    "  traces retain retrieval, citations, route, attempts, degraded, effects"
)

HANDOFF = (
    "M41 receives a versioned eval suite, a failure taxonomy, release gates, "
    "and observability field expectations. M42 may reuse the suite. M40 does "
    "not close V11: architecture boundaries remain M41. Package existence is "
    "not phase completion and is not learner completion."
)

INFERENCE_BOUNDARY = (
    "M40 evaluates inference-time system behavior. Graders do not update "
    "M34 or M39 weights. Optional LLM-as-judge is not a required grader."
)


class EvalVersionError(ValueError):
    """Eval pack version does not match the required freeze."""


class EvalContaminationError(ValueError):
    """Eval pack was tuned against or is not the canonical freeze."""


class RubricCannotGateInvariants(ValueError):
    """Rubric/judge scores cannot replace deterministic invariant graders."""


class OptionalLLMJudgeUnavailable(RuntimeError):
    """Raised when the optional LLM-as-judge adapter is requested."""


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by dependency contract
        raise RuntimeError("M40 optional NumPy path requires requirements/m40.txt") from exc
    return np


def numpy_available() -> bool:
    return importlib.util.find_spec("numpy") is not None


def default_dataset_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "datasets" / "M40"
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled eval fixtures: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return json.loads(json.dumps(dict(value)))


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


def _load_package(mission_id: str, filename: str):
    """Prefer the package so class identity matches. File-load is fallback."""

    packaged_name = f"missions.{mission_id}.{filename[:-3]}"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module(packaged_name)
    except ImportError:
        return _load_sibling(mission_id, filename, packaged_name)


def _load_m34():
    existing = sys.modules.get(M34_PACKAGE)
    if existing is not None:
        return existing
    return _load_package("M34", "rag_pipeline.py")


def _load_m39():
    existing = sys.modules.get(M39_PACKAGE)
    if existing is not None:
        return existing
    return _load_package("M39", "robust_agent.py")


def _load_m37():
    existing = sys.modules.get(M37_PACKAGE)
    if existing is not None:
        return existing
    return _load_package("M37", "tool_runtime.py")


def _normalize_system_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none"}:
        return "none"
    if key not in SYSTEM_INJECTABLE:
        raise ValueError(f"unsupported system defect {defect!r}; use one of {SYSTEM_INJECTABLE}")
    return key


def _normalize_governance_defect(defect: str | None) -> str:
    key = "none" if defect is None else str(defect).lower()
    if key in {"", "none"}:
        return "none"
    if key not in GOVERNANCE_DEFECTS:
        raise ValueError(
            f"unsupported governance defect {defect!r}; use one of {GOVERNANCE_DEFECTS}"
        )
    return key


@dataclass(frozen=True)
class ReleasePolicy:
    name: str
    min_task_success: float
    max_critical_fail_rate: float
    slice_max_fail: tuple[tuple[str, float], ...]
    require_eval_version: str | None
    reject_contaminated: bool

    def slice_map(self) -> dict[str, float]:
        return {name: limit for name, limit in self.slice_max_fail}


CANONICAL_POLICY = ReleasePolicy(
    name="canonical_slice_gates",
    min_task_success=0.8,
    max_critical_fail_rate=0.0,
    slice_max_fail=(
        ("citation_support", 0.0),
        ("tool_schema", 0.0),
        ("idempotency", 0.0),
        ("state_termination", 0.0),
    ),
    require_eval_version=EVAL_VERSION,
    reject_contaminated=True,
)

AGGREGATE_ONLY_POLICY = ReleasePolicy(
    name="aggregate_only",
    min_task_success=0.8,
    max_critical_fail_rate=1.0,
    slice_max_fail=(),
    require_eval_version=None,
    reject_contaminated=False,
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    family: str
    scenario: str
    severity: str
    split: str
    graders: tuple[str, ...]
    invoke: dict[str, Any]
    gold: dict[str, Any]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "family": self.family,
            "scenario": self.scenario,
            "severity": self.severity,
            "split": self.split,
            "graders": list(self.graders),
            "invoke": _copy_mapping(self.invoke),
            "gold": _copy_mapping(self.gold),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class EvalPack:
    eval_version: str
    cases: tuple[EvalCase, ...]
    contaminated: bool
    downloaded: bool
    network_required: bool
    held_out_untuned: bool
    note: str
    source_hash: str
    path: str
    tuned_against: bool = False

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def get(self, case_id: str) -> EvalCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "eval_version": self.eval_version,
            "case_ids": list(self.case_ids),
            "contaminated": self.contaminated,
            "downloaded": self.downloaded,
            "network_required": self.network_required,
            "held_out_untuned": self.held_out_untuned,
            "tuned_against": self.tuned_against,
            "source_hash": self.source_hash,
            "n": len(self.cases),
        }


def _case_from_payload(row: Mapping[str, Any]) -> EvalCase:
    graders = tuple(str(item) for item in row.get("graders") or ())
    for name in graders:
        if name not in DETERMINISTIC_GRADERS:
            raise ValueError(f"unknown grader {name!r} on case {row.get('id')}")
    family = str(row["family"])
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    severity = str(row.get("severity") or "major")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return EvalCase(
        case_id=str(row["id"]),
        family=family,
        scenario=str(row["scenario"]),
        severity=severity,
        split=str(row.get("split") or "dev"),
        graders=graders,
        invoke=_copy_mapping(row.get("invoke") or {}),
        gold=_copy_mapping(row.get("gold") or {}),
        notes=str(row.get("notes") or ""),
    )


def load_eval_pack(
    path: str | Path | None = None,
    *,
    version: str | None = None,
    require_canonical: bool = False,
) -> EvalPack:
    """Load a versioned eval pack. Canonical freeze is m40.eval.v1.

    ``require_canonical`` rejects tuned/contaminated packs and version drift.
    It does not run the system and does not retune M34 or M39.
    """

    target = Path(path) if path is not None else default_dataset_dir() / "eval_pack.json"
    payload = _read_json(target)
    eval_version = str(payload.get("eval_version") or "")
    contaminated = bool(payload.get("contaminated", False))
    tuned_against = bool(payload.get("tuned_against", False))
    if version is not None and eval_version != version:
        raise EvalVersionError(f"eval pack {eval_version!r} != required {version!r}")
    if require_canonical:
        if eval_version != EVAL_VERSION:
            raise EvalVersionError(
                f"canonical pack must be {EVAL_VERSION!r}, found {eval_version!r}"
            )
        if contaminated or tuned_against:
            raise EvalContaminationError(
                "canonical load refuses a pack marked contaminated or tuned_against"
            )
        if payload.get("downloaded", False) or payload.get("network_required", False):
            raise ValueError("canonical pack must be local and offline")
    cases = tuple(_case_from_payload(row) for row in payload.get("cases") or ())
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("eval pack case ids must be unique")
    source_hash = str(payload.get("source_hash") or _sha256_text(target.read_text(encoding="utf-8")))
    return EvalPack(
        eval_version=eval_version,
        cases=cases,
        contaminated=contaminated,
        downloaded=bool(payload.get("downloaded", False)),
        network_required=bool(payload.get("network_required", False)),
        held_out_untuned=bool(payload.get("held_out_untuned", False)),
        note=str(payload.get("note") or ""),
        source_hash=source_hash,
        path=str(target),
        tuned_against=tuned_against,
    )


def load_rubric_labels(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else default_dataset_dir() / "rubric_labels.json"
    return _read_json(target)


def load_expected_payload(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else default_dataset_dir() / "expected.json"
    return _read_json(target)


def load_transfer_payload(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else default_dataset_dir() / "transfer.json"
    return _read_json(target)


def pack_fingerprint(pack: EvalPack) -> str:
    payload = {
        "eval_version": pack.eval_version,
        "case_ids": list(pack.case_ids),
        "contaminated": pack.contaminated,
        "held_out_untuned": pack.held_out_untuned,
    }
    return _sha256_text(_dump_canonical(payload))[:16]


@dataclass(frozen=True)
class GradeResult:
    grader: str
    passed: bool
    severity: str
    localized_failure: str | None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "grader": self.grader,
            "passed": self.passed,
            "severity": self.severity,
            "localized_failure": self.localized_failure,
            "evidence": _copy_mapping(self.evidence),
        }


@dataclass(frozen=True)
class InvocationTrace:
    case_id: str
    family: str
    scenario: str
    eval_version: str
    status: str | None = None
    terminal: str | None = None
    degraded: bool = False
    circuit_open: bool = False
    attempts: int = 0
    route: str | None = None
    retrieval_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    support_ok: bool | None = None
    abstain_reason: str | None = None
    used_memory_ids: tuple[str, ...] = ()
    excluded: tuple[dict[str, str], ...] = ()
    effect_count: int | None = None
    posted_amount: float | None = None
    tool_name: str | None = None
    schema_ok: bool | None = None
    replayed: bool | None = None
    execution_reached: bool | None = None
    executions: tuple[str, ...] = ()
    step_count: int = 0
    cost_proxy: float = 0.0
    packed_chars: int = 0
    index_id: str | None = None
    source_hash: str | None = None
    rag: Any = None
    agent: Any = None
    tool_first: Any = None
    tool_second: Any = None
    proposal: Any = None
    defect: str = "none"

    def observability(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "eval_version": self.eval_version,
            "retrieval_ids": list(self.retrieval_ids),
            "citation_ids": list(self.citation_ids),
            "support_ok": self.support_ok,
            "abstain_reason": self.abstain_reason,
            "route": self.route,
            "terminal": self.terminal,
            "degraded": self.degraded,
            "used_memory_ids": list(self.used_memory_ids),
            "excluded": [dict(row) for row in self.excluded],
            "attempts": self.attempts,
            "circuit_open": self.circuit_open,
            "effect_count": self.effect_count,
            "posted_amount": self.posted_amount,
            "tool_name": self.tool_name,
            "schema_ok": self.schema_ok,
            "replayed": self.replayed,
            "execution_reached": self.execution_reached,
            "step_count": self.step_count,
            "cost_proxy": self.cost_proxy,
            "index_id": self.index_id,
            "source_hash": self.source_hash,
        }


def _system_defect_applies(case: EvalCase, defect: str) -> bool:
    if defect == "none":
        return False
    if defect == "unsupported_citation":
        return case.case_id == HIDDEN_CRITICAL_CASE
    if defect == "duplicate_side_effect":
        return case.case_id == "tool-idempotency-replay"
    if defect == "stale_memory_trusted":
        return case.case_id == "agent-purchase-complete"
    return False


def _cost_proxy(*, step_count: int, effect_count: int | None, packed_chars: int) -> float:
    extra = 0 if effect_count is None else int(effect_count)
    return float(step_count + extra) + (float(packed_chars) / 1000.0)


def _invoke_rag(case: EvalCase, *, defect: str) -> InvocationTrace:
    if not numpy_available():
        raise RuntimeError("M40 RAG cases require NumPy from requirements/m40.txt")
    m34 = _load_m34()
    applied = m34.DEFECT_UNSUPPORTED if _system_defect_applies(case, defect) else m34.DEFECT_NONE
    query_id = str(case.invoke["query_id"])
    top_k = int(case.invoke.get("top_k") or m34.DEFAULT_TOP_K)
    rag = m34.answer_labeled(query_id, top_k=top_k, defect=applied)
    packed_chars = int(rag.pack.char_count())
    step_count = 3
    return InvocationTrace(
        case_id=case.case_id,
        family="rag",
        scenario=case.scenario,
        eval_version=EVAL_VERSION,
        status=rag.answer.status,
        citation_ids=tuple(rag.answer.citation_ids()),
        support_ok=bool(rag.answer.support.ok),
        abstain_reason=rag.answer.abstain_reason,
        retrieval_ids=tuple(rag.retrieval_ids),
        step_count=step_count,
        packed_chars=packed_chars,
        cost_proxy=_cost_proxy(step_count=step_count, effect_count=0, packed_chars=packed_chars),
        index_id=rag.index_id,
        source_hash=rag.source_hash,
        rag=rag,
        defect=applied,
    )


def _agent_store(case: EvalCase, m39: Any, *, defect: str):
    kind = str(case.invoke.get("store") or "catalog")
    if _system_defect_applies(case, defect) or kind == "stale_and_fresh":
        return m39.demo_store_stale_and_fresh()
    if kind == "relevant_and_irrelevant":
        return m39.demo_store_relevant_and_irrelevant()
    return m39.MemoryStore().put(m39.catalog_price_entry())


def _invoke_agent(case: EvalCase, *, defect: str) -> InvocationTrace:
    m39 = _load_m39()
    applied = "stale_memory_trusted" if _system_defect_applies(case, defect) else "none"
    store = _agent_store(case, m39, defect=defect)
    task = str(case.invoke.get("task") or m39.DEFAULT_TASK)
    inject = case.invoke.get("inject")
    result = m39.run_robust_task(
        task,
        store=store,
        inject=None if inject in {None, "none"} else str(inject),
        defect=applied,
    )
    executions: tuple[str, ...] = ()
    if result.workflow is not None:
        executions = tuple(result.workflow.session.executions)
    step_count = int(result.attempts)
    packed_chars = 0
    return InvocationTrace(
        case_id=case.case_id,
        family="agent",
        scenario=case.scenario,
        eval_version=EVAL_VERSION,
        status=result.status,
        terminal=result.terminal,
        degraded=bool(result.degraded),
        circuit_open=bool(result.circuit_open),
        attempts=int(result.attempts),
        route=result.route,
        retrieval_ids=tuple(result.retrieved_ids),
        used_memory_ids=tuple(result.used_memory_ids),
        excluded=tuple({"id": str(row["id"]), "reason": str(row["reason"])} for row in result.excluded),
        effect_count=int(result.effect_count),
        posted_amount=result.posted_amount,
        executions=executions,
        tool_name=executions[-1] if executions else None,
        step_count=step_count,
        cost_proxy=_cost_proxy(
            step_count=step_count,
            effect_count=int(result.effect_count),
            packed_chars=packed_chars,
        ),
        agent=result,
        defect=applied,
    )


def _invoke_tool(case: EvalCase, *, defect: str) -> InvocationTrace:
    m37 = _load_m37()
    session = m37.RuntimeSession()
    raw = case.invoke.get("proposal")
    skip_validation = bool(case.invoke.get("skip_validation", False))
    skip_idempotency = bool(case.invoke.get("skip_idempotency", False))
    if _system_defect_applies(case, defect) and defect == "duplicate_side_effect":
        skip_idempotency = True
    if defect == "malformed_reaches_side_effect" and case.scenario == "tool_schema":
        skip_validation = True
    approved = True
    first = m37.run_tool_call(
        raw,
        session=session,
        approved=approved,
        skip_validation=skip_validation,
        skip_idempotency=skip_idempotency,
        skip_permission=True,
    )
    second = None
    if case.scenario == "idempotency":
        second = m37.run_tool_call(
            raw,
            session=session,
            approved=approved,
            skip_validation=skip_validation,
            skip_idempotency=skip_idempotency,
            skip_permission=True,
        )
    validation = first.validation
    schema_ok = bool(validation.ok)
    execution_reached = bool(first.execution_reached) or bool(
        second.execution_reached if second is not None else False
    )
    replayed = None if second is None else bool(second.replayed)
    step_count = 1 if second is None else 2
    effect_count = int(session.ledger.effect_count)
    return InvocationTrace(
        case_id=case.case_id,
        family="tool",
        scenario=case.scenario,
        eval_version=EVAL_VERSION,
        status=None if second is None else second.result.status,
        tool_name=first.selected_tool,
        schema_ok=schema_ok,
        replayed=replayed,
        execution_reached=execution_reached,
        effect_count=effect_count,
        step_count=step_count,
        cost_proxy=_cost_proxy(step_count=step_count, effect_count=effect_count, packed_chars=0),
        tool_first=first,
        tool_second=second,
        proposal=raw,
        defect=defect if _system_defect_applies(case, defect) else "none",
    )


def invoke_case(case: EvalCase, *, defect: str | None = "none") -> InvocationTrace:
    """Dispatch one frozen case to M34, M39, or M37. Does not edit those modules."""

    key = _normalize_system_defect(defect)
    if case.family == "rag":
        return _invoke_rag(case, defect=key)
    if case.family == "agent":
        return _invoke_agent(case, defect=key)
    if case.family == "tool":
        return _invoke_tool(case, defect=key)
    raise ValueError(f"unknown family {case.family!r}")


def _critical_severity(grader: str) -> str:
    return "critical" if grader in CRITICAL_GRADERS else "major"


def grade_tool_schema(case: EvalCase, trace: InvocationTrace) -> GradeResult:
    """Fail when invalid arguments reach a tool, or an unknown tool ran."""

    m37 = _load_m37()
    registry = m37.default_registry()
    expect_error = bool(case.gold.get("expect_schema_error"))
    if trace.family == "tool":
        proposal = m37.parse_proposal(trace.proposal)
        validation = m37.validate_proposal(proposal, registry)
        issues = tuple(
            f"{issue.kind}:{issue.field}" for issue in validation.issues
        )
        if expect_error:
            passed = (not validation.ok) and (not bool(trace.execution_reached))
            localized = None
            if not passed:
                localized = (
                    "invalid_executed"
                    if trace.execution_reached
                    else (issues[0] if issues else "expected_schema_error")
                )
        else:
            passed = bool(validation.ok)
            localized = None if passed else (issues[0] if issues else "schema_failed")
        return GradeResult(
            grader="tool_schema",
            passed=passed,
            severity=_critical_severity("tool_schema"),
            localized_failure=localized,
            evidence={
                "schema_ok": validation.ok,
                "execution_reached": trace.execution_reached,
                "issues": list(issues),
                "tool_name": proposal.tool_name,
            },
        )
    unknown = [name for name in trace.executions if name not in registry.names()]
    passed = not unknown
    return GradeResult(
        grader="tool_schema",
        passed=passed,
        severity=_critical_severity("tool_schema"),
        localized_failure=None if passed else f"unregistered:{unknown[0]}",
        evidence={"executions": list(trace.executions), "unknown": unknown},
    )


def grade_citation_support(case: EvalCase, trace: InvocationTrace) -> GradeResult:
    """Fail when a cited span does not support the claim. Uses M34 verify_support."""

    del case
    m34 = _load_m34()
    if trace.rag is None:
        return GradeResult(
            grader="citation_support",
            passed=False,
            severity=_critical_severity("citation_support"),
            localized_failure="missing_rag_trace",
            evidence={},
        )
    support = m34.verify_support(trace.rag.answer, trace.rag.pack)
    issues = tuple(
        f"{issue.kind}:{issue.chunk_id}" for issue in support.issues
    )
    passed = bool(support.ok)
    return GradeResult(
        grader="citation_support",
        passed=passed,
        severity=_critical_severity("citation_support"),
        localized_failure=None if passed else (issues[0] if issues else "support_failed"),
        evidence={
            "support_ok": support.ok,
            "issues": list(issues),
            "citation_ids": list(trace.citation_ids),
            "status": trace.status,
        },
    )


def grade_state_termination(case: EvalCase, trace: InvocationTrace) -> GradeResult:
    """Fail when the terminal/degraded/circuit contract does not match gold."""

    gold = case.gold
    expected_terminal = gold.get("terminal")
    parts: list[str] = []
    if expected_terminal is not None and trace.terminal != expected_terminal:
        parts.append(f"terminal:{trace.terminal}!={expected_terminal}")
    if "degraded" in gold and bool(trace.degraded) != bool(gold["degraded"]):
        parts.append(f"degraded:{trace.degraded}!={gold['degraded']}")
    if "circuit_open" in gold and bool(trace.circuit_open) != bool(gold["circuit_open"]):
        parts.append(f"circuit_open:{trace.circuit_open}!={gold['circuit_open']}")
    if bool(gold.get("forbid_complete_when_degraded")) and trace.degraded and trace.terminal == "complete":
        parts.append("degraded_labeled_complete")
    passed = not parts
    return GradeResult(
        grader="state_termination",
        passed=passed,
        severity=_critical_severity("state_termination"),
        localized_failure=None if passed else parts[0],
        evidence={
            "terminal": trace.terminal,
            "degraded": trace.degraded,
            "circuit_open": trace.circuit_open,
            "attempts": trace.attempts,
            "expected_terminal": expected_terminal,
        },
    )


def grade_idempotency(case: EvalCase, trace: InvocationTrace) -> GradeResult:
    """Fail when a repeated side-effecting call posts twice."""

    gold_count = case.gold.get("effect_count")
    expected_replay = case.gold.get("second_replayed")
    parts: list[str] = []
    if gold_count is not None and trace.effect_count != int(gold_count):
        parts.append(f"double_post:effect_count={trace.effect_count}")
    if expected_replay is not None and bool(trace.replayed) != bool(expected_replay):
        parts.append(f"replayed:{trace.replayed}!={expected_replay}")
    if trace.effect_count is not None and int(trace.effect_count) > 1 and gold_count == 1:
        if "double_post:effect_count=" not in (parts[0] if parts else ""):
            parts.append(f"double_post:effect_count={trace.effect_count}")
    passed = not parts
    return GradeResult(
        grader="idempotency",
        passed=passed,
        severity=_critical_severity("idempotency"),
        localized_failure=None if passed else parts[0],
        evidence={
            "effect_count": trace.effect_count,
            "replayed": trace.replayed,
            "execution_reached": trace.execution_reached,
        },
    )


GRADER_IMPL = {
    "tool_schema": grade_tool_schema,
    "citation_support": grade_citation_support,
    "state_termination": grade_state_termination,
    "idempotency": grade_idempotency,
}


def grade_case(case: EvalCase, trace: InvocationTrace) -> tuple[GradeResult, ...]:
    results = []
    for name in case.graders:
        impl = GRADER_IMPL[name]
        results.append(impl(case, trace))
    return tuple(results)


def outcome_success(case: EvalCase, trace: InvocationTrace) -> bool:
    """Coarse task outcome. Does not substitute for invariant graders.

    RAG: status match only (answered vs abstained), not citation support.
    Agent: terminal match only, not posted amount.
    Tool schema: the call returned (schema error or success).
    Idempotency: the second call reported success, ignoring effect_count.
    """

    gold = case.gold
    if case.family == "rag":
        return trace.status == gold.get("status")
    if case.family == "agent":
        return trace.terminal == gold.get("terminal")
    if case.scenario == "tool_schema":
        return trace.tool_first is not None
    if case.scenario == "idempotency":
        second = trace.tool_second
        return second is not None and second.result.status == "success"
    return False


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    family: str
    scenario: str
    split: str
    severity: str
    task_success: bool
    invariant_pass: bool
    critical_fail: bool
    grades: tuple[GradeResult, ...]
    trace: InvocationTrace
    defect: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "scenario": self.scenario,
            "split": self.split,
            "severity": self.severity,
            "task_success": self.task_success,
            "invariant_pass": self.invariant_pass,
            "critical_fail": self.critical_fail,
            "grades": {grade.grader: grade.passed for grade in self.grades},
            "localized": {
                grade.grader: grade.localized_failure
                for grade in self.grades
                if grade.localized_failure
            },
            "status": self.trace.status,
            "terminal": self.trace.terminal,
            "citation_ids": list(self.trace.citation_ids),
            "support_ok": self.trace.support_ok,
            "route": self.trace.route,
            "degraded": self.trace.degraded,
            "attempts": self.trace.attempts,
            "effect_count": self.trace.effect_count,
            "replayed": self.trace.replayed,
            "execution_reached": self.trace.execution_reached,
            "schema_ok": self.trace.schema_ok,
            "step_count": self.trace.step_count,
            "cost_proxy": self.trace.cost_proxy,
            "retrieved_ids": list(self.trace.retrieval_ids),
            "used_memory_ids": list(self.trace.used_memory_ids),
            "posted_amount": self.trace.posted_amount,
            "defect": self.defect,
        }


@dataclass(frozen=True)
class SuiteReport:
    eval_version: str
    harness_version: str
    n: int
    n_task_success: int
    task_success_rate: float
    n_invariant_pass: int
    invariant_pass_rate: float
    n_critical_fail: int
    critical_fail_rate: float
    slice_fail_rates: dict[str, float]
    family_success: dict[str, float]
    component_rates: dict[str, float]
    mean_cost_proxy: float
    mean_step_count: float
    pack_contaminated: bool
    held_out_untuned: bool
    tuned_against: bool
    defect: str
    case_ids: tuple[str, ...]
    rows: tuple[CaseReport, ...]
    pack_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "eval_version": self.eval_version,
            "harness_version": self.harness_version,
            "n": self.n,
            "n_task_success": self.n_task_success,
            "task_success_rate": self.task_success_rate,
            "n_invariant_pass": self.n_invariant_pass,
            "invariant_pass_rate": self.invariant_pass_rate,
            "n_critical_fail": self.n_critical_fail,
            "critical_fail_rate": self.critical_fail_rate,
            "slice_fail_rates": dict(self.slice_fail_rates),
            "family_success": dict(self.family_success),
            "component_rates": dict(self.component_rates),
            "mean_cost_proxy": self.mean_cost_proxy,
            "mean_step_count": self.mean_step_count,
            "pack_contaminated": self.pack_contaminated,
            "held_out_untuned": self.held_out_untuned,
            "tuned_against": self.tuned_against,
            "defect": self.defect,
            "case_ids": list(self.case_ids),
            "pack_hash": self.pack_hash,
            "rows": [row.as_dict() for row in self.rows],
        }


def _rate(numer: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return numer / denom


def aggregate_report(
    rows: tuple[CaseReport, ...] | list[CaseReport],
    pack: EvalPack,
    *,
    defect: str = "none",
) -> SuiteReport:
    """Roll up outcome success, invariant slices, and cost proxies."""

    rows_t = tuple(rows)
    n = len(rows_t)
    n_task = sum(1 for row in rows_t if row.task_success)
    n_inv = sum(1 for row in rows_t if row.invariant_pass)
    n_crit = sum(1 for row in rows_t if row.critical_fail)
    slice_fail: dict[str, float] = {}
    for grader in DETERMINISTIC_GRADERS:
        relevant = [
            row
            for row in rows_t
            if any(grade.grader == grader for grade in row.grades)
        ]
        n_fail = sum(
            1
            for row in relevant
            if any(grade.grader == grader and not grade.passed for grade in row.grades)
        )
        slice_fail[grader] = _rate(n_fail, len(relevant))
    family_success: dict[str, float] = {}
    for family in FAMILIES:
        subset = [row for row in rows_t if row.family == family]
        family_success[family] = _rate(sum(1 for row in subset if row.task_success), len(subset))
    component_rates = {
        "invariant_pass": _rate(n_inv, n),
        "critical_fail": _rate(n_crit, n),
        "holdout_success": _rate(
            sum(1 for row in rows_t if row.split == "holdout" and row.task_success),
            sum(1 for row in rows_t if row.split == "holdout"),
        ),
    }
    mean_cost = sum(row.trace.cost_proxy for row in rows_t) / n if n else 0.0
    mean_steps = sum(row.trace.step_count for row in rows_t) / n if n else 0.0
    return SuiteReport(
        eval_version=pack.eval_version,
        harness_version=HARNESS_VERSION,
        n=n,
        n_task_success=n_task,
        task_success_rate=_rate(n_task, n),
        n_invariant_pass=n_inv,
        invariant_pass_rate=_rate(n_inv, n),
        n_critical_fail=n_crit,
        critical_fail_rate=_rate(n_crit, n),
        slice_fail_rates=slice_fail,
        family_success=family_success,
        component_rates=component_rates,
        mean_cost_proxy=mean_cost,
        mean_step_count=mean_steps,
        pack_contaminated=pack.contaminated,
        held_out_untuned=pack.held_out_untuned,
        tuned_against=pack.tuned_against,
        defect=defect,
        case_ids=tuple(row.case_id for row in rows_t),
        rows=rows_t,
        pack_hash=pack_fingerprint(pack),
    )


def slice_rates(report: SuiteReport) -> dict[str, float]:
    return dict(report.slice_fail_rates)


def run_suite(
    pack: EvalPack | None = None,
    *,
    defect: str | None = "none",
    families: tuple[str, ...] | None = None,
) -> SuiteReport:
    """Invoke every selected case, grade, and aggregate. Eval set is not edited."""

    used = pack if pack is not None else load_eval_pack(require_canonical=True)
    key = _normalize_system_defect(defect)
    wanted = FAMILIES if families is None else families
    rows: list[CaseReport] = []
    for case in used.cases:
        if case.family not in wanted:
            continue
        trace = invoke_case(case, defect=key)
        grades = grade_case(case, trace)
        invariant_pass = all(grade.passed for grade in grades)
        critical_fail = any(
            (not grade.passed) and grade.severity == "critical" for grade in grades
        )
        rows.append(
            CaseReport(
                case_id=case.case_id,
                family=case.family,
                scenario=case.scenario,
                split=case.split,
                severity=case.severity,
                task_success=outcome_success(case, trace),
                invariant_pass=invariant_pass,
                critical_fail=critical_fail,
                grades=grades,
                trace=trace,
                defect=trace.defect,
            )
        )
    return aggregate_report(rows, used, defect=key)


def inject_regression(
    pack: EvalPack | None = None,
    *,
    defect: str,
) -> SuiteReport:
    """Re-run the same pack with one named M34/M37–M39 defect. Pack is unchanged."""

    used = pack if pack is not None else load_eval_pack(require_canonical=True)
    key = _normalize_system_defect(defect)
    if key == "none":
        raise ValueError("inject_regression requires a named system defect")
    return run_suite(used, defect=key)


@dataclass(frozen=True)
class ReleaseDecision:
    passed: bool
    fail_reasons: tuple[str, ...]
    policy_name: str
    task_success_rate: float
    critical_fail_rate: float
    eval_version: str
    slice_fail_rates: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "fail_reasons": list(self.fail_reasons),
            "policy_name": self.policy_name,
            "task_success_rate": self.task_success_rate,
            "critical_fail_rate": self.critical_fail_rate,
            "eval_version": self.eval_version,
            "slice_fail_rates": dict(self.slice_fail_rates),
        }


def decide_release_gate(
    report: SuiteReport,
    policy: ReleasePolicy | None = None,
    *,
    rubric_only: bool = False,
) -> ReleaseDecision:
    """Ship/no-ship from version, contamination, averages, and slice gates.

    Rubric-only gating is rejected: deterministic invariant graders stay
    required. Fail reasons are explicit strings, not substring search.
    """

    if rubric_only:
        raise RubricCannotGateInvariants(
            "rubric/LLM-as-judge cannot be the sole required grader for release"
        )
    used = policy or CANONICAL_POLICY
    reasons: list[str] = []
    if used.require_eval_version and report.eval_version != used.require_eval_version:
        reasons.append(f"eval_version:{report.eval_version}")
    if used.reject_contaminated and (report.pack_contaminated or report.tuned_against):
        reasons.append("contaminated_pack")
    if report.task_success_rate < used.min_task_success:
        reasons.append(
            f"task_success:{report.task_success_rate:.3f}<{used.min_task_success}"
        )
    if report.critical_fail_rate > used.max_critical_fail_rate:
        reasons.append(
            f"critical_fail_rate:{report.critical_fail_rate:.3f}>{used.max_critical_fail_rate}"
        )
    for slice_name, limit in used.slice_max_fail:
        rate = float(report.slice_fail_rates.get(slice_name, 0.0))
        if rate > limit:
            reasons.append(f"slice:{slice_name}:{rate:.3f}>{limit}")
    return ReleaseDecision(
        passed=not reasons,
        fail_reasons=tuple(reasons),
        policy_name=used.name,
        task_success_rate=report.task_success_rate,
        critical_fail_rate=report.critical_fail_rate,
        eval_version=report.eval_version,
        slice_fail_rates=dict(report.slice_fail_rates),
    )


def local_rubric_score(dimension: str, trace: InvocationTrace) -> int:
    """Deterministic checklist. Not an invariant grader."""

    if dimension == "abstention_clarity":
        text = ""
        if trace.rag is not None:
            text = str(trace.rag.answer.text).lower()
        mentions_gap = "insufficient evidence" in text
        invents = "ceo" in text and "valley" in text
        if mentions_gap and not invents:
            return 2
        if mentions_gap or (trace.status == "abstained"):
            return 1
        return 0
    if dimension == "honest_degraded_label":
        if trace.degraded and trace.terminal == "degraded":
            return 2
        if trace.degraded or trace.terminal == "degraded":
            return 1
        return 0
    if dimension == "exclusion_reason_readable":
        reasons = {row["reason"] for row in trace.excluded}
        allowed = {
            "scope_mismatch",
            "sku_mismatch",
            "expired",
            "superseded",
            "no_retrieval_scope",
            "unscoped_entry",
            "unscoped_query",
        }
        if reasons and reasons <= allowed:
            return 2
        if reasons:
            return 1
        return 0
    if dimension == "answer_fluency":
        text = ""
        if trace.rag is not None:
            text = str(trace.rag.answer.text)
        # Heuristic overscores terse extractive copies. Gold may disagree.
        if len(text) >= 8:
            return 2
        if text:
            return 1
        return 0
    raise ValueError(f"unknown rubric dimension {dimension!r}")


def calibrate_rubric(
    labels: Mapping[str, Any] | None = None,
    *,
    traces_by_source: Mapping[str, InvocationTrace] | None = None,
) -> dict[str, Any]:
    """Score a frozen hand-labeled set. Record disagreement and limits."""

    payload = labels if labels is not None else load_rubric_labels()
    rows = []
    disagreements = 0
    for item in payload["cases"]:
        source_id = str(item["source_case"])
        dimension = str(item["dimension"])
        gold = int(item["gold"])
        trace = None if traces_by_source is None else traces_by_source.get(source_id)
        if trace is None:
            raise KeyError(f"missing trace for rubric source {source_id}")
        predicted = local_rubric_score(dimension, trace)
        disagree = predicted != gold
        if disagree:
            disagreements += 1
        rows.append(
            {
                "id": item["id"],
                "source_case": source_id,
                "dimension": dimension,
                "gold": gold,
                "predicted": predicted,
                "disagree": disagree,
            }
        )
    n = len(rows)
    return {
        "n": n,
        "n_disagree": disagreements,
        "disagreement_rate": _rate(disagreements, n),
        "deterministic_required_for_invariants": True,
        "llm_judge_required": False,
        "rows": rows,
        "limit": (
            "Rubric disagreement is a calibration signal. Citation support, "
            "tool schema, state termination, and idempotency stay deterministic."
        ),
    }


def optional_llm_judge(*args: Any, **kwargs: Any) -> Any:
    """Optional LLM-as-judge. Canonical validation must not call a vendor API.

    Named source: anthropic-evals. A judge adapter is optional, fail-closed,
    and cannot be the sole required grader.
    """

    del args, kwargs
    raise OptionalLLMJudgeUnavailable(
        "M40 canonical graders are deterministic; an LLM-as-judge adapter "
        "is optional, fail-closed, and not required for validation"
    )


@dataclass(frozen=True)
class AblationReport:
    removed: str
    remaining: tuple[str, ...]
    blocked_diagnoses: tuple[str, ...]
    still_possible: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "removed": self.removed,
            "remaining": list(self.remaining),
            "blocked_diagnoses": list(self.blocked_diagnoses),
            "still_possible": list(self.still_possible),
        }


def ablate_trace(obs: Mapping[str, Any], field: str) -> AblationReport:
    """Drop one observability field and name diagnoses that become impossible."""

    if field not in obs:
        raise KeyError(field)
    remaining = tuple(key for key in obs if key != field)
    blocked = ABLATION_DIAGNOSES.get(field, ())
    still = []
    for name, diagnoses in ABLATION_DIAGNOSES.items():
        if name == field:
            continue
        if name in remaining:
            still.extend(diagnoses)
    return AblationReport(
        removed=field,
        remaining=remaining,
        blocked_diagnoses=tuple(blocked),
        still_possible=tuple(still),
    )


@dataclass(frozen=True)
class FailureTrace:
    defect: str
    claim: str
    report: SuiteReport
    decision: ReleaseDecision
    pack_version: str
    pack_path: str
    system_defect: str
    policy_name: str
    initial_case_ids: tuple[str, ...]
    audit: dict[str, Any] = field(default_factory=dict)
    version: str = HARNESS_VERSION


def pipeline_with_defect(*, defect: str) -> FailureTrace:
    """Named eval-governance defect: hidden critical, or a tuned pack."""

    key = _normalize_governance_defect(defect)
    if key == "none":
        raise ValueError("pipeline_with_defect requires a named governance defect")

    if key == "hidden_critical":
        pack = load_eval_pack(require_canonical=True)
        report = run_suite(pack, defect=HIDDEN_CRITICAL_INJECT)
        decision = decide_release_gate(report, AGGREGATE_ONLY_POLICY)
        return FailureTrace(
            defect=key,
            claim="aggregate_hides_critical_citation",
            report=report,
            decision=decision,
            pack_version=pack.eval_version,
            pack_path=pack.path,
            system_defect=HIDDEN_CRITICAL_INJECT,
            policy_name=AGGREGATE_ONLY_POLICY.name,
            initial_case_ids=pack.case_ids,
            audit={
                "hidden_case": HIDDEN_CRITICAL_CASE,
                "hidden_grader": HIDDEN_CRITICAL_GRADER,
                "task_success_rate": report.task_success_rate,
                "critical_fail_rate": report.critical_fail_rate,
                "slice_citation": report.slice_fail_rates.get("citation_support"),
                "aggregate_passed": decision.passed,
                "n_critical_fail": report.n_critical_fail,
            },
        )

    if key == "contaminated_pack":
        pack = load_eval_pack(default_dataset_dir() / "contaminated_pack.json")
        report = run_suite(pack, defect="none")
        decision = decide_release_gate(report, AGGREGATE_ONLY_POLICY)
        return FailureTrace(
            defect=key,
            claim="eval_set_tuned_against",
            report=report,
            decision=decision,
            pack_version=pack.eval_version,
            pack_path=pack.path,
            system_defect="none",
            policy_name=AGGREGATE_ONLY_POLICY.name,
            initial_case_ids=pack.case_ids,
            audit={
                "contaminated": pack.contaminated,
                "tuned_against": pack.tuned_against,
                "eval_version": pack.eval_version,
                "dropped_holdout": True,
                "task_success_rate": report.task_success_rate,
                "canonical_version": EVAL_VERSION,
                "aggregate_passed": decision.passed,
            },
        )

    raise ValueError(f"unsupported defect {defect!r}")


def repair_run(trace: FailureTrace) -> FailureTrace:
    """Repair eval governance from the broken object.

    hidden_critical: apply canonical slice gates to the *same* suite report.
    contaminated_pack: reload the clean versioned pack and rerun the same
    system defect (none). Does not start an unrelated happy-path from defaults
    with a different case list mixed in.
    """

    if trace.defect not in GOVERNANCE_DEFECTS:
        raise ValueError(f"repair_run expects a governance defect, not {trace.defect!r}")

    if trace.defect == "hidden_critical":
        decision = decide_release_gate(trace.report, CANONICAL_POLICY)
        return FailureTrace(
            defect="none",
            claim="slice_gates_restored",
            report=trace.report,
            decision=decision,
            pack_version=trace.pack_version,
            pack_path=trace.pack_path,
            system_defect=trace.system_defect,
            policy_name=CANONICAL_POLICY.name,
            initial_case_ids=trace.initial_case_ids,
            audit={
                "from_defect": trace.defect,
                "same_report": True,
                "same_case_ids": list(trace.report.case_ids) == list(trace.initial_case_ids),
                "gate_passed": decision.passed,
                "fail_reasons": list(decision.fail_reasons),
                "critical_fail_rate": trace.report.critical_fail_rate,
            },
        )

    clean = load_eval_pack(require_canonical=True)
    report = run_suite(clean, defect=trace.system_defect)
    decision = decide_release_gate(report, CANONICAL_POLICY)
    return FailureTrace(
        defect="none",
        claim="clean_versioned_pack_restored",
        report=report,
        decision=decision,
        pack_version=clean.eval_version,
        pack_path=clean.path,
        system_defect=trace.system_defect,
        policy_name=CANONICAL_POLICY.name,
        initial_case_ids=trace.initial_case_ids,
        audit={
            "from_defect": trace.defect,
            "clean_version": clean.eval_version,
            "clean_contaminated": clean.contaminated,
            "gate_passed": decision.passed,
            "fail_reasons": list(decision.fail_reasons),
            "n": report.n,
            "held_out_untuned": report.held_out_untuned,
        },
    )


def numpy_slice_matrix(report: SuiteReport):
    """Optional NumPy parity helper. Required tests use stdlib rates."""

    np = _require_numpy()
    families = list(FAMILIES)
    values = np.array(
        [report.family_success.get(name, 0.0) for name in families],
        dtype=float,
    )
    return families, values


def observability_report(trace: InvocationTrace) -> dict[str, Any]:
    payload = trace.observability()
    payload.update(
        {
            "version": HARNESS_VERSION,
            "trace_fields": list(OBSERVABILITY_FIELDS),
            "scale_limit": SCALE_LIMIT,
            "handoff": HANDOFF,
            "inference_boundary": INFERENCE_BOUNDARY,
            "weights_updated": False,
        }
    )
    return payload


def handoff_contract() -> dict[str, Any]:
    return {
        "version": HARNESS_VERSION,
        "eval_version": EVAL_VERSION,
        "sources": list(CANONICAL_SOURCES),
        "content_bundle": CONTENT_BUNDLE,
        "graders": list(DETERMINISTIC_GRADERS),
        "taxonomy": {key: list(value) for key, value in SCENARIO_TAXONOMY.items()},
        "observability_fields": list(OBSERVABILITY_FIELDS),
        "release_policy": CANONICAL_POLICY.name,
        "llm_judge": "optional-fail-closed-not-sole-grader",
        "architecture_diagram": "deferred to M41",
        "v11_closed": False,
        "phase_end_honesty": (
            "P7 opens evaluation here. V11 does not close because this package exists."
        ),
        "handoff": HANDOFF,
        "scale_limit": SCALE_LIMIT,
    }


def graph_public() -> dict[str, Any]:
    return {
        "version": HARNESS_VERSION,
        "eval_version": EVAL_VERSION,
        "families": list(FAMILIES),
        "graders": list(DETERMINISTIC_GRADERS),
        "scenarios": {key: list(value) for key, value in SCENARIO_TAXONOMY.items()},
        "observability_fields": list(OBSERVABILITY_FIELDS),
        "hidden_critical_case": HIDDEN_CRITICAL_CASE,
        "canonical_policy": CANONICAL_POLICY.name,
    }
