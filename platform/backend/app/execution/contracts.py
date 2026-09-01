"""Typed job, status, and WP-137 structured-result contracts.

WP-137 required envelope: execution_id, status, exit_code, duration_ms, blocks.
Reproducibility metadata is an additional (schema-legal) field consumed by 31B/31D.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Literal, Mapping

ExecutionStatus = Literal["SUCCESS", "FAILED", "TIMEOUT", "CRASHED"]
JobStatus = Literal["PENDING", "RUNNING", "COMPLETED", "CANCELLED"]
BlockType = Literal[
    "table",
    "chart",
    "trace",
    "state_diff",
    "diagram",
    "markdown",
    "metric",
    "artifact",
]
JobKind = Literal["python", "notebook"]

EXECUTION_STATUSES: Final[tuple[str, ...]] = ("SUCCESS", "FAILED", "TIMEOUT", "CRASHED")
JOB_STATUSES: Final[tuple[str, ...]] = ("PENDING", "RUNNING", "COMPLETED", "CANCELLED")
BLOCK_TYPES: Final[tuple[str, ...]] = (
    "table",
    "chart",
    "trace",
    "state_diff",
    "diagram",
    "markdown",
    "metric",
    "artifact",
)
JOB_KINDS: Final[tuple[str, ...]] = ("python", "notebook")
WP137_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "execution_id",
    "status",
    "exit_code",
    "duration_ms",
    "blocks",
)

EXIT_SUCCESS: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_TIMEOUT: Final[int] = -1
EXIT_CRASHED: Final[int] = -1

PYTHON_RUNNER_ID: Final[str] = "learningos.python_inprocess.v1"
NOTEBOOK_RUNNER_ID: Final[str] = "learningos.notebook_adapter.v1"
TEST_RUNNER_ID: Final[str] = "learningos.test_harness.v1"
DEFAULT_TIMEOUT_SEC: Final[float] = 30.0


def canonical_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def hash_payload(value: Any) -> str:
    return sha256_hex(canonical_dumps(value))


def new_execution_id() -> str:
    return str(uuid.uuid4())


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


class ContractError(ValueError):
    """Invalid job/result contract (programmer or payload error)."""

    def __init__(
        self,
        message: str,
        *,
        path: str = "$",
        code: str = "CONTRACT_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.code = code
        self.details: dict[str, Any] = details or {}


def _require_status(value: Any, *, path: str) -> ExecutionStatus:
    if value not in EXECUTION_STATUSES:
        raise ContractError(
            f"Invalid execution status {value!r}",
            path=path,
            details={"expected": list(EXECUTION_STATUSES), "found": value},
        )
    return value  # type: ignore[return-value]


def _require_job_status(value: Any, *, path: str) -> JobStatus:
    if value not in JOB_STATUSES:
        raise ContractError(
            f"Invalid job status {value!r}",
            path=path,
            details={"expected": list(JOB_STATUSES), "found": value},
        )
    return value  # type: ignore[return-value]


def _require_block_type(value: Any, *, path: str) -> BlockType:
    if value not in BLOCK_TYPES:
        raise ContractError(
            f"Invalid block type {value!r}",
            path=path,
            details={"expected": list(BLOCK_TYPES), "found": value},
        )
    return value  # type: ignore[return-value]


def _require_job_kind(value: Any, *, path: str) -> JobKind:
    if value not in JOB_KINDS:
        raise ContractError(
            f"Invalid job kind {value!r}",
            path=path,
            details={"expected": list(JOB_KINDS), "found": value},
        )
    return value  # type: ignore[return-value]


def _as_int(value: Any, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"Expected integer at {path}", path=path, details={"found": value})
    if minimum is not None and value < minimum:
        raise ContractError(
            f"Expected integer >= {minimum} at {path}",
            path=path,
            details={"found": value},
        )
    return value


@dataclass(frozen=True)
class ResultBlock:
    type: str
    payload: Mapping[str, Any]
    title: str | None = None

    def __post_init__(self) -> None:
        _require_block_type(self.type, path="type")
        if not isinstance(self.payload, Mapping):
            raise ContractError("Block payload must be an object", path="payload")
        object.__setattr__(self, "payload", dict(self.payload))
        if self.title is not None and not isinstance(self.title, str):
            raise ContractError("Block title must be a string", path="title")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type, "payload": dict(self.payload)}
        if self.title is not None:
            payload["title"] = self.title
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, path: str = "blocks") -> ResultBlock:
        if not isinstance(raw, Mapping):
            raise ContractError("Block must be an object", path=path)
        title = raw.get("title")
        return cls(
            type=_require_block_type(raw.get("type"), path=f"{path}.type"),
            payload=raw.get("payload") if isinstance(raw.get("payload"), Mapping) else {},
            title=title if isinstance(title, str) else None,
        )


@dataclass(frozen=True)
class Diagnostics:
    stdout: str = ""
    stderr: str = ""
    system_metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stdout", str(self.stdout or ""))
        object.__setattr__(self, "stderr", str(self.stderr or ""))
        if self.system_metrics is not None:
            object.__setattr__(self, "system_metrics", dict(self.system_metrics))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"stdout": self.stdout, "stderr": self.stderr}
        if self.system_metrics:
            payload["system_metrics"] = dict(self.system_metrics)
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> Diagnostics | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ContractError("diagnostics must be an object", path="diagnostics")
        metrics = raw.get("system_metrics")
        return cls(
            stdout=str(raw.get("stdout") or ""),
            stderr=str(raw.get("stderr") or ""),
            system_metrics=metrics if isinstance(metrics, Mapping) else None,
        )


@dataclass(frozen=True)
class Reproducibility:
    python_version: str
    runner_id: str
    code_hash: str
    params_hash: str
    harness_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python_version": self.python_version,
            "runner_id": self.runner_id,
            "code_hash": self.code_hash,
            "params_hash": self.params_hash,
        }
        if self.harness_hash is not None:
            payload["harness_hash"] = self.harness_hash
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> Reproducibility | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ContractError("reproducibility must be an object", path="reproducibility")
        harness = raw.get("harness_hash")
        return cls(
            python_version=str(raw.get("python_version") or ""),
            runner_id=str(raw.get("runner_id") or ""),
            code_hash=str(raw.get("code_hash") or ""),
            params_hash=str(raw.get("params_hash") or ""),
            harness_hash=str(harness) if harness is not None else None,
        )


@dataclass(frozen=True)
class StructuredResult:
    execution_id: str
    status: str
    exit_code: int
    duration_ms: int
    blocks: tuple[ResultBlock, ...] = ()
    diagnostics: Diagnostics | None = None
    reproducibility: Reproducibility | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not self.execution_id:
            raise ContractError("execution_id must be a non-empty string", path="execution_id")
        _require_status(self.status, path="status")
        object.__setattr__(self, "exit_code", _as_int(self.exit_code, path="exit_code"))
        object.__setattr__(
            self, "duration_ms", _as_int(self.duration_ms, path="duration_ms", minimum=0)
        )
        blocks = tuple(self.blocks or ())
        for index, block in enumerate(blocks):
            if not isinstance(block, ResultBlock):
                raise ContractError("blocks items must be ResultBlock", path=f"blocks[{index}]")
        object.__setattr__(self, "blocks", blocks)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "execution_id": self.execution_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "blocks": [block.to_dict() for block in self.blocks],
        }
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics.to_dict()
        if self.reproducibility is not None:
            payload["reproducibility"] = self.reproducibility.to_dict()
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StructuredResult:
        if not isinstance(raw, Mapping):
            raise ContractError("Structured result must be an object", path="$")
        blocks_raw = raw.get("blocks")
        if not isinstance(blocks_raw, list):
            raise ContractError("blocks must be an array", path="blocks")
        parsed: list[ResultBlock] = []
        for index, item in enumerate(blocks_raw):
            if not isinstance(item, Mapping):
                raise ContractError("Block must be an object", path=f"blocks[{index}]")
            parsed.append(ResultBlock.from_mapping(item, path=f"blocks[{index}]"))
        blocks = tuple(parsed)
        diagnostics = raw.get("diagnostics")
        reproducibility = raw.get("reproducibility")
        return cls(
            execution_id=str(raw.get("execution_id") or ""),
            status=_require_status(raw.get("status"), path="status"),
            exit_code=_as_int(raw.get("exit_code"), path="exit_code"),
            duration_ms=_as_int(raw.get("duration_ms"), path="duration_ms", minimum=0),
            blocks=blocks,
            diagnostics=Diagnostics.from_mapping(diagnostics if isinstance(diagnostics, Mapping) else None),
            reproducibility=Reproducibility.from_mapping(
                reproducibility if isinstance(reproducibility, Mapping) else None
            ),
        )


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    memory_mb: int | None = None

    def __post_init__(self) -> None:
        timeout = self.timeout_sec
        if timeout is None:
            object.__setattr__(self, "timeout_sec", DEFAULT_TIMEOUT_SEC)
            return
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ContractError("timeout_sec must be a number", path="limits.timeout_sec")
        if float(timeout) <= 0:
            raise ContractError("timeout_sec must be > 0", path="limits.timeout_sec")
        object.__setattr__(self, "timeout_sec", float(timeout))
        if self.memory_mb is not None:
            object.__setattr__(self, "memory_mb", _as_int(self.memory_mb, path="limits.memory_mb", minimum=1))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"timeout_sec": self.timeout_sec}
        if self.memory_mb is not None:
            payload["memory_mb"] = self.memory_mb
        return payload


@dataclass(frozen=True)
class ExecutionJob:
    """JSON-serializable execution job. Production isolation is owned by 31B."""

    job_id: str
    kind: str = "python"
    source: str = ""
    entrypoint: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    execution_id: str | None = None
    runner_id: str = PYTHON_RUNNER_ID
    notebook: Mapping[str, Any] | None = None
    workdir: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ContractError("job_id must be a non-empty string", path="job_id")
        object.__setattr__(self, "kind", _require_job_kind(self.kind, path="kind"))
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "parameters", dict(self.parameters or {}))
        if self.notebook is not None:
            if not isinstance(self.notebook, Mapping):
                raise ContractError("notebook must be an object", path="notebook")
            object.__setattr__(self, "notebook", dict(self.notebook))
        if self.entrypoint is not None and not isinstance(self.entrypoint, str):
            raise ContractError("entrypoint must be a string", path="entrypoint")
        if not isinstance(self.limits, ExecutionLimits):
            raise ContractError("limits must be ExecutionLimits", path="limits")
        if self.kind == "notebook" and not self.runner_id:
            object.__setattr__(self, "runner_id", NOTEBOOK_RUNNER_ID)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "source": self.source,
            "parameters": dict(self.parameters),
            "limits": self.limits.to_dict(),
            "runner_id": self.runner_id,
        }
        if self.entrypoint is not None:
            payload["entrypoint"] = self.entrypoint
        if self.execution_id is not None:
            payload["execution_id"] = self.execution_id
        if self.notebook is not None:
            payload["notebook"] = dict(self.notebook)
        if self.workdir is not None:
            payload["workdir"] = self.workdir
        return payload

    @classmethod
    def create(
        cls,
        *,
        job_id: str | None = None,
        kind: str = "python",
        source: str = "",
        entrypoint: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        memory_mb: int | None = None,
        execution_id: str | None = None,
        runner_id: str | None = None,
        notebook: Mapping[str, Any] | None = None,
        workdir: str | None = None,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionJob:
        resolved_kind = _require_job_kind(kind, path="kind")
        default_runner = NOTEBOOK_RUNNER_ID if resolved_kind == "notebook" else PYTHON_RUNNER_ID
        return cls(
            job_id=job_id or new_job_id(),
            kind=resolved_kind,
            source=source,
            entrypoint=entrypoint,
            parameters=parameters or {},
            limits=limits or ExecutionLimits(timeout_sec=timeout_sec, memory_mb=memory_mb),
            execution_id=execution_id,
            runner_id=runner_id or default_runner,
            notebook=notebook,
            workdir=workdir,
        )


@dataclass(frozen=True)
class JobState:
    job_id: str
    status: str
    result: StructuredResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ContractError("job_id must be a non-empty string", path="job_id")
        _require_job_status(self.status, path="status")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_id": self.job_id, "status": self.status}
        if self.result is not None:
            payload["result"] = self.result.to_dict()
        return payload


def blocks_from_iterable(items: Iterable[Any]) -> tuple[ResultBlock, ...]:
    blocks: list[ResultBlock] = []
    for index, item in enumerate(items):
        if isinstance(item, ResultBlock):
            blocks.append(item)
        elif isinstance(item, Mapping):
            blocks.append(ResultBlock.from_mapping(item, path=f"blocks[{index}]"))
        else:
            raise ContractError("Block must be an object", path=f"blocks[{index}]")
    return tuple(blocks)
