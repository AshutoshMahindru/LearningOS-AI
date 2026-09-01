"""Assertion harness over a WP-137 structured result, with provenance hash."""

from __future__ import annotations

import inspect
import json
import operator
import platform
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from app.execution.contracts import (
    EXIT_FAILED,
    EXIT_SUCCESS,
    TEST_RUNNER_ID,
    Diagnostics,
    Reproducibility,
    ResultBlock,
    StructuredResult,
    new_execution_id,
    sha256_hex,
)
from app.execution.result_schema import validate_structured_result

AssertionCallable = Callable[[dict[str, Any]], Any]

_PATH_TOKEN = re.compile(r"\[(\d+)\]|([^.\[\]]+)")
_COMPARE = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


@dataclass(frozen=True)
class AssertionSpec:
    name: str
    path: str | None = None
    op: str = "eq"
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "op": self.op}
        if self.path is not None:
            payload["path"] = self.path
        if self.expected is not None or self.op != "exists":
            payload["expected"] = self.expected
        return payload


@dataclass(frozen=True)
class AssertionOutcome:
    name: str
    passed: bool
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class HarnessReport:
    passed: bool
    passed_count: int
    failed_count: int
    cases: tuple[AssertionOutcome, ...]
    harness_hash: str
    harness_id: str
    execution_id: str | None
    duration_ms: int
    runner_id: str = TEST_RUNNER_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "cases": [case.to_dict() for case in self.cases],
            "harness_hash": self.harness_hash,
            "harness_id": self.harness_id,
            "execution_id": self.execution_id,
            "duration_ms": self.duration_ms,
            "runner_id": self.runner_id,
        }

    def to_structured_result(self, *, execution_id: str | None = None) -> StructuredResult:
        result = StructuredResult(
            execution_id=execution_id or new_execution_id(),
            status="SUCCESS" if self.passed else "FAILED",
            exit_code=EXIT_SUCCESS if self.passed else EXIT_FAILED,
            duration_ms=self.duration_ms,
            blocks=(
                ResultBlock(
                    type="metric",
                    title="harness",
                    payload={
                        "passed": self.passed,
                        "passed_count": self.passed_count,
                        "failed_count": self.failed_count,
                        "harness_hash": self.harness_hash,
                        "harness_id": self.harness_id,
                    },
                ),
            ),
            diagnostics=Diagnostics(stdout="", stderr=""),
            reproducibility=Reproducibility(
                python_version=platform.python_version(),
                runner_id=self.runner_id,
                code_hash=self.harness_hash,
                params_hash=self.harness_hash,
                harness_hash=self.harness_hash,
            ),
        )
        validate_structured_result(result)
        return result


def _as_result_dict(result: StructuredResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, StructuredResult):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError("result must be a StructuredResult or mapping")


def resolve_path(data: Any, path: str) -> Any:
    if path in {"", "$", "."}:
        return data
    current = data
    for match in _PATH_TOKEN.finditer(path.lstrip("$").lstrip(".")):
        index, name = match.group(1), match.group(2)
        if index is not None:
            if not isinstance(current, (list, tuple)):
                raise KeyError(path)
            current = current[int(index)]
            continue
        token = name or ""
        if token == "":
            continue
        if isinstance(current, Mapping):
            current = current[token]
        elif isinstance(current, (list, tuple)) and token.isdigit():
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def _callable_source(fn: Callable[..., Any]) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))
        module = getattr(fn, "__module__", "")
        return f"{module}:{qualname}"


def _normalize_item(item: AssertionSpec | Mapping[str, Any] | AssertionCallable) -> dict[str, Any]:
    if callable(item) and not isinstance(item, AssertionSpec):
        name = getattr(item, "__qualname__", getattr(item, "__name__", "callable"))
        return {
            "kind": "callable",
            "name": str(name),
            "source": _callable_source(item),
        }
    if isinstance(item, AssertionSpec):
        payload = item.to_dict()
        payload["kind"] = "path"
        return payload
    if not isinstance(item, Mapping):
        raise TypeError("assertion must be a mapping, AssertionSpec, or callable")
    op = str(item.get("op") or "eq")
    expected = item.get("expected", item.get("value"))
    for key in ("equals", "eq", "ne", "gt", "gte", "lt", "lte", "contains", "exists", "typeof"):
        if key in item and key not in {"op", "path", "name", "expected", "value"}:
            op = "eq" if key in {"equals", "eq"} else key
            if key != "exists":
                expected = item[key]
            break
    name = item.get("name") or item.get("path") or "assertion"
    return {
        "kind": "path",
        "name": str(name),
        "path": item.get("path"),
        "op": op,
        "expected": expected,
    }


def hash_harness(
    assertions: Sequence[AssertionSpec | Mapping[str, Any] | AssertionCallable],
    *,
    harness_id: str = "default",
) -> str:
    material = {
        "runner_id": TEST_RUNNER_ID,
        "harness_id": harness_id,
        "assertions": [_normalize_item(item) for item in assertions],
    }
    return sha256_hex(
        json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    )


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if op == "exists":
        return True
    if op == "contains":
        return expected in actual
    if op == "typeof":
        return type(actual).__name__ == str(expected)
    compare = _COMPARE.get(op)
    if compare is None:
        raise ValueError(f"Unsupported assertion op {op!r}")
    if op in {"gt", "gte", "lt", "lte"}:
        return compare(float(actual), float(expected))
    return compare(actual, expected)


def _run_one(
    result: dict[str, Any],
    item: AssertionSpec | Mapping[str, Any] | AssertionCallable,
    spec: dict[str, Any],
) -> AssertionOutcome:
    name = str(spec.get("name") or "assertion")
    if spec.get("kind") == "callable":
        try:
            outcome = item(result)  # type: ignore[misc]
        except AssertionError as exc:
            return AssertionOutcome(name=name, passed=False, message=str(exc) or "assertion failed")
        except Exception as exc:
            return AssertionOutcome(name=name, passed=False, message=f"{type(exc).__name__}: {exc}")
        if outcome is False:
            return AssertionOutcome(name=name, passed=False, message="callable returned False")
        return AssertionOutcome(name=name, passed=True, message="ok")

    path = spec.get("path")
    op = str(spec.get("op") or "eq")
    expected = spec.get("expected")
    if not isinstance(path, str) or not path:
        return AssertionOutcome(name=name, passed=False, message="assertion path is required", path=path)
    try:
        actual = resolve_path(result, path)
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        if op == "exists":
            return AssertionOutcome(
                name=name,
                passed=False,
                message=f"missing path {path}",
                path=path,
            )
        return AssertionOutcome(
            name=name,
            passed=False,
            message=f"cannot resolve {path}: {type(exc).__name__}",
            path=path,
        )
    if op == "exists":
        return AssertionOutcome(name=name, passed=True, message="exists", path=path)
    try:
        ok = _compare(op, actual, expected)
    except Exception as exc:
        return AssertionOutcome(
            name=name,
            passed=False,
            message=f"compare failed: {type(exc).__name__}: {exc}",
            path=path,
        )
    if ok:
        return AssertionOutcome(name=name, passed=True, message="ok", path=path)
    return AssertionOutcome(
        name=name,
        passed=False,
        message=f"{path} {op} {expected!r} (actual {actual!r})",
        path=path,
    )


def run_harness(
    result: StructuredResult | Mapping[str, Any],
    assertions: Sequence[AssertionSpec | Mapping[str, Any] | AssertionCallable],
    *,
    harness_id: str = "default",
) -> HarnessReport:
    started = time.perf_counter()
    payload = _as_result_dict(result)
    digest = hash_harness(assertions, harness_id=harness_id)
    cases = tuple(
        _run_one(payload, item, _normalize_item(item)) for item in assertions
    )
    passed_count = sum(1 for case in cases if case.passed)
    failed_count = len(cases) - passed_count
    duration_ms = max(0, int((time.perf_counter() - started) * 1000))
    execution_id = payload.get("execution_id")
    return HarnessReport(
        passed=failed_count == 0,
        passed_count=passed_count,
        failed_count=failed_count,
        cases=cases,
        harness_hash=digest,
        harness_id=harness_id,
        execution_id=str(execution_id) if execution_id is not None else None,
        duration_ms=duration_ms,
    )
