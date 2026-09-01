"""In-process Python runner used as a library by tests and by 31B.

This is not a production sandbox. Isolated production exec lives in the 31B worker.
"""

from __future__ import annotations

import ast
import builtins as builtins_mod
import inspect
import math
import os
import platform
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.execution.contracts import (
    BLOCK_TYPES,
    DEFAULT_TIMEOUT_SEC,
    EXIT_CRASHED,
    EXIT_FAILED,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    NOTEBOOK_RUNNER_ID,
    PYTHON_RUNNER_ID,
    ContractError,
    Diagnostics,
    ExecutionJob,
    Reproducibility,
    ResultBlock,
    StructuredResult,
    hash_payload,
    new_execution_id,
    sha256_hex,
)
from app.execution.result_schema import ResultSchemaError, validate_structured_result

IN_PROCESS_LIBRARY_ONLY = True


class ExecutionTimeout(TimeoutError):
    """Snippet exceeded timeout_sec. Distinct from user TimeoutError."""


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        return str(value)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = _jsonable(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return str(value)


def _as_payload(value: Any) -> dict[str, Any]:
    jsonable = _jsonable(value)
    if isinstance(jsonable, dict):
        return jsonable
    return {"value": jsonable}


def _callable_source(fn: Callable[..., Any]) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))
        module = getattr(fn, "__module__", "")
        return f"{module}:{qualname}"


def hash_callable(fn: Callable[..., Any]) -> str:
    return sha256_hex(_callable_source(fn))


def _rss_to_mb(ru_maxrss: int) -> float:
    if sys.platform == "darwin":
        return float(ru_maxrss) / (1024.0 * 1024.0)
    return float(ru_maxrss) / 1024.0


def _system_metrics(start_ru: Any, end_ru: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if start_ru is None or end_ru is None:
        return metrics
    cpu = (end_ru.ru_utime + end_ru.ru_stime) - (start_ru.ru_utime + start_ru.ru_stime)
    metrics["cpu_time_ms"] = max(0.0, cpu * 1000.0)
    metrics["memory_peak_mb"] = max(0.0, _rss_to_mb(int(end_ru.ru_maxrss)))
    return metrics


def _rusage() -> Any:
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF)


@contextmanager
def _maybe_chdir(workdir: str | Path | None):
    if workdir is None:
        yield
        return
    path = Path(workdir)
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _with_timeout(fn: Callable[[], Any], timeout_sec: float | None) -> Any:
    if timeout_sec is None:
        return fn()
    timeout = float(timeout_sec)
    if timeout <= 0:
        raise ExecutionTimeout(f"timeout_sec must be > 0, got {timeout_sec}")
    use_alarm = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
        and hasattr(signal, "SIGALRM")
    )
    if not use_alarm:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout as exc:
                raise ExecutionTimeout(f"execution exceeded {timeout}s") from exc

    def _on_alarm(_signum: int, _frame: object) -> None:
        raise ExecutionTimeout(f"execution exceeded {timeout}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _exec_source(source: str, namespace: dict[str, Any], filename: str) -> Any:
    tree = ast.parse(source, filename=filename)
    if not tree.body:
        return None
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
        ast.fix_missing_locations(prefix)
        if prefix.body:
            exec(compile(prefix, filename, "exec"), namespace, namespace)
        expr = ast.Expression(last.value)
        ast.fix_missing_locations(expr)
        return eval(compile(expr, filename, "eval"), namespace, namespace)
    exec(compile(tree, filename, "exec"), namespace, namespace)
    return None


def _call_with_parameters(fn: Callable[..., Any], parameters: Mapping[str, Any] | None) -> Any:
    params = dict(parameters or {})
    if not params:
        return fn()
    try:
        return fn(**params)
    except TypeError:
        return fn(params)


def _traceback_block(exc: BaseException) -> ResultBlock:
    return ResultBlock(
        type="trace",
        title=type(exc).__name__,
        payload={
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


class ExecSession:
    """Shared in-process namespace used by the Python runner and notebook adapter."""

    def __init__(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
        runner_id: str = PYTHON_RUNNER_ID,
        workdir: str | Path | None = None,
        code_hash: str = "",
        params_hash: str | None = None,
    ) -> None:
        self.parameters = dict(parameters or {})
        self.runner_id = runner_id
        self.workdir = Path(workdir) if workdir is not None else None
        self.code_hash = code_hash
        self.params_hash = (
            params_hash if params_hash is not None else hash_payload(_jsonable(self.parameters))
        )
        self.blocks: list[ResultBlock] = []
        self.namespace: dict[str, Any] = self._make_namespace()

    def _make_namespace(self) -> dict[str, Any]:
        def emit(block_type: str, payload: Any, title: str | None = None) -> None:
            self.emit(block_type, payload, title=title)

        builtin_ns = dict(builtins_mod.__dict__)
        builtin_ns["emit"] = emit
        namespace: dict[str, Any] = {
            "__name__": "__learningos_exec__",
            "__builtins__": builtin_ns,
            "emit": emit,
            "parameters": dict(self.parameters),
        }
        if self.workdir is not None:
            namespace["__file__"] = str(self.workdir / "_snippet.py")
        return namespace

    def emit(self, block_type: str, payload: Any, title: str | None = None) -> None:
        if block_type not in BLOCK_TYPES:
            raise ContractError(
                f"Unknown block type {block_type!r}",
                path="blocks.type",
                details={"expected": list(BLOCK_TYPES), "found": block_type},
            )
        block = ResultBlock(
            type=block_type,
            payload=_as_payload(payload),
            title=title if title is None or isinstance(title, str) else str(title),
        )
        self.blocks.append(block)

    def ingest(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, ResultBlock):
            self.blocks.append(value)
            return
        if isinstance(value, Mapping) and value.get("type") in BLOCK_TYPES:
            payload = value.get("payload")
            title = value.get("title")
            self.emit(
                str(value["type"]),
                payload if payload is not None else {},
                title=title if isinstance(title, str) else None,
            )
            return
        if isinstance(value, (list, tuple)) and value:
            if all(
                isinstance(item, ResultBlock)
                or (isinstance(item, Mapping) and item.get("type") in BLOCK_TYPES)
                for item in value
            ):
                for item in value:
                    self.ingest(item)
                return
        self.emit("metric", {"value": _jsonable(value)}, title="return")

    def run_source(self, source: str, filename: str = "<learningos:python_runner>") -> Any:
        return _exec_source(source, self.namespace, filename)

    def run_entrypoint(self, entrypoint: str) -> Any:
        target = self.namespace.get(entrypoint)
        if not callable(target):
            raise NameError(f"entrypoint {entrypoint!r} is not a callable")
        return _call_with_parameters(target, self.parameters)


def _finalize(
    session: ExecSession,
    *,
    execution_id: str,
    status: str,
    exit_code: int,
    duration_ms: int,
    stdout: str,
    stderr: str,
    metrics: Mapping[str, float] | None,
) -> StructuredResult:
    result = StructuredResult(
        execution_id=execution_id,
        status=status,
        exit_code=exit_code,
        duration_ms=max(0, int(duration_ms)),
        blocks=tuple(session.blocks),
        diagnostics=Diagnostics(stdout=stdout, stderr=stderr, system_metrics=metrics),
        reproducibility=Reproducibility(
            python_version=platform.python_version(),
            runner_id=session.runner_id,
            code_hash=session.code_hash,
            params_hash=session.params_hash,
        ),
    )
    try:
        validate_structured_result(result)
    except ResultSchemaError:
        fallback = StructuredResult(
            execution_id=execution_id,
            status="CRASHED",
            exit_code=EXIT_CRASHED,
            duration_ms=max(0, int(duration_ms)),
            blocks=(),
            diagnostics=Diagnostics(
                stdout=stdout,
                stderr=stderr + "\nWP-137 validation failed for runner output\n",
                system_metrics=metrics,
            ),
            reproducibility=Reproducibility(
                python_version=platform.python_version(),
                runner_id=session.runner_id,
                code_hash=session.code_hash,
                params_hash=session.params_hash,
            ),
        )
        validate_structured_result(fallback)
        return fallback
    return result


def run_session(
    session: ExecSession,
    body: Callable[[], Any],
    *,
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC,
    execution_id: str | None = None,
) -> StructuredResult:
    exec_id = execution_id or new_execution_id()
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    started = time.perf_counter()
    start_ru = _rusage()
    status = "SUCCESS"
    exit_code = EXIT_SUCCESS
    caught: BaseException | None = None

    def invoke() -> Any:
        return body()

    try:
        with _maybe_chdir(session.workdir), redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            value = _with_timeout(invoke, timeout_sec)
        session.ingest(value)
    except ExecutionTimeout as exc:
        status = "TIMEOUT"
        exit_code = EXIT_TIMEOUT
        caught = exc
        stderr_buf.write(f"\n{exc}\n")
        session.blocks.append(_traceback_block(exc))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        status = "FAILED"
        exit_code = EXIT_FAILED
        caught = exc
        stderr_buf.write(traceback.format_exc())
        session.blocks.append(_traceback_block(exc))
    except BaseException as exc:
        status = "CRASHED"
        exit_code = EXIT_CRASHED
        caught = exc
        stderr_buf.write(traceback.format_exc())
        session.blocks.append(_traceback_block(exc))
    duration_ms = max(0, int((time.perf_counter() - started) * 1000))
    metrics = _system_metrics(start_ru, _rusage())
    if caught is not None and isinstance(caught, SystemExit):
        code = caught.code
        if isinstance(code, int):
            exit_code = code if status == "CRASHED" else exit_code
        elif code not in (None, False):
            exit_code = EXIT_CRASHED
    return _finalize(
        session,
        execution_id=exec_id,
        status=status,
        exit_code=exit_code,
        duration_ms=duration_ms,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        metrics=metrics,
    )


def run_source(
    source: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    entrypoint: str | None = None,
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC,
    execution_id: str | None = None,
    workdir: str | Path | None = None,
    runner_id: str = PYTHON_RUNNER_ID,
    filename: str = "<learningos:python_runner>",
) -> StructuredResult:
    session = ExecSession(
        parameters=parameters,
        runner_id=runner_id,
        workdir=workdir,
        code_hash=sha256_hex(source),
    )

    def body() -> Any:
        value = session.run_source(source, filename=filename)
        if entrypoint:
            return session.run_entrypoint(entrypoint)
        return value

    return run_session(session, body, timeout_sec=timeout_sec, execution_id=execution_id)


def run_callable(
    fn: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC,
    execution_id: str | None = None,
    workdir: str | Path | None = None,
    runner_id: str = PYTHON_RUNNER_ID,
) -> StructuredResult:
    params = dict(parameters or {})
    call_kwargs = dict(kwargs or {})
    hashed = params if parameters is not None else call_kwargs
    session = ExecSession(
        parameters=params,
        runner_id=runner_id,
        workdir=workdir,
        code_hash=hash_callable(fn),
        params_hash=hash_payload(_jsonable(hashed)),
    )
    session.namespace[getattr(fn, "__name__", "fn")] = fn

    def body() -> Any:
        if args or call_kwargs:
            return fn(*args, **call_kwargs)
        return _call_with_parameters(fn, params)

    return run_session(session, body, timeout_sec=timeout_sec, execution_id=execution_id)


def run_job(job: ExecutionJob) -> StructuredResult:
    if job.kind == "notebook" or job.notebook is not None:
        from app.execution.notebook_adapter import run_notebook

        notebook = job.notebook if job.notebook is not None else {"cells": [{"cell_type": "code", "source": job.source}]}
        return run_notebook(
            notebook,
            parameters=job.parameters,
            timeout_sec=job.limits.timeout_sec,
            execution_id=job.execution_id,
            workdir=job.workdir,
            runner_id=job.runner_id or NOTEBOOK_RUNNER_ID,
        )
    return run_source(
        job.source,
        parameters=job.parameters,
        entrypoint=job.entrypoint,
        timeout_sec=job.limits.timeout_sec,
        execution_id=job.execution_id,
        workdir=job.workdir,
        runner_id=job.runner_id or PYTHON_RUNNER_ID,
    )
