#!/usr/bin/env python3
"""Restricted learner-code child process. Spawned by platform/worker/sandbox.py.

This file is the only process that may exec learner snippets. The worker daemon
never calls exec/eval/compile and never imports sqlite3.

Network policy (best-effort deny):
- Parent starts this process with a scrubbed environment (no proxy/API keys).
- Imports of socket, ssl, http, urllib, requests, and similar are rejected.
- On Darwin the parent may wrap the process in sandbox-exec denying network-*.
- On Linux the parent may unshare the network namespace when permitted.
- This is not a full seccomp/MAC sandbox; it blocks obvious network use.

Path policy: open() and writes are confined to the job workdir. Traversal (".."),
symlink escape, Git worktree writes, and $LEARNINGOS_HOME/*.db are refused.
"""

from __future__ import annotations

import builtins
import io as _stdlib_io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ALLOWED_MODULES = frozenset(
    {
        "abc",
        "array",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "collections.abc",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "hmac",
        "io",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "pprint",
        "random",
        "re",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "time",
        "types",
        "typing",
        "unicodedata",
        "warnings",
    }
)

BANNED_MODULES = frozenset(
    {
        "_thread",
        "_posixsubprocess",
        "asyncio",
        "builtins",
        "code",
        "codeop",
        "concurrent",
        "ctypes",
        "fcntl",
        "ftplib",
        "http",
        "importlib",
        "inspect",
        "mmap",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "pkgutil",
        "posix",
        "pty",
        "requests",
        "resource",
        "runpy",
        "shelve",
        "shutil",
        "signal",
        "site",
        "smtplib",
        "socket",
        "sqlite3",
        "ssl",
        "subprocess",
        "sys",
        "syslog",
        "telnetlib",
        "termios",
        "threading",
        "tty",
        "urllib",
        "webbrowser",
        "xmlrpc",
    }
)

_REAL_EXEC = builtins.exec
_REAL_COMPILE = builtins.compile
_REAL_OPEN = builtins.open
_REAL_FILEIO = _stdlib_io.FileIO

ALLOWED_BUILTIN_NAMES = frozenset(
    {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "EOFError",
        "Ellipsis",
        "Exception",
        "False",
        "FileNotFoundError",
        "FloatingPointError",
        "GeneratorExit",
        "ImportError",
        "IndentationError",
        "IndexError",
        "IsADirectoryError",
        "KeyError",
        "LookupError",
        "MemoryError",
        "NameError",
        "None",
        "NotADirectoryError",
        "NotImplemented",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "PermissionError",
        "RuntimeError",
        "StopIteration",
        "SyntaxError",
        "TimeoutError",
        "True",
        "TypeError",
        "UnboundLocalError",
        "UnicodeError",
        "ValueError",
        "ZeroDivisionError",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
)


class SandboxViolation(Exception):
    """Learner code attempted a blocked operation."""


def _apply_resource_limits(memory_mb: int, cpu_sec: int) -> None:
    try:
        import resource
    except ImportError:
        return

    def _set(name: str, value: int) -> None:
        lim = getattr(resource, name, None)
        if lim is None:
            return
        try:
            resource.setrlimit(lim, (value, value))
            return
        except (ValueError, OSError):
            pass
        try:
            _soft, hard = resource.getrlimit(lim)
            chosen = value if hard in (-1, resource.RLIM_INFINITY) else min(value, hard)
            resource.setrlimit(lim, (chosen, hard))
        except (ValueError, OSError):
            return

    mem = max(16, int(memory_mb)) * 1024 * 1024
    cpu = max(1, int(cpu_sec))
    _set("RLIMIT_AS", mem)
    _set("RLIMIT_DATA", mem)
    _set("RLIMIT_CPU", cpu)
    _set("RLIMIT_CORE", 0)
    _set("RLIMIT_FSIZE", 32 * 1024 * 1024)
    _set("RLIMIT_NOFILE", 64)
    _set("RLIMIT_NPROC", 32)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _assert_allowed_path(
    raw: str | os.PathLike[str],
    *,
    workdir: Path,
    repo_root: Path | None,
    data_home: Path | None,
    writing: bool,
) -> Path:
    text = os.fspath(raw)
    if not text:
        raise SandboxViolation("empty path is not allowed")
    candidate = Path(text)
    if ".." in candidate.parts:
        raise SandboxViolation("path traversal rejected")
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (workdir / candidate).resolve()
    except OSError as exc:
        raise SandboxViolation(f"path resolve failed: {exc}") from exc
    if data_home is not None:
        home = data_home.resolve()
        if resolved.name == "learningos.db" or (
            resolved.suffix.lower() == ".db" and resolved.parent == home
        ):
            raise SandboxViolation("refusing LEARNINGOS_HOME database path")
    if writing and repo_root is not None and _is_within(resolved, repo_root):
        raise SandboxViolation("refusing write into the Git worktree")
    if not _is_within(resolved, workdir):
        raise SandboxViolation("path is outside the job workdir")
    return resolved


def _blocked(name: str):  # type: ignore[no-untyped-def]
    def _fn(*_args: object, **_kwargs: object) -> None:
        raise SandboxViolation(f"{name} is not allowed in the sandbox")

    _fn.__name__ = name
    return _fn


def _make_path_guards(spec: dict[str, Any]) -> tuple[Any, Any, Any]:
    workdir = Path(spec["workdir"]).resolve()
    repo_root = Path(spec["repo_root"]).resolve() if spec.get("repo_root") else None
    data_home = Path(spec["data_home"]).resolve() if spec.get("data_home") else None
    original_import = builtins.__import__

    def safe_open(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(file, int):
            raise SandboxViolation("opening raw file descriptors is not allowed")
        writing = any(flag in str(mode) for flag in ("w", "a", "x", "+"))
        _assert_allowed_path(
            file,
            workdir=workdir,
            repo_root=repo_root,
            data_home=data_home,
            writing=writing,
        )
        return _REAL_OPEN(file, mode, *args, **kwargs)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if level != 0:
            raise SandboxViolation("relative imports are not allowed")
        root = str(name).split(".", 1)[0]
        if root in BANNED_MODULES or root not in ALLOWED_MODULES:
            raise SandboxViolation(f"import of {name!r} is blocked")
        module = original_import(name, globals, locals, fromlist, level)
        if root in {"io", "_io"}:
            _patch_io_module(module, safe_open, guarded_fileio)
        return module

    guarded_fileio = _make_guarded_fileio(spec)
    return safe_open, safe_import, guarded_fileio


def _make_guarded_fileio(spec: dict[str, Any]) -> Any:
    """Factory wrapping io.FileIO so the real constructor never runs on a denied path."""
    workdir = Path(spec["workdir"]).resolve()
    repo_root = Path(spec["repo_root"]).resolve() if spec.get("repo_root") else None
    data_home = Path(spec["data_home"]).resolve() if spec.get("data_home") else None

    def guarded_fileio(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(file, int):
            raise SandboxViolation("opening raw file descriptors is not allowed")
        writing = any(flag in str(mode) for flag in ("w", "a", "x", "+"))
        _assert_allowed_path(
            file,
            workdir=workdir,
            repo_root=repo_root,
            data_home=data_home,
            writing=writing,
        )
        return _REAL_FILEIO(file, mode, *args, **kwargs)

    guarded_fileio.__name__ = "FileIO"
    guarded_fileio.__qualname__ = "FileIO"
    return guarded_fileio


def _patch_io_module(module: Any, safe_open: Any, guarded_fileio: Any) -> None:
    if module is None:
        return
    if hasattr(module, "open"):
        module.open = safe_open
    if hasattr(module, "FileIO"):
        module.FileIO = guarded_fileio


def _patch_open_primitives(safe_open: Any, guarded_fileio: Any) -> None:
    """Wrap open and FileIO aliases on every exec path (including 31A)."""
    builtins.open = safe_open  # type: ignore[assignment]
    import io as io_mod

    _patch_io_module(io_mod, safe_open, guarded_fileio)
    try:
        import _io as io_c

        _patch_io_module(io_c, safe_open, guarded_fileio)
    except ImportError:
        pass
    for name in ("io", "_io"):
        cached = sys.modules.get(name)
        if cached is not None:
            _patch_io_module(cached, safe_open, guarded_fileio)


def _restricted_builtins(spec: dict[str, Any]) -> dict[str, Any]:
    safe_open, safe_import, _guarded_fileio = _make_path_guards(spec)
    restricted: dict[str, Any] = {"__import__": safe_import, "open": safe_open}
    for name in ALLOWED_BUILTIN_NAMES:
        if hasattr(builtins, name):
            restricted[name] = getattr(builtins, name)
    return restricted


def _install_path_guards(spec: dict[str, Any], *, block_dynamic_exec: bool = True) -> None:
    """Wrap every open() alias so 31A runners cannot escape the job dir.

    31A's in-process runner copies builtins and uses exec/eval/compile for the
    snippet (block_dynamic_exec=False). Isolation is the subprocess, rlimits,
    and path-guarded open — including io.open / io.FileIO / _io.*, not only
    builtins.open.
    """
    safe_open, _safe_import, guarded_fileio = _make_path_guards(spec)
    _patch_open_primitives(safe_open, guarded_fileio)
    if block_dynamic_exec:
        builtins.eval = _blocked("eval")  # type: ignore[assignment]
        builtins.exec = _blocked("exec")  # type: ignore[assignment]
        builtins.compile = _blocked("compile")  # type: ignore[assignment]


def _try_upstream_runner(spec: dict[str, Any]) -> Any | None:
    backend_root = spec.get("backend_root")
    if backend_root and backend_root not in sys.path:
        sys.path.insert(0, str(backend_root))
    try:
        from app.execution.python_runner import run_job
    except ImportError:
        return None
    return run_job


def _job_from_spec(spec: dict[str, Any]) -> Any:
    """Adapt the sandbox spec dict to 31A ExecutionJob."""
    from app.execution.contracts import ExecutionJob

    parameters = spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {}
    try:
        timeout_sec = float(spec.get("timeout_sec") or 30)
    except (TypeError, ValueError):
        timeout_sec = 30.0
    if timeout_sec <= 0:
        timeout_sec = 30.0
    memory_raw = spec.get("memory_mb")
    try:
        memory_mb = int(memory_raw) if memory_raw is not None else None
    except (TypeError, ValueError):
        memory_mb = None
    entrypoint = spec.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        entrypoint = None
    workdir = spec.get("workdir")
    kind = "notebook" if isinstance(spec.get("notebook"), dict) else "python"
    execution_id = spec.get("execution_id")
    runner_id = spec.get("runner_id")
    return ExecutionJob.create(
        job_id=str(spec.get("job_id") or "job"),
        kind=kind,
        source=str(spec.get("code") or spec.get("source") or ""),
        entrypoint=entrypoint,
        parameters=parameters,
        timeout_sec=timeout_sec,
        memory_mb=memory_mb,
        execution_id=str(execution_id) if execution_id else None,
        runner_id=str(runner_id) if runner_id else None,
        notebook=spec.get("notebook") if isinstance(spec.get("notebook"), dict) else None,
        workdir=str(workdir) if workdir else None,
    )


def _restricted_exec(spec: dict[str, Any]) -> dict[str, Any]:
    code = spec.get("code") or ""
    parameters = spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {}
    namespace: dict[str, Any] = {
        "__name__": "__sandbox__",
        "__builtins__": _restricted_builtins(spec),
        "parameters": parameters,
    }
    _REAL_EXEC(_REAL_COMPILE(code, "<learner>", "exec"), namespace, namespace)  # noqa: S102 — isolated child
    result_value = namespace.get("result")
    payload: dict[str, Any] = {"status": "SUCCESS", "exit_code": 0}
    if result_value is not None:
        try:
            json.dumps(result_value)
        except (TypeError, ValueError):
            payload["result"] = repr(result_value)
        else:
            payload["result"] = result_value
    return payload


def _normalize_upstream(raw: Any) -> dict[str, Any]:
    if raw is not None and callable(getattr(raw, "to_dict", None)):
        payload = raw.to_dict()
        out = dict(payload) if isinstance(payload, dict) else {"result": payload}
    elif isinstance(raw, dict):
        out = dict(raw)
    else:
        return {"status": "SUCCESS", "exit_code": 0, "result": raw}
    out.setdefault("status", "SUCCESS")
    out.setdefault("exit_code", 0 if out.get("status") == "SUCCESS" else 1)
    diagnostics = out.get("diagnostics") if isinstance(out.get("diagnostics"), dict) else {}
    out.setdefault("stdout", str(diagnostics.get("stdout") or ""))
    out.setdefault("stderr", str(diagnostics.get("stderr") or ""))
    return out


def _invoke_upstream(upstream: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Run 31A run_job(ExecutionJob) inside the isolated child.

    WP-137 schema is loaded before path guards so validation does not open
    architecture files through the job-dir open() wrap.
    """
    try:
        from app.execution.result_schema import load_result_schema

        load_result_schema()
    except Exception:
        pass
    _install_path_guards(spec, block_dynamic_exec=False)
    return _normalize_upstream(upstream(_job_from_spec(spec)))


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("restricted_child: missing spec path", file=sys.stderr)
        return 2
    spec_path = Path(args[0])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    workdir = Path(spec["workdir"])
    result_path = Path(spec.get("result_path") or (workdir / ".sandbox" / "result.json"))
    stdout_path = Path(spec.get("stdout_path") or (workdir / ".sandbox" / "stdout.log"))
    stderr_path = Path(spec.get("stderr_path") or (workdir / ".sandbox" / "stderr.log"))
    memory_mb = int(spec.get("memory_mb") or 2048)
    cpu_sec = int(spec.get("cpu_sec") or max(1, int(float(spec.get("timeout_sec") or 30))))
    _apply_resource_limits(memory_mb, cpu_sec)
    os.chdir(str(workdir))
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    status = "FAILED"
    exit_code = 1
    payload: dict[str, Any]
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out_f, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err_f:
        sys.stdout = out_f
        sys.stderr = err_f
        try:
            upstream = _try_upstream_runner(spec)
            if upstream is not None:
                try:
                    payload = _invoke_upstream(upstream, spec)
                except (TypeError, ValueError, AttributeError):
                    _install_path_guards(spec, block_dynamic_exec=True)
                    payload = _restricted_exec(spec)
            else:
                _install_path_guards(spec, block_dynamic_exec=True)
                payload = _restricted_exec(spec)
            status = str(payload.get("status") or "SUCCESS")
            exit_code = int(payload.get("exit_code") or 0)
        except SandboxViolation as exc:
            payload = {"status": "DENIED", "exit_code": 1, "reason": str(exc)}
            status = "DENIED"
            exit_code = 1
            print(f"SandboxViolation: {exc}", file=err_f)
        except SyntaxError as exc:
            payload = {"status": "FAILED", "exit_code": 1, "reason": f"syntax error: {exc}"}
            print(traceback.format_exc(), file=err_f)
        except MemoryError:
            payload = {"status": "CRASHED", "exit_code": 1, "reason": "memory limit exceeded"}
            print("MemoryError", file=err_f)
        except Exception as exc:
            payload = {"status": "FAILED", "exit_code": 1, "reason": f"{type(exc).__name__}: {exc}"}
            print(traceback.format_exc(), file=err_f)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            out_f.flush()
            err_f.flush()
    payload.setdefault("status", status)
    payload.setdefault("exit_code", exit_code)
    payload["job_id"] = spec.get("job_id")
    _write_result(result_path, payload)
    return 0 if payload.get("status") == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
