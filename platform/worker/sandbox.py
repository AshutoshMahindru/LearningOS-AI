"""WP400 worker sandbox: learner code runs only in an isolated subprocess.

Path isolation
    Jobs execute with cwd $LEARNINGOS_HOME/run/jobs/<job_id> (or a temp dir).
    Writes to the Git worktree and to $LEARNINGOS_HOME/*.db are refused.
    Relative ".." segments and symlink escape from the job dir are rejected.

Resource limits
    Wall-clock timeout_sec (from execute_task limits), RLIMIT_CPU, RLIMIT_AS,
    RLIMIT_FSIZE, and a 1 MiB captured-output cap. Timeout and cancel kill the
    entire child process group via killpg.

Secret boundary
    The child environment is constructed from an allowlist. OPENAI_API_KEY and
    other provider/secret variables are never forwarded.

Network policy (best-effort deny)
    The child does not inherit proxy settings. Restricted builtins block
    socket/urllib/http/ssl/requests. When the OS allows it, the parent wraps
    the child in sandbox-exec (Darwin) or `unshare --net` (Linux). This is not
    a full MAC/seccomp sandbox.

31A hook
    Prefer `from app.execution.python_runner import run_job` when that module
    exists (invoked inside the isolated child). On ImportError the local
    restricted-python snippet runner is used.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

CHILD_PATH = Path(__file__).resolve().parent / "restricted_child.py"
MAX_OUTPUT_BYTES = 1_048_576
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_MEMORY_MB = 2048
MAX_TIMEOUT_SEC = 600.0
MAX_MEMORY_MB = 8192
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SECRET_ENV_RE = re.compile(
    r"(API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|AUTHORIZATION)",
    re.IGNORECASE,
)
SECRET_ENV_EXACT = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_ORG",
        "OPENAI_ORGANIZATION",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)
ENV_PASSTHROUGH = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "TERM",
        "USER",
        "LOGNAME",
        "TMPDIR",
    }
)
BANNED_AST_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "ssl",
        "http",
        "urllib",
        "requests",
        "ctypes",
        "pathlib",
        "shutil",
        "signal",
        "resource",
        "importlib",
        "multiprocessing",
        "threading",
        "pickle",
        "sqlite3",
        "builtins",
        "code",
        "pty",
        "fcntl",
        "mmap",
    }
)

_net_prefix_cache: list[str] | None = None


class SandboxViolation(Exception):
    """Host-side sandbox policy rejection (before spawn)."""


def detect_repo_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        has_platform = (candidate / "platform" / "backend" / "app").is_dir()
        has_git = (candidate / ".git").exists() or (candidate / ".git").is_file()
        has_arch = (candidate / "architecture" / "learningos-v3").is_dir()
        if has_platform and (has_git or has_arch):
            return candidate
    return None


def detect_backend_root() -> Path | None:
    backend = Path(__file__).resolve().parents[1] / "backend"
    if (backend / "app").is_dir():
        return backend
    return None


def sanitize_job_id(raw: str | None) -> str:
    text = (raw or "").strip()
    if text and JOB_ID_RE.fullmatch(text):
        return text
    if text:
        raise SandboxViolation("invalid job_id")
    return uuid.uuid4().hex


def coerce_timeout_sec(value: Any, default: float = DEFAULT_TIMEOUT_SEC) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default
    if timeout <= 0:
        return default
    return min(timeout, MAX_TIMEOUT_SEC)


def coerce_memory_mb(value: Any, default: int = DEFAULT_MEMORY_MB) -> int:
    try:
        memory = int(value)
    except (TypeError, ValueError):
        return default
    if memory <= 0:
        return default
    return max(16, min(memory, MAX_MEMORY_MB))


def data_home_from_env(raw: str | None = None) -> Path | None:
    text = raw if raw is not None else os.environ.get("LEARNINGOS_HOME")
    if not text:
        return None
    return Path(text).expanduser().resolve()


def prepare_workdir(job_id: str, data_home: Path | None) -> Path:
    job_id = sanitize_job_id(job_id)
    if data_home is not None:
        jobs_root = (data_home / "run" / "jobs").resolve()
        jobs_root.mkdir(parents=True, exist_ok=True)
        if jobs_root.is_symlink():
            raise SandboxViolation("jobs root must not be a symlink")
        workdir = (jobs_root / job_id).resolve()
        if not workdir.is_relative_to(jobs_root):
            raise SandboxViolation("job workdir escapes jobs root")
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix=f"los-job-{job_id}-")).resolve()
    if workdir.is_symlink():
        raise SandboxViolation("job workdir must not be a symlink")
    os.chmod(workdir, 0o700)
    sandbox_dir = workdir / ".sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "tmp").mkdir(exist_ok=True)
    return workdir


_IO_FILE_PRIMITIVES = frozenset({"open", "FileIO"})


def _is_io_file_primitive_call(node: ast.Call) -> bool:
    """True for io.open / io.FileIO / _io.* — filesystem write primitives."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _IO_FILE_PRIMITIVES:
        base = func.value
        if isinstance(base, ast.Name) and base.id in {"io", "_io"}:
            return True
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.attr in {"io", "_io"}
        ):
            return True
    return False


def ast_preflight(code: str) -> None:
    try:
        tree = ast.parse(code, filename="<learner>")
    except SyntaxError as exc:
        raise SyntaxError(str(exc)) from exc
    banned_names = {"eval", "exec", "compile", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_AST_MODULES:
                    raise SandboxViolation(f"import of {alias.name!r} is blocked")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise SandboxViolation("relative imports are not allowed")
            root = (node.module or "").split(".", 1)[0]
            if root in BANNED_AST_MODULES:
                raise SandboxViolation(f"import of {node.module!r} is blocked")
            # from io import open rebinds a filesystem primitive; wrap at runtime
            # still applies, but reject the unguarded alias so 31A cannot skip io.open.
            if root in {"io", "_io"}:
                for alias in node.names:
                    if alias.name in _IO_FILE_PRIMITIVES:
                        raise SandboxViolation(
                            f"from io import {alias.name} is not allowed; use open() inside the job workdir"
                        )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in banned_names:
                raise SandboxViolation(f"{node.func.id} is not allowed in the sandbox")
        elif isinstance(node, ast.Call) and _is_io_file_primitive_call(node):
            # io.open / io.FileIO are filesystem write primitives. Runtime wrapping
            # on every exec path (including 31A) enforces workdir / db / repo policy.
            continue


def is_secret_env_name(name: str) -> bool:
    if name in SECRET_ENV_EXACT:
        return True
    if SECRET_ENV_RE.search(name):
        return True
    return False


def build_child_env(workdir: Path, source: dict[str, str] | None = None) -> dict[str, str]:
    """Allowlisted child environment. Never forwards provider secrets."""
    src = source if source is not None else os.environ
    env: dict[str, str] = {
        "HOME": str(workdir),
        "TMPDIR": str(workdir / "tmp"),
        "TEMP": str(workdir / "tmp"),
        "TMP": str(workdir / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "LEARNINGOS_SANDBOX": "1",
    }
    for key in ENV_PASSTHROUGH:
        value = src.get(key)
        if not value or is_secret_env_name(key):
            continue
        env[key] = value
    if "PATH" not in env:
        env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
    for proxy in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
        "ftp_proxy",
        "FTP_PROXY",
    ):
        env.pop(proxy, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def kill_process_group(proc: subprocess.Popen[bytes] | None, pgid: int | None = None) -> None:
    if proc is None and pgid is None:
        return
    target = pgid if pgid is not None else (proc.pid if proc is not None else None)
    if target is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(target, sig)
        except ProcessLookupError:
            break
        except OSError:
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            break
        if sig == signal.SIGTERM:
            time.sleep(0.05)
    if proc is not None:
        try:
            proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass


def _probe_command(prefix: list[str]) -> bool:
    try:
        completed = subprocess.run(
            [*prefix, sys.executable, "-I", "-c", "pass"],
            check=False,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def network_deny_prefix() -> list[str]:
    """Best-effort OS wrappers that prevent new sockets. Cached per process."""
    global _net_prefix_cache
    if _net_prefix_cache is not None:
        return list(_net_prefix_cache)
    prefix: list[str] = []
    if sys.platform == "darwin":
        exe = shutil.which("sandbox-exec")
        if exe:
            profile = "(version 1)(allow default)(deny network*)"
            candidate = [exe, "-p", profile]
            if _probe_command(candidate):
                prefix = candidate
    elif sys.platform.startswith("linux"):
        exe = shutil.which("unshare")
        if exe:
            for candidate in ([exe, "--net", "--"], [exe, "-n", "--"]):
                if _probe_command(candidate):
                    prefix = candidate
                    break
    _net_prefix_cache = prefix
    return list(prefix)


def _read_text_capped(path: Path, limit: int = MAX_OUTPUT_BYTES) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > limit:
        data = data[:limit]
    return data.decode("utf-8", errors="replace")


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def try_upstream_runner() -> Callable[..., Any] | None:
    """31A hook: app.execution.python_runner.run_job, else None.

    The isolated child also performs this import. Resolving it here lets tests
    observe the fallback without executing learner code in the daemon process.
    """
    backend = detect_backend_root()
    if backend and str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    try:
        from app.execution.python_runner import run_job as upstream
    except ImportError:
        return None
    return upstream


def run_job(
    job: dict[str, Any],
    *,
    cancel_event: threading.Event | None = None,
    pgid_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run learner code in an isolated subprocess (31A-compatible).

    job keys: job_id, code, parameters, timeout_sec, memory_mb, data_home,
    workdir (optional), limits (optional).
    """
    started = time.monotonic()
    job_id = sanitize_job_id(str(job.get("job_id") or "") or None)
    code = job.get("code") if isinstance(job.get("code"), str) else ""
    parameters = job.get("parameters") if isinstance(job.get("parameters"), dict) else {}
    limits = job.get("limits") if isinstance(job.get("limits"), dict) else {}
    timeout_sec = coerce_timeout_sec(
        job.get("timeout_sec", limits.get("timeout_sec", DEFAULT_TIMEOUT_SEC))
    )
    memory_mb = coerce_memory_mb(job.get("memory_mb", limits.get("memory_mb", DEFAULT_MEMORY_MB)))
    home = data_home_from_env(job.get("data_home"))
    workdir = Path(job["workdir"]).resolve() if job.get("workdir") else prepare_workdir(job_id, home)
    sandbox_dir = workdir / ".sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    spec_path = sandbox_dir / "spec.json"
    result_path = sandbox_dir / "result.json"
    stdout_path = sandbox_dir / "stdout.log"
    stderr_path = sandbox_dir / "stderr.log"

    def _finish(payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("job_id", job_id)
        payload.setdefault("workdir", str(workdir))
        payload.setdefault("duration_ms", int((time.monotonic() - started) * 1000))
        payload.setdefault("stdout", _read_text_capped(stdout_path))
        payload.setdefault("stderr", _read_text_capped(stderr_path))
        if "exit_code" not in payload:
            payload["exit_code"] = 0 if payload.get("status") == "SUCCESS" else 1
        return payload

    try:
        ast_preflight(code)
    except SyntaxError as exc:
        return _finish({"status": "FAILED", "reason": f"syntax error: {exc}", "exit_code": 1})
    except SandboxViolation as exc:
        return _finish({"status": "DENIED", "reason": str(exc), "exit_code": 1})

    spec = {
        "job_id": job_id,
        "code": code,
        "parameters": parameters,
        "workdir": str(workdir),
        "repo_root": str(job.get("repo_root") or detect_repo_root() or ""),
        "data_home": str(home) if home is not None else "",
        "backend_root": str(detect_backend_root() or ""),
        "timeout_sec": timeout_sec,
        "memory_mb": memory_mb,
        "cpu_sec": max(1, int(timeout_sec + 0.999)),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    argv = [
        *network_deny_prefix(),
        sys.executable,
        "-I",
        "-B",
        str(CHILD_PATH),
        str(spec_path),
    ]
    env = build_child_env(workdir)
    proc: subprocess.Popen[bytes] | None = None
    timed_out = False
    cancelled = False
    output_exceeded = False
    proc = subprocess.Popen(
        argv,
        cwd=str(workdir),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pgid = proc.pid
    if pgid_callback is not None:
        try:
            pgid_callback(pgid)
        except Exception:
            pass
    deadline = time.monotonic() + timeout_sec
    while True:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            kill_process_group(proc, pgid)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            kill_process_group(proc, pgid)
            break
        try:
            if stdout_path.stat().st_size > MAX_OUTPUT_BYTES or stderr_path.stat().st_size > MAX_OUTPUT_BYTES:
                output_exceeded = True
                kill_process_group(proc, pgid)
                break
        except OSError:
            pass
        remaining = deadline - time.monotonic()
        try:
            proc.wait(timeout=min(0.05, max(0.01, remaining)))
            break
        except subprocess.TimeoutExpired:
            continue

    if proc is not None and proc.poll() is None:
        kill_process_group(proc, proc.pid)

    returncode = proc.returncode if proc is not None else -1
    child_result = _load_result(result_path)

    if cancelled:
        return _finish({"status": "CANCELLED", "exit_code": -1, "reason": "cancelled"})
    if timed_out:
        return _finish({"status": "TIMEOUT", "exit_code": -1, "reason": "timeout_sec exceeded"})
    if output_exceeded:
        return _finish({"status": "FAILED", "exit_code": 1, "reason": "output limit exceeded"})
    if returncode is not None and returncode < 0:
        sig = -returncode
        if sig in {signal.SIGXCPU, getattr(signal, "SIGXFSZ", 0)}:
            return _finish({"status": "TIMEOUT", "exit_code": returncode, "reason": f"signal {sig}"})
        if child_result:
            return _finish(child_result)
        return _finish({"status": "CRASHED", "exit_code": returncode, "reason": f"signal {sig}"})
    if child_result:
        return _finish(child_result)
    if returncode == 0:
        return _finish({"status": "SUCCESS", "exit_code": 0})
    return _finish({"status": "FAILED", "exit_code": returncode if returncode is not None else 1})
