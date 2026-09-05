from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from app.core.config import resolve_worker_socket
from app.core.worker_client import WorkerClient

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
API_EXEC_PATHS = (
    BACKEND_ROOT / "app" / "api" / "runtime.py",
    BACKEND_ROOT / "app" / "api" / "routes.py",
    BACKEND_ROOT / "app" / "core" / "worker_client.py",
    BACKEND_ROOT / "app" / "main.py",
)


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def start_daemon(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=env)


def _limits(timeout_sec: float = 5, memory_mb: int = 256) -> dict:
    return {"limits": {"timeout_sec": timeout_sec, "memory_mb": memory_mb}}


def test_worker_socket_is_not_in_the_git_worktree(isolated_home: Path) -> None:
    sock = resolve_worker_socket(isolated_home)
    assert sock == (isolated_home / "run" / "worker.sock").resolve()
    assert isolated_home.is_relative_to(Path("/tmp"))
    assert not sock.is_relative_to(REPO_ROOT)
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    tracked_text = [os.fsdecode(item) for item in tracked if item]
    assert not any(path.endswith("worker.sock") or path.endswith(".sock") for path in tracked_text)
    assert not (REPO_ROOT / "platform" / "worker" / "worker.sock").exists()
    assert not (REPO_ROOT / "worker.sock").exists()


def test_api_execute_does_not_exec_in_the_api_process() -> None:
    banned = {"exec", "eval"}
    for path in API_EXEC_PATHS:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in banned, path
        if path.name == "runtime.py":
            assert "WorkerClient" in source
            assert "python_runner" not in source
            assert "run_source" not in source
            assert "_call_worker_execute" in source
    runner = (BACKEND_ROOT / "app" / "execution" / "python_runner.py").read_text(encoding="utf-8")
    assert "IN_PROCESS_LIBRARY_ONLY = True" in runner


def test_timeout_or_crash_leaves_daemon_alive(worker_env: dict) -> None:
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health), "daemon did not become healthy"
        result = client.execute("x = 0\nwhile True:\n    x += 1\n", _limits(timeout_sec=1))
        # CI has flaked TIMEOUT vs CRASHED; both prove the job ended and the daemon lived.
        assert result.get("status") in {"TIMEOUT", "CRASHED"}, result
        assert client.health() is True
        follow_up = client.execute("print('still-alive')", _limits())
        assert follow_up.get("status") == "SUCCESS", follow_up
        assert "still-alive" in (follow_up.get("stdout") or "")
        client.shutdown()
        assert wait_until(lambda: proc.poll() is not None)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_io_denials_still_hold(worker_env: dict, isolated_home: Path) -> None:
    db_path = isolated_home / "learningos.db"
    db_path.write_bytes(b"untouched")
    marker = isolated_home.parent / "pwned"
    repo_marker = REPO_ROOT / f".g7-sec-pwn-{uuid.uuid4().hex}.txt"
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health), "daemon did not become healthy"
        probes = (
            f"open({str(db_path)!r}, 'w').write('corrupt')",
            f"import io; io.open({str(db_path)!r}, 'w').write('corrupt')",
            f"import io; io.FileIO({str(db_path)!r}, 'w').write(b'corrupt')",
            f"from io import FileIO; FileIO({str(db_path)!r}, 'w').write(b'corrupt')",
            f"open({str(marker)!r}, 'w').write('pwned')",
            f"import io; io.open({str(repo_marker)!r}, 'w').write('pwned')",
            f"import io; io.FileIO({str(repo_marker)!r}, 'w').write(b'pwned')",
            "open('/etc/passwd').read()",
        )
        for code in probes:
            hit = client.execute(code, _limits())
            assert hit.get("status") in {"DENIED", "FAILED"}, (code, hit)
        assert db_path.read_bytes() == b"untouched"
        assert not marker.exists()
        assert not repo_marker.exists()
        ping = client._rpc("ping")
        assert ping.get("alive") is True
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        repo_marker.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)
