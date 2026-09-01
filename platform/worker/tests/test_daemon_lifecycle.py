from __future__ import annotations

import ast
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from app.core.worker_client import WorkerClient

DAEMON_PATH = Path(__file__).resolve().parents[1] / "daemon.py"
WATCHDOG_PATH = Path(__file__).resolve().parents[1] / "watchdog.py"


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def start_daemon(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=env)


def test_health_then_shutdown(worker_env):
    proc = start_daemon(worker_env["env"])
    try:
        client = WorkerClient(worker_env["sock"])
        assert wait_until(client.health), "daemon did not become healthy"
        payload = client.execute("print('hello')", {"limits": {"timeout_sec": 5, "memory_mb": 256}})
        assert payload.get("status") == "SUCCESS"
        assert "hello" in (payload.get("stdout") or "")
        echoed = client.execute("", {"echo": "ping"})
        assert echoed.get("status") == "ACCEPTED"
        assert echoed.get("echo") == "ping"
        cancelled = client.cancel(payload["job_id"])
        assert cancelled.get("status") in {"CANCELLED", "NOT_FOUND"}
        client.shutdown()
        assert wait_until(lambda: proc.poll() is not None)
        assert proc.returncode == 0
        assert client.health() is False
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_kill9_unavailable_then_restart_same_socket(worker_env):
    sock = worker_env["sock"]
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(sock)
    try:
        assert wait_until(client.health)
        proc.kill()
        proc.wait(timeout=3)
        assert client.health() is False
        unavailable = client.execute("open('/etc/passwd')", {})
        assert unavailable["error"]["code"] == "WORKER_UNAVAILABLE"
        proc = start_daemon(worker_env["env"])
        client = WorkerClient(sock)
        assert wait_until(client.health), "restart on the same socket path failed"
        assert client.health() is True
        client.shutdown()
        assert wait_until(lambda: proc.poll() is not None)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_bare_connect_keeps_serving_health(worker_env):
    """11A smoke connects to the socket without sending a JSON-RPC body."""
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(1)
        probe.connect(str(worker_env["sock"]))
        probe.close()
        assert client.health() is True
        client.shutdown()
        assert wait_until(lambda: proc.poll() is not None)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_sigterm_unlinks_socket_and_exits_zero(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        proc.send_signal(signal.SIGTERM)
        assert wait_until(lambda: proc.poll() is not None)
        assert proc.returncode == 0
        assert not worker_env["sock"].exists()
        assert client.health() is False
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def test_execute_task_does_not_run_attacker_payload(worker_env, tmp_path):
    marker = tmp_path / "pwned"
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        payloads = [
            f"open({str(marker)!r}, 'w').write('pwned')",
            "open('/etc/passwd').read()",
            f"__import__('os').system('touch {marker}')",
        ]
        for code in payloads:
            result = client.execute(code, {"limits": {"timeout_sec": 5, "memory_mb": 256}})
            assert result.get("status") in {"DENIED", "FAILED"}
            assert "error" not in result or result.get("error", {}).get("code") != "WORKER_UNAVAILABLE"
        assert not marker.exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def _assert_no_sqlite(tree: ast.AST, path: Path) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] != "sqlite3", path
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] != "sqlite3", path


def test_daemon_source_has_no_exec_eval_or_sqlite():
    source = DAEMON_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"exec", "eval", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned
    _assert_no_sqlite(tree, DAEMON_PATH)


def test_worker_package_does_not_import_sqlite3():
    worker_root = DAEMON_PATH.parent
    for path in worker_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _assert_no_sqlite(tree, path)


def test_watchdog_launches_daemon(worker_env):
    proc = subprocess.Popen([sys.executable, str(WATCHDOG_PATH)], env=worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health, timeout=6.0)
        proc.send_signal(signal.SIGTERM)
        assert wait_until(lambda: proc.poll() is not None, timeout=6.0)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
