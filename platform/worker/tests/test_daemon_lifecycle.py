from __future__ import annotations

import ast
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

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


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    # macOS AF_UNIX sun_path is ~104 bytes; pytest tmp paths are too long.
    sock = Path(f"/tmp/los-g3-{uuid.uuid4().hex}.sock")
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        yield {"home": home, "sock": sock, "env": env}
    finally:
        sock.unlink(missing_ok=True)


def test_health_then_shutdown(worker_env):
    proc = start_daemon(worker_env["env"])
    try:
        client = WorkerClient(worker_env["sock"])
        assert wait_until(client.health), "daemon did not become healthy"
        payload = client.execute("print('hello')", {})
        assert payload.get("status") == "UNSUPPORTED"
        assert payload.get("reason")
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
            result = client.execute(code, {})
            assert result.get("status") == "UNSUPPORTED"
            assert "error" not in result or result.get("error", {}).get("code") != "WORKER_UNAVAILABLE"
        assert not marker.exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_daemon_source_has_no_exec_eval_or_sqlite():
    source = DAEMON_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"exec", "eval", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] != "sqlite3"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] != "sqlite3"


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
