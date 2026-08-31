from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from app.core.worker_client import WorkerClient, resolve_worker_socket

DAEMON_PATH = Path(__file__).resolve().parents[3] / "worker" / "daemon.py"


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
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


def test_resolve_prefers_explicit_then_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    monkeypatch.delenv("LEARNINGOS_HOME", raising=False)
    assert resolve_worker_socket() == Path("/tmp/learningos_worker.sock")
    home = tmp_path / "home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    assert resolve_worker_socket() == home.resolve() / "run" / "worker.sock"
    sock = tmp_path / "custom.sock"
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    assert resolve_worker_socket() == sock


def test_missing_socket_is_structured_error_not_exception(tmp_path, monkeypatch):
    sock = tmp_path / "missing.sock"
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    client = WorkerClient(sock)
    assert client.health() is False
    result = client.execute("open('/etc/passwd')", {"x": 1})
    assert result["error"]["code"] == "WORKER_UNAVAILABLE"
    cancelled = client.cancel("job_missing")
    assert cancelled["error"]["code"] == "WORKER_UNAVAILABLE"
    stopped = client.shutdown()
    assert stopped["error"]["code"] == "WORKER_UNAVAILABLE"


def test_start_health_execute_boundary_and_shutdown(worker_env, tmp_path):
    proc = subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=worker_env["env"])
    client = WorkerClient()
    marker = tmp_path / "pwned"
    try:
        assert wait_until(client.health)
        assert client.health() is True
        attacked = client.execute(
            f"open({str(marker)!r}, 'w').write('pwned'); __import__('os').system('touch {marker}')",
            {},
        )
        assert attacked.get("status") == "UNSUPPORTED"
        assert attacked.get("job_id")
        assert not marker.exists()
        ping = client._rpc("ping")
        assert ping.get("alive") is True
        assert isinstance(ping.get("pid"), int)
        client.shutdown()
        assert wait_until(lambda: proc.poll() is not None)
        assert proc.returncode == 0
        assert client.health() is False
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_kill9_then_restart_same_socket(worker_env):
    proc = subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=3)
        assert client.health() is False
        assert client.execute("print(1)", {})["error"]["code"] == "WORKER_UNAVAILABLE"
        proc = subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=worker_env["env"])
        client = WorkerClient(worker_env["sock"])
        assert wait_until(client.health)
        assert client.health() is True
        client.shutdown()
        proc.wait(timeout=3)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
