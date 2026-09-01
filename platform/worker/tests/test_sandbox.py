from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from app.core.worker_client import WorkerClient
from sandbox import (
    build_child_env,
    is_secret_env_name,
    run_job,
    try_upstream_runner,
)

DAEMON_PATH = Path(__file__).resolve().parents[1] / "daemon.py"


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_31a_hook_falls_back_to_local_runner():
    assert try_upstream_runner() is None
    result = run_job(
        {
            "job_id": "hook-fallback",
            "code": "result = 1 + 1\nprint(result)",
            "timeout_sec": 5,
            "memory_mb": 256,
        }
    )
    assert result["status"] == "SUCCESS"
    assert "2" in (result.get("stdout") or "")


def test_child_env_strips_provider_secrets():
    source = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "OPENAI_API_KEY": "sk-secret",
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "GITHUB_TOKEN": "ghs_secret",
        "MY_API_KEY": "nope",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "http_proxy": "http://127.0.0.1:8080",
        "LEARNINGOS_HOME": "/tmp/should-not-pass",
    }
    workdir = Path("/tmp/los-sandbox-env-test")
    env = build_child_env(workdir, source)
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "MY_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "LEARNINGOS_HOME" not in env
    assert env.get("http_proxy") is None
    assert env["HOME"] == str(workdir)
    assert is_secret_env_name("OPENAI_API_KEY")


def test_success_write_stays_in_job_dir(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    job_id = "write-ok"
    try:
        assert wait_until(client.health)
        result = client._rpc(
            "execute_task",
            {
                "job_id": job_id,
                "code": "open('out.txt','w').write('ok')\nprint('wrote')",
                "limits": {"timeout_sec": 5, "memory_mb": 256},
            },
            timeout=10,
        )
        assert result.get("status") == "SUCCESS"
        workdir = worker_env["home"] / "run" / "jobs" / job_id
        assert (workdir / "out.txt").read_text(encoding="utf-8") == "ok"
        assert "wrote" in (result.get("stdout") or "")
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_path_escape_to_repo_and_tmp_is_denied(worker_env, tmp_path):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    marker = tmp_path / "pwned"
    repo_marker = _repo_root() / f".wp400-pwn-{uuid.uuid4().hex}.txt"
    try:
        assert wait_until(client.health)
        payloads = [
            f"open({str(marker)!r}, 'w').write('pwned')",
            "open('../pwned', 'w').write('pwned')",
            "open('../../etc/passwd').read()",
            f"open({str(repo_marker)!r}, 'w').write('pwned')",
            "open('/etc/passwd').read()",
        ]
        for code in payloads:
            result = client.execute(code, _limits())
            assert result.get("status") in {"DENIED", "FAILED"}, result
        assert not marker.exists()
        assert not repo_marker.exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        repo_marker.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_refuses_learningos_home_db_write(worker_env):
    home = worker_env["home"]
    db_path = home / "learningos.db"
    db_path.write_bytes(b"safe-bytes")
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        result = client.execute(
            f"open({str(db_path)!r}, 'w').write('corrupt')",
            _limits(),
        )
        assert result.get("status") in {"DENIED", "FAILED"}
        assert db_path.read_bytes() == b"safe-bytes"
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_symlink_escape_is_denied(worker_env, tmp_path):
    job_id = "sym-escape"
    workdir = worker_env["home"] / "run" / "jobs" / job_id
    workdir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workdir / "escape").symlink_to(outside)
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        result = client._rpc(
            "execute_task",
            {
                "job_id": job_id,
                "code": "open('escape/pwned','w').write('nope')",
                "limits": {"timeout_sec": 5, "memory_mb": 256},
            },
            timeout=10,
        )
        assert result.get("status") in {"DENIED", "FAILED"}, result
        assert not (outside / "pwned").exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_timeout_kills_busy_loop(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        started = time.monotonic()
        result = client.execute("x = 0\nwhile True:\n    x += 1\n", _limits(timeout_sec=1))
        elapsed = time.monotonic() - started
        assert result.get("status") == "TIMEOUT"
        assert elapsed < 8
        assert client.health() is True
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_urllib_and_socket_imports_are_blocked(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        for code in (
            "import urllib.request\nurllib.request.urlopen('http://example.com')",
            "import socket\nsocket.socket()",
            "import os",
            "import sqlite3",
        ):
            result = client.execute(code, _limits())
            assert result.get("status") in {"DENIED", "FAILED"}, result
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_cancel_kills_process_group(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    job_id = "cancel-loop"
    holder: dict[str, dict] = {}

    def _run() -> None:
        holder["result"] = client._rpc(
            "execute_task",
            {
                "job_id": job_id,
                "code": "x = 0\nwhile True:\n    x += 1\n",
                "limits": {"timeout_sec": 20, "memory_mb": 256},
            },
            timeout=25,
        )

    try:
        assert wait_until(client.health)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        cancelled = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cancelled = client.cancel(job_id)
            if cancelled.get("status") == "CANCELLED":
                break
            time.sleep(0.05)
        assert cancelled is not None
        assert cancelled.get("status") == "CANCELLED"
        thread.join(timeout=8)
        assert holder.get("result", {}).get("status") in {"CANCELLED", "TIMEOUT", "CRASHED"}
        assert client.health() is True
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_health_during_running_job(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    job_id = "health-during"
    holder: dict[str, dict] = {}

    def _run() -> None:
        holder["result"] = client._rpc(
            "execute_task",
            {
                "job_id": job_id,
                "code": "x = 0\nwhile True:\n    x += 1\n",
                "limits": {"timeout_sec": 8, "memory_mb": 256},
            },
            timeout=15,
        )

    try:
        assert wait_until(client.health)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        assert wait_until(lambda: client.health() is True, timeout=3)
        ping = client._rpc("ping")
        assert ping.get("alive") is True
        client.cancel(job_id)
        thread.join(timeout=8)
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_worker_kill_leaves_home_db_intact(worker_env):
    db_path = worker_env["home"] / "learningos.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE intact (id INTEGER PRIMARY KEY, note TEXT)")
        conn.execute("INSERT INTO intact (note) VALUES ('before-crash')")
        conn.commit()
    finally:
        conn.close()
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        client.execute("print('before-kill')", _limits())
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=3)
        assert client.health() is False
        check = sqlite3.connect(str(db_path))
        try:
            assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            row = check.execute("SELECT note FROM intact").fetchone()
            assert row[0] == "before-crash"
        finally:
            check.close()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
