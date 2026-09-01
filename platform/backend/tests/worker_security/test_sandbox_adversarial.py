from __future__ import annotations

import ast
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from app.core.worker_client import WorkerClient
from sandbox import build_child_env, try_upstream_runner

DAEMON_PATH = Path(__file__).resolve().parents[3] / "worker" / "daemon.py"
WORKER_ROOT = DAEMON_PATH.parent


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
    return Path(__file__).resolve().parents[4]


def test_31a_hook_imports_python_runner_run_job():
    runner = try_upstream_runner()
    assert runner is not None
    assert getattr(runner, "__name__", "") == "run_job"


def test_secrets_are_not_forwarded_to_child_env():
    env = build_child_env(
        Path("/tmp/job"),
        {
            "PATH": "/bin",
            "OPENAI_API_KEY": "sk-secret",
            "ANTHROPIC_API_KEY": "sk-ant",
            "LEARNINGOS_HOME": "/tmp/home",
        },
    )
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "LEARNINGOS_HOME" not in env


def test_worker_sources_never_import_sqlite3():
    for path in WORKER_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] != "sqlite3", path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] != "sqlite3", path


def test_path_escape_and_traversal_denied(worker_env, tmp_path):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    marker = tmp_path / "pwned"
    repo_marker = _repo_root() / f".wp400-pwn-{uuid.uuid4().hex}.txt"
    try:
        assert wait_until(client.health)
        for code in (
            f"open({str(marker)!r}, 'w').write('pwned')",
            "open('../pwned', 'w').write('pwned')",
            f"open({str(repo_marker)!r}, 'w').write('pwned')",
            "open('/etc/passwd').read()",
            f"__import__('os').system('touch {marker}')",
        ):
            result = client.execute(code, _limits())
            assert result.get("status") in {"DENIED", "FAILED"}
        assert not marker.exists()
        assert not repo_marker.exists()
        ping = client._rpc("ping")
        assert ping.get("alive") is True
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        repo_marker.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_symlink_escape_denied(worker_env, tmp_path):
    job_id = "sym-ci"
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
        assert result.get("status") in {"DENIED", "FAILED"}
        assert not (outside / "pwned").exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_timeout_status_and_worker_stays_alive(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        result = client.execute("x = 0\nwhile True:\n    x += 1\n", _limits(timeout_sec=1))
        assert result.get("status") == "TIMEOUT"
        assert client.health() is True
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_io_fileio_db_and_worktree_denied(worker_env):
    """Reviewer probe: import io; io.FileIO(db) and from io import FileIO."""
    db_path = worker_env["home"] / "learningos.db"
    db_path.write_bytes(b"untouched")
    repo_marker = _repo_root() / f".wp400-pwn-{uuid.uuid4().hex}.txt"
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        for code in (
            f"import io; io.FileIO({str(db_path)!r}, 'w').write(b'corrupt')",
            f"from io import FileIO; FileIO({str(db_path)!r}, 'w').write(b'corrupt')",
        ):
            hit = client.execute(code, _limits())
            assert hit.get("status") in {"DENIED", "FAILED"}, (code, hit)
            assert db_path.read_bytes() == b"untouched"
        for code in (
            f"import io; io.FileIO({str(repo_marker)!r}, 'w').write(b'pwned')",
            f"from io import FileIO; FileIO({str(repo_marker)!r}, 'w').write(b'pwned')",
        ):
            hit = client.execute(code, _limits())
            assert hit.get("status") in {"DENIED", "FAILED"}, (code, hit)
            assert not repo_marker.exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        repo_marker.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_io_open_db_and_worktree_denied(worker_env):
    """Reviewer probe: import io; io.open(LEARNINGOS_HOME/learningos.db) and repo writes."""
    db_path = worker_env["home"] / "learningos.db"
    db_path.write_bytes(b"untouched")
    repo_marker = _repo_root() / f".wp400-pwn-{uuid.uuid4().hex}.txt"
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        db_hit = client.execute(
            f"import io; io.open({str(db_path)!r}, 'w').write('corrupt')",
            _limits(),
        )
        assert db_hit.get("status") in {"DENIED", "FAILED"}, db_hit
        assert db_path.read_bytes() == b"untouched"
        tree_hit = client.execute(
            f"import io; io.open({str(repo_marker)!r}, 'w').write('pwned')",
            _limits(),
        )
        assert tree_hit.get("status") in {"DENIED", "FAILED"}, tree_hit
        assert not repo_marker.exists()
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        repo_marker.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_db_file_write_denied(worker_env):
    db_path = worker_env["home"] / "learningos.db"
    db_path.write_bytes(b"untouched")
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        result = client.execute(f"open({str(db_path)!r}, 'w').write('corrupt')", _limits())
        assert result.get("status") in {"DENIED", "FAILED"}
        assert db_path.read_bytes() == b"untouched"
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)


def test_success_snippet_and_echo_and_empty_unsupported(worker_env):
    proc = start_daemon(worker_env["env"])
    client = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(client.health)
        ok = client.execute("print('sandbox-ok')", _limits())
        assert ok.get("status") == "SUCCESS"
        assert "sandbox-ok" in (ok.get("stdout") or "")
        echoed = client.execute("", {"echo": "ping"})
        assert echoed.get("status") == "ACCEPTED"
        empty = client.execute("", {})
        assert empty.get("status") == "UNSUPPORTED"
        client.shutdown()
        proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)
