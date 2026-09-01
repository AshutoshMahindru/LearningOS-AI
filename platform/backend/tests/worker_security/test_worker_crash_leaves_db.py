from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from app.core.worker_client import WorkerClient
from app.db.database import get_connection

DAEMON_PATH = Path(__file__).resolve().parents[3] / "worker" / "daemon.py"


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_worker_crash_leaves_api_and_db_intact(data_home, client, auth_headers, worker_env):
    health = client.get("/api/v1/system/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "HEALTHY"

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            ("learner_wp400_31b", "wp40031b", "WP400 31B"),
        )
        conn.commit()
    finally:
        conn.close()

    db_path = data_home / "learningos.db"
    assert db_path.is_file()

    env = worker_env["env"]
    env["OPENAI_API_KEY"] = "sk-must-not-leak"
    proc = subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=env)
    worker = WorkerClient(worker_env["sock"])
    try:
        assert wait_until(worker.health), "worker did not become healthy"
        ran = worker.execute(
            "print('child-ok')",
            {"limits": {"timeout_sec": 5, "memory_mb": 256}},
        )
        assert ran.get("status") == "SUCCESS"
        denied = worker.execute(
            f"open({str(db_path)!r}, 'w').write('corrupt')",
            {"limits": {"timeout_sec": 5, "memory_mb": 256}},
        )
        assert denied.get("status") in {"DENIED", "FAILED"}
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=3)
        assert worker.health() is False

        after = client.get("/api/v1/system/health")
        assert after.status_code == 200, after.text
        assert after.json()["status"] == "HEALTHY"
        learners = client.get("/api/v1/learners/learner_wp400_31b", headers=auth_headers)
        assert learners.status_code == 200, learners.text
        assert learners.json()["username"] == "wp40031b"

        check = sqlite3.connect(str(db_path))
        try:
            assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            row = check.execute(
                "SELECT username FROM learners WHERE id = ?",
                ("learner_wp400_31b",),
            ).fetchone()
            assert row is not None
            assert row[0] == "wp40031b"
        finally:
            check.close()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
