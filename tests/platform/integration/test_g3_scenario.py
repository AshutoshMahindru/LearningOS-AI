"""G3 platform scenario: persist, restart, worker recovery, backup/restore."""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
FIXTURE_PACKAGE = REPO_ROOT / "platform" / "worker" / "fixtures" / "g3_curriculum"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _start_worker(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(DAEMON_PATH)],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_worker(proc: subprocess.Popen[bytes] | None, sig: signal.Signals = signal.SIGTERM) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(sig)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _outside_repo(path: Path) -> None:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    with pytest.raises(ValueError):
        resolved.relative_to(repo)


@pytest.fixture
def g3_env(monkeypatch):
    home_a = Path(tempfile.mkdtemp(prefix="los-g3-home-a-", dir="/tmp"))
    home_b = Path(tempfile.mkdtemp(prefix="los-g3-home-b-", dir="/tmp"))
    sock = Path(f"/tmp/los-g3-int-{uuid.uuid4().hex}.sock")
    sock.unlink(missing_ok=True)
    monkeypatch.setenv("LEARNINGOS_HOME", str(home_a))
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-never-leak")
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(home_a)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        yield {"home_a": home_a, "home_b": home_b, "sock": sock, "env": env}
    finally:
        sock.unlink(missing_ok=True)
        shutil.rmtree(home_a, ignore_errors=True)
        shutil.rmtree(home_b, ignore_errors=True)


def test_g3_platform_scenario(g3_env, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    home_a: Path = g3_env["home_a"]
    home_b: Path = g3_env["home_b"]
    sock: Path = g3_env["sock"]
    env: dict[str, str] = g3_env["env"]
    blob = b"g3-artifact-bytes-roundtrip"
    worker: subprocess.Popen[bytes] | None = None

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            health = client.get("/api/v1/system/health")
            assert health.status_code == 200, health.text
            body = health.json()
            assert body["status"] == "HEALTHY"
            assert body["version"] == "3.0.0"
            assert body["worker_alive"] is True
            assert body["database_path"] == str((home_a / "learningos.db").resolve())
            assert Path(body["database_path"]).is_file()

            assert home_a.is_dir()
            _outside_repo(home_a)

            boot = client.post("/api/v1/auth/bootstrap")
            assert boot.status_code == 200, boot.text
            token = boot.json()["token"]
            assert boot.json()["token_type"] == "bearer"
            headers = {"Authorization": f"Bearer {token}"}

            loaded = client.post(
                "/api/v1/curriculum/packages/load",
                json={"package_dir": str(FIXTURE_PACKAGE)},
                headers=headers,
            )
            assert loaded.status_code == 200, loaded.text
            loaded_body = loaded.json()
            assert loaded_body.get("id") == "g3.fixture.curriculum"

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            mission_ids = [item.get("id") for item in missions.json().get("missions", [])]
            assert "g3.fixture.orientation" in mission_ids

            learner = client.post(
                "/api/v1/learners",
                json={"username": "g3-learner", "display_name": "G3 Learner"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]
            assert learner_id

            session = client.post(
                "/api/v1/sessions",
                json={"mission_id": "g3.fixture.orientation", "learner_id": learner_id},
                headers=headers,
            )
            assert session.status_code == 200, session.text
            assert session.json()["mission_id"] == "g3.fixture.orientation"
            assert session.json()["learner_id"] == learner_id

            created = client.post(
                "/api/v1/artifacts",
                json={"bytes_b64": base64.b64encode(blob).decode("ascii")},
                headers=headers,
            )
            assert created.status_code == 200, created.text
            artifact_hash = created.json()["artifact_hash"]
            assert len(artifact_hash) == 64
            fetched = client.get(f"/api/v1/artifacts/{artifact_hash}", headers=headers)
            assert fetched.status_code == 200, fetched.text
            assert fetched.content == blob

            config = client.get("/api/v1/system/config")
            assert config.status_code == 200, config.text
            config_text = json.dumps(config.json()).lower()
            assert "api_key" not in config_text
            assert "openai" not in config_text
            assert "sk-test" not in config_text
            assert "OPENAI" not in json.dumps(config.json())

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            boot = client.post("/api/v1/auth/bootstrap")
            assert boot.status_code == 200, boot.text
            headers = {"Authorization": f"Bearer {boot.json()['token']}"}
            got = client.get(f"/api/v1/learners/{learner_id}", headers=headers)
            assert got.status_code == 200, got.text
            fetched = client.get(f"/api/v1/artifacts/{artifact_hash}", headers=headers)
            assert fetched.status_code == 200, fetched.text
            assert fetched.content == blob

            worker.send_signal(signal.SIGKILL)
            worker.wait(timeout=3)
            assert worker.poll() is not None
            assert _wait_until(
                lambda: client.get("/api/v1/system/health").json().get("worker_alive") is False
            ), "health still reports worker_alive after SIGKILL"
            health = client.get("/api/v1/system/health")
            assert health.status_code == 200
            assert health.json()["status"] == "HEALTHY"
            assert health.json()["worker_alive"] is False
            got = client.get(f"/api/v1/learners/{learner_id}", headers=headers)
            assert got.status_code == 200, got.text

            worker = _start_worker(env)
            assert _wait_until(
                lambda: client.get("/api/v1/system/health").json().get("worker_alive") is True
            ), "worker did not recover on the same socket"
            health = client.get("/api/v1/system/health")
            assert health.status_code == 200
            assert health.json()["worker_alive"] is True

            backup = client.post("/api/v1/system/backup", headers=headers)
            assert backup.status_code == 200, backup.text
            backup_path = Path(backup.json()["path"])
            assert backup_path.is_file()

            tutor = client.post(
                "/api/v1/tutor/chat",
                json={
                    "session_id": "not-executed",
                    "stage_id": "orientation",
                    "role": "learner",
                    "prompt": "hello",
                },
                headers=headers,
            )
            assert tutor.status_code == 501, tutor.text
            assert tutor.json()["error"]["code"] == "TUTOR_NOT_AVAILABLE"
            assert "openai" not in sys.modules

            restore = client.post(
                "/api/v1/system/restore",
                json={"path": str(backup_path), "dest_home": str(home_b)},
                headers=headers,
            )
            assert restore.status_code == 200, restore.text
            assert (home_b / "learningos.db").is_file()

        monkeypatch.setenv("LEARNINGOS_HOME", str(home_b))
        with TestClient(app, client=("127.0.0.1", 50000)) as client_b:
            boot = client_b.post("/api/v1/auth/bootstrap")
            assert boot.status_code == 200, boot.text
            headers_b = {"Authorization": f"Bearer {boot.json()['token']}"}
            got = client_b.get(f"/api/v1/learners/{learner_id}", headers=headers_b)
            assert got.status_code == 200, got.text
            assert got.json().get("username") == "g3-learner"
            fetched = client_b.get(f"/api/v1/artifacts/{artifact_hash}", headers=headers_b)
            assert fetched.status_code == 200, fetched.text
            assert fetched.content == blob

            cfg = client_b.get("/api/v1/system/config")
            dumped = json.dumps(cfg.json())
            assert "api_key" not in dumped.lower()
            assert "openai" not in dumped.lower()
            assert "OPENAI" not in dumped
            for config_file in (home_a / "config.json", home_b / "config.json"):
                if config_file.is_file():
                    text = config_file.read_text(encoding="utf-8")
                    assert "api_key" not in text.lower()
                    assert "openai" not in text.lower()
                    assert "OPENAI" not in text

        completed = subprocess.run(
            [sys.executable, str(STATE_GUARD), "--repo", str(REPO_ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    finally:
        _stop_worker(worker)
