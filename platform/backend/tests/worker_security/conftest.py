from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ROOT = BACKEND_ROOT.parent
WORKER_ROOT = PLATFORM_ROOT / "worker"
for _path in (BACKEND_ROOT, WORKER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    return home


@pytest.fixture
def worker_env(data_home, monkeypatch):
    sock = Path(f"/tmp/los-g3-{uuid.uuid4().hex}.sock")
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(data_home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENAI_API_KEY"] = "sk-should-never-reach-child"
    try:
        yield {"home": data_home, "sock": sock, "env": env}
    finally:
        sock.unlink(missing_ok=True)


@pytest.fixture
def client(data_home):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    response = client.post("/api/v1/auth/bootstrap")
    assert response.status_code == 200, response.text
    payload = response.json()
    return {"Authorization": f"Bearer {payload['token']}"}
