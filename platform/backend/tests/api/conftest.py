from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    return home


@pytest.fixture
def client(data_home):
    from app.main import app

    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    response = client.post("/api/v1/auth/bootstrap")
    assert response.status_code == 200, response.text
    payload = response.json()
    token = payload["token"]
    assert payload["token_type"] == "bearer"
    return {"Authorization": f"Bearer {token}"}
