from __future__ import annotations

import os
import socket
import threading
from pathlib import Path




def test_health_reports_real_path_and_version(client, data_home):
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "3.0.0"
    assert isinstance(data["worker_alive"], bool)
    assert data["worker_alive"] is False
    database_path = Path(data["database_path"])
    assert database_path.is_absolute()
    assert database_path.name == "learningos.db"
    assert database_path.parent == data_home.resolve()
    assert data["database_path"] != "TBD"


def test_version_endpoint(client):
    response = client.get("/api/v1/system/version")
    assert response.status_code == 200
    assert response.json() == {"version": "3.0.0"}


def test_health_worker_alive_when_socket_accepts(data_home, monkeypatch):
    sock_path = Path(f"/tmp/learningos-g3-test-worker-{os.getpid()}.sock")
    if sock_path.exists():
        sock_path.unlink()
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock_path))

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(sock_path))
        server.listen(1)
        server.settimeout(2)

        def _accept() -> None:
            try:
                conn, _ = server.accept()
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=_accept, daemon=True)
        thread.start()

        from app.main import app
        from fastapi.testclient import TestClient

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            response = client.get("/api/v1/system/health")
        thread.join(timeout=1)
    finally:
        server.close()
        try:
            sock_path.unlink()
        except OSError:
            pass

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["worker_alive"] is True


def test_health_survives_worker_client_health_error(client, monkeypatch):
    from app.api import routes

    def _boom() -> bool:
        raise RuntimeError("worker probe failed")

    monkeypatch.setattr(routes, "worker_alive", _boom)
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["worker_alive"] is False


def test_unauthenticated_health_does_not_use_typed_error(client):
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    assert "error" not in response.json()
