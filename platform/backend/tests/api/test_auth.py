from __future__ import annotations

import stat

from fastapi.testclient import TestClient

from tests.api.helpers import assert_typed_error


def test_unauthenticated_protected_route_returns_typed_401(client):
    response = client.get("/api/v1/missions")
    error = assert_typed_error(response, 401, "UNAUTHORIZED")
    assert "error" in response.json()
    assert error["details"] == {} or isinstance(error["details"], dict)


def test_bootstrap_returns_token_and_accesses_protected_route(client, data_home):
    response = client.post("/api/v1/auth/bootstrap")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {"token", "token_type"}
    assert payload["token_type"] == "bearer"
    token = payload["token"]
    assert isinstance(token, str) and len(token) == 64

    token_path = data_home / ".auth_token"
    assert token_path.is_file()
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    assert token_path.read_text(encoding="ascii").strip() == token

    denied = client.get("/api/v1/missions")
    assert_typed_error(denied, 401, "UNAUTHORIZED")

    allowed = client.get("/api/v1/missions", headers={"Authorization": f"Bearer {token}"})
    assert allowed.status_code != 401
    if allowed.status_code == 501:
        assert allowed.json()["error"]["code"] in {"STORAGE_UNAVAILABLE", "INTERNAL"}
    else:
        assert allowed.status_code == 200
        assert "missions" in allowed.json()


def test_invalid_token_is_401(client, auth_headers):
    response = client.get("/api/v1/missions", headers={"Authorization": "Bearer deadbeef"})
    assert_typed_error(response, 401, "UNAUTHORIZED")


def test_logout_rotates_token(client, auth_headers):
    first = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert first.status_code == 200
    assert first.json()["status"] == "LOGGED_OUT"

    stale = client.get("/api/v1/missions", headers=auth_headers)
    assert_typed_error(stale, 401, "UNAUTHORIZED")

    bootstrap = client.post("/api/v1/auth/bootstrap")
    new_token = bootstrap.json()["token"]
    assert new_token != auth_headers["Authorization"].split(" ", 1)[1]
    refreshed = client.get("/api/v1/missions", headers={"Authorization": f"Bearer {new_token}"})
    assert refreshed.status_code != 401


def test_bootstrap_rejected_from_non_loopback(data_home):
    from app.main import app

    with TestClient(app, client=("8.8.8.8", 12345)) as remote:
        response = remote.post("/api/v1/auth/bootstrap")
    assert_typed_error(response, 403, "UNAUTHORIZED")
