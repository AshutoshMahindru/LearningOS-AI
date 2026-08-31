from __future__ import annotations

import base64
from pathlib import Path

from tests.api.helpers import assert_typed_error


def test_artifacts_return_501_when_store_missing(client, auth_headers, monkeypatch):
    from app.api import routes
    from app.core.errors import StorageUnavailableError

    def _missing():
        raise StorageUnavailableError("Artifact store is not available")

    monkeypatch.setattr(routes, "_get_artifact_store", _missing)
    created = client.post(
        "/api/v1/artifacts",
        json={"bytes_b64": base64.b64encode(b"hello").decode("ascii")},
        headers=auth_headers,
    )
    assert_typed_error(created, 501, "STORAGE_UNAVAILABLE")
    fetched = client.get("/api/v1/artifacts/" + "a" * 64, headers=auth_headers)
    assert_typed_error(fetched, 501, "STORAGE_UNAVAILABLE")


def test_curriculum_returns_501_when_loader_missing_methods(client, auth_headers, monkeypatch):
    from app.api import routes

    def _missing(_name: str, *args, **kwargs):
        raise ImportError(_name)

    monkeypatch.setattr(routes.importlib, "import_module", _missing)
    load = client.post(
        "/api/v1/curriculum/packages/load",
        json={"package_dir": "/tmp/missing-package"},
        headers=auth_headers,
    )
    assert_typed_error(load, 501, "CURRICULUM_UNAVAILABLE")
    listed = client.get("/api/v1/curriculum/packages", headers=auth_headers)
    assert_typed_error(listed, 501, "CURRICULUM_UNAVAILABLE")


def test_backup_restore_return_501_when_storage_missing(client, auth_headers, monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "_import_backup_fn", lambda _name: None)
    backup = client.post("/api/v1/system/backup", headers=auth_headers)
    assert_typed_error(backup, 501, "STORAGE_UNAVAILABLE")
    restore = client.post(
        "/api/v1/system/restore",
        json={"backup_id": "backup_test.tar.gz"},
        headers=auth_headers,
    )
    assert_typed_error(restore, 501, "STORAGE_UNAVAILABLE")


def test_learners_and_sessions_typed_when_storage_unavailable(client, auth_headers):
    created = client.post(
        "/api/v1/learners",
        json={"username": "ada", "display_name": "Ada"},
        headers=auth_headers,
    )
    assert created.status_code in {200, 409, 500, 501, 503}
    if created.status_code >= 400:
        body = created.json()
        assert "error" in body
        assert body["error"]["code"] in {"STORAGE_UNAVAILABLE", "INTERNAL", "CONFLICT"}

    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": "g3.fixture.orientation", "learner_id": "learner-1"},
        headers=auth_headers,
    )
    assert session.status_code in {200, 404, 500, 501, 503}
    if session.status_code >= 400:
        body = session.json()
        assert "error" in body
        assert body["error"]["code"] in {"STORAGE_UNAVAILABLE", "INTERNAL", "NOT_FOUND"}
    text = session.text.lower()
    assert "learner_default" not in text or "learner_id" in session.text


def test_tmp_home_does_not_write_into_repo(client, data_home):
    client.post("/api/v1/auth/bootstrap")
    repo_backend = Path(__file__).resolve().parents[2]
    stray = list(repo_backend.glob("**/learningos.db"))
    assert stray == []
    assert (data_home / ".auth_token").is_file()
    developer_token = Path.home() / ".learningos" / ".auth_token"
    if developer_token.exists():
        assert developer_token.resolve() != (data_home / ".auth_token").resolve()
