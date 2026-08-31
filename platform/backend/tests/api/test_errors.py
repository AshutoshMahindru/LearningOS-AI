from __future__ import annotations

from tests.api.helpers import assert_typed_error


def test_session_create_requires_learner_id(client, auth_headers):
    response = client.post(
        "/api/v1/sessions",
        json={"mission_id": "g3.fixture.orientation"},
        headers=auth_headers,
    )
    error = assert_typed_error(response, 422, "VALIDATION_ERROR")
    assert error["details"]
    blob = str(error["details"]).lower()
    assert "learner_id" in blob


def test_unknown_route_is_typed_404(client):
    response = client.get("/api/v1/does-not-exist")
    assert_typed_error(response, 404, "NOT_FOUND")


def test_restore_requires_target(client, auth_headers):
    response = client.post("/api/v1/system/restore", json={}, headers=auth_headers)
    assert_typed_error(response, 422, "VALIDATION_ERROR")
