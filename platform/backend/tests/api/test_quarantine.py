from __future__ import annotations

import sys

from tests.api.helpers import assert_no_forbidden_payload, assert_typed_error

TUTOR_BODY = {
    "session_id": "sess-1",
    "stage_id": "stage-1",
    "role": "SOCRATIC",
    "prompt": "Why does this fail?",
}

PREDICT_BODY = {"hypothesis": "it works", "expected_values": {"n": 1}}
EXECUTE_BODY = {"code": "print(1)", "parameters": {"x": 1}}
SUBMIT_BODY = {"artifacts": [], "explanation": "done"}


def test_tutor_chat_is_501_and_does_not_import_openai(client, auth_headers):
    sys.modules.pop("openai", None)
    response = client.post("/api/v1/tutor/chat", json=TUTOR_BODY, headers=auth_headers)
    error = assert_typed_error(response, 501, "TUTOR_NOT_AVAILABLE")
    assert_no_forbidden_payload(response)
    assert "openai" not in sys.modules
    assert "TUTOR" in error["code"]


def test_runtime_unknown_session_is_typed_not_quarantined(client, auth_headers, monkeypatch):
    from app.core.worker_client import WorkerClient

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("worker execute must not be called for a missing session")

    monkeypatch.setattr(WorkerClient, "execute", _forbidden)

    predict = client.post(
        "/api/v1/sessions/sess-1/stages/stage-1/predict",
        json=PREDICT_BODY,
        headers=auth_headers,
    )
    assert_typed_error(predict, 404, "NOT_FOUND")
    assert_no_forbidden_payload(predict)
    assert predict.status_code != 501

    execute = client.post(
        "/api/v1/sessions/sess-1/stages/stage-1/execute",
        json=EXECUTE_BODY,
        headers=auth_headers,
    )
    assert_typed_error(execute, 404, "NOT_FOUND")
    assert_no_forbidden_payload(execute)

    submit = client.post(
        "/api/v1/sessions/sess-1/stages/stage-1/submit",
        json=SUBMIT_BODY,
        headers=auth_headers,
    )
    assert_typed_error(submit, 404, "NOT_FOUND")
    assert_no_forbidden_payload(submit)

    enter = client.post(
        "/api/v1/sessions/sess-1/stages/stage-1/enter",
        headers=auth_headers,
    )
    assert_typed_error(enter, 404, "NOT_FOUND")
    assert_no_forbidden_payload(enter)

    gates = client.post(
        "/api/v1/sessions/sess-1/gates/evaluate",
        headers=auth_headers,
    )
    assert_typed_error(gates, 404, "NOT_FOUND")
    assert_no_forbidden_payload(gates)


def test_tutor_unauthenticated_is_401(client):
    response = client.post("/api/v1/tutor/chat", json=TUTOR_BODY)
    assert_typed_error(response, 401, "UNAUTHORIZED")
