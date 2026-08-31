from __future__ import annotations

import hashlib
import json
import sys

from fastapi.testclient import TestClient

from tests.api.helpers import (
    RUNTIME_MISSION_ID,
    assert_no_forbidden_payload,
    assert_typed_error,
    canonical_sha256,
    seed_runtime_mission,
)


PREDICT_BODY = {"hypothesis": "output grows", "expected_values": {"y": 2}}
EXECUTE_BODY = {"code": "print(1)", "parameters": {"x": 1}}
SUBMIT_BODY = {"artifacts": [{"artifact_type": "trace"}], "explanation": "observed y=2"}


def _seed(client) -> None:
    from app.db.database import get_connection

    conn = get_connection()
    try:
        seed_runtime_mission(conn)
    finally:
        conn.close()


def _start_session(client, auth_headers) -> dict:
    _seed(client)
    learner = client.post(
        "/api/v1/learners",
        json={"username": "runtime-learner", "display_name": "Runtime"},
        headers=auth_headers,
    )
    assert learner.status_code == 200, learner.text
    learner_id = learner.json()["learner_id"]
    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": RUNTIME_MISSION_ID, "learner_id": learner_id},
        headers=auth_headers,
    )
    assert session.status_code == 200, session.text
    assert_no_forbidden_payload(session)
    body = session.json()
    assert body["mission_id"] == RUNTIME_MISSION_ID
    assert body["status"] == "ACTIVE"
    assert body["current_stage_id"] == "s1_orient"
    return {"learner_id": learner_id, "session_id": body["session_id"], "headers": auth_headers}


def _unsupported_execute(self, code, parameters=None):
    return {
        "status": "UNSUPPORTED",
        "job_id": "job_runtime_test",
        "reason": "execution sandbox is WP400",
    }


def test_enter_predict_execute_submit_gate_happy_path(client, auth_headers, monkeypatch):
    from app.core.worker_client import WorkerClient

    monkeypatch.setattr(WorkerClient, "execute", _unsupported_execute)
    ctx = _start_session(client, auth_headers)
    session_id = ctx["session_id"]
    headers = ctx["headers"]

    entered = client.post(f"/api/v1/sessions/{session_id}/stages/s1_orient/enter", headers=headers)
    assert entered.status_code == 200, entered.text
    assert_no_forbidden_payload(entered)
    enter_body = entered.json()
    assert enter_body["status"] == "ACTIVE"
    assert enter_body["stage_id"] == "s1_orient"
    assert enter_body["current_stage_id"] == "s1_orient"
    attempt_id = enter_body["attempt_id"]

    again = client.post(f"/api/v1/sessions/{session_id}/stages/s1_orient/enter", headers=headers)
    assert again.status_code == 200, again.text
    assert again.json()["attempt_id"] == attempt_id
    assert again.json()["resumed"] is True

    skip = client.post(f"/api/v1/sessions/{session_id}/stages/s2_lab/enter", headers=headers)
    assert_typed_error(skip, 409, "CONFLICT")
    assert_no_forbidden_payload(skip)

    submitted = client.post(
        f"/api/v1/sessions/{session_id}/stages/s1_orient/submit",
        json={"explanation": "framed the problem"},
        headers=headers,
    )
    assert submitted.status_code == 200, submitted.text
    assert_no_forbidden_payload(submitted)
    assert submitted.json()["status"] == "SUBMITTED"
    assert submitted.json()["current_stage_id"] == "s2_lab"
    assert submitted.json()["next_stage_id"] == "s2_lab"

    lab_enter = client.post(f"/api/v1/sessions/{session_id}/stages/s2_lab/enter", headers=headers)
    assert lab_enter.status_code == 200, lab_enter.text
    assert lab_enter.json()["stage_type"] == "experiment"

    blocked = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/execute",
        json=EXECUTE_BODY,
        headers=headers,
    )
    error = assert_typed_error(blocked, 409, "CONFLICT")
    assert error["details"].get("reason") == "PREDICTION_REQUIRED"
    assert_no_forbidden_payload(blocked)

    predicted = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/predict",
        json=PREDICT_BODY,
        headers=headers,
    )
    assert predicted.status_code == 200, predicted.text
    assert_no_forbidden_payload(predicted)
    pred = predicted.json()
    expected_hash = canonical_sha256(
        {"expected_values": PREDICT_BODY["expected_values"], "hypothesis": PREDICT_BODY["hypothesis"]}
    )
    assert pred["prediction_hash"] == expected_hash
    assert pred["prediction_hash"] != "dummy_hash"
    assert len(pred["prediction_hash"]) == 64
    assert pred["is_sealed"] is True

    executed = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/execute",
        json=EXECUTE_BODY,
        headers=headers,
    )
    assert executed.status_code in {200, 202}, executed.text
    assert_no_forbidden_payload(executed)
    exec_body = executed.json()
    assert exec_body["status"] == "UNSUPPORTED"
    assert exec_body["code_hash"] != "dummy_hash"
    assert exec_body["code_hash"] == hashlib.sha256(EXECUTE_BODY["code"].encode("utf-8")).hexdigest()
    assert exec_body["execution_id"]

    lab_submit = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/submit",
        json=SUBMIT_BODY,
        headers=headers,
    )
    assert lab_submit.status_code == 200, lab_submit.text
    assert_no_forbidden_payload(lab_submit)
    assert lab_submit.json()["current_stage_id"] == "s3_wrap"

    wrap = client.post(f"/api/v1/sessions/{session_id}/stages/s3_wrap/enter", headers=headers)
    assert wrap.status_code == 200, wrap.text
    wrap_submit = client.post(
        f"/api/v1/sessions/{session_id}/stages/s3_wrap/submit",
        json={"explanation": "ready for gate"},
        headers=headers,
    )
    assert wrap_submit.status_code == 200, wrap_submit.text

    gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
    assert gate.status_code == 200, gate.text
    assert_no_forbidden_payload(gate)
    gate_body = gate.json()
    assert gate_body["status"] == "PASSED"
    assert gate_body["reason"] == "GATE_CRITERIA_MET"

    got = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert got.status_code == 200, got.text
    assert got.json()["status"] == "COMPLETED"
    assert got.json()["current_stage_id"] == "s3_wrap"


def test_resume_after_new_testclient(data_home, client, auth_headers):
    ctx = _start_session(client, auth_headers)
    session_id = ctx["session_id"]
    headers = ctx["headers"]
    entered = client.post(f"/api/v1/sessions/{session_id}/stages/s1_orient/enter", headers=headers)
    assert entered.status_code == 200, entered.text
    attempt_id = entered.json()["attempt_id"]

    from app.main import app

    with TestClient(app, client=("127.0.0.1", 50000)) as resumed:
        boot = resumed.post("/api/v1/auth/bootstrap")
        assert boot.status_code == 200, boot.text
        resume_headers = {"Authorization": f"Bearer {boot.json()['token']}"}
        got = resumed.get(f"/api/v1/sessions/{session_id}", headers=resume_headers)
        assert got.status_code == 200, got.text
        assert_no_forbidden_payload(got)
        body = got.json()
        assert body["current_stage_id"] == "s1_orient"
        assert body["status"] == "ACTIVE"
        assert body.get("current_stage", {}).get("id") == "s1_orient"
        again = resumed.post(f"/api/v1/sessions/{session_id}/stages/s1_orient/enter", headers=resume_headers)
        assert again.status_code == 200, again.text
        assert again.json()["attempt_id"] == attempt_id
        assert again.json()["status"] == "ACTIVE"


def test_runtime_routes_require_auth(client):
    paths = [
        ("post", "/api/v1/sessions/sess/stages/s1/enter", None),
        ("post", "/api/v1/sessions/sess/stages/s1/predict", PREDICT_BODY),
        ("post", "/api/v1/sessions/sess/stages/s1/execute", EXECUTE_BODY),
        ("post", "/api/v1/sessions/sess/stages/s1/submit", SUBMIT_BODY),
        ("post", "/api/v1/sessions/sess/gates/evaluate", None),
    ]
    for _method, path, body in paths:
        response = client.post(path, json=body) if body is not None else client.post(path)
        assert_typed_error(response, 401, "UNAUTHORIZED")
        assert_no_forbidden_payload(response)


def test_execute_does_not_exec_user_code_or_import_openai(client, auth_headers, monkeypatch):
    from app.core.worker_client import WorkerClient

    def _boom(*_args, **_kwargs):
        raise AssertionError("API process must not exec or eval untrusted code")

    monkeypatch.setattr("builtins.exec", _boom)
    monkeypatch.setattr("builtins.eval", _boom)
    monkeypatch.setattr(WorkerClient, "execute", _unsupported_execute)
    sys.modules.pop("openai", None)

    ctx = _start_session(client, auth_headers)
    session_id = ctx["session_id"]
    headers = ctx["headers"]
    assert client.post(f"/api/v1/sessions/{session_id}/stages/s1_orient/enter", headers=headers).status_code == 200
    assert client.post(
        f"/api/v1/sessions/{session_id}/stages/s1_orient/submit",
        json={"explanation": "go"},
        headers=headers,
    ).status_code == 200
    assert client.post(f"/api/v1/sessions/{session_id}/stages/s2_lab/enter", headers=headers).status_code == 200
    predicted = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/predict",
        json=PREDICT_BODY,
        headers=headers,
    )
    assert predicted.status_code == 200, predicted.text
    executed = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/execute",
        json={"code": "raise SystemExit('should stay in worker')", "parameters": {}},
        headers=headers,
    )
    assert executed.status_code in {200, 202}, executed.text
    assert_no_forbidden_payload(executed)
    assert executed.json()["status"] == "UNSUPPORTED"
    assert "openai" not in sys.modules


def test_gate_without_contract_returns_generic_placeholder(client, auth_headers):
    from app.db.database import get_connection

    conn = get_connection()
    try:
        seed_runtime_mission(conn, include_gate=False)
    finally:
        conn.close()
    learner = client.post(
        "/api/v1/learners",
        json={"username": "runtime-placeholder", "display_name": "Runtime"},
        headers=auth_headers,
    )
    assert learner.status_code == 200, learner.text
    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": RUNTIME_MISSION_ID, "learner_id": learner.json()["learner_id"]},
        headers=auth_headers,
    )
    assert session.status_code == 200, session.text
    session_id = session.json()["session_id"]
    gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=auth_headers)
    assert gate.status_code == 200, gate.text
    assert_no_forbidden_payload(gate)
    body = gate.json()
    assert body["status"] == "EVALUATED"
    assert body["reason"] == "GENERIC_PLACEHOLDER_NO_WP400_ASSESSMENT"
    assert "dummy_hash" not in json.dumps(body)
