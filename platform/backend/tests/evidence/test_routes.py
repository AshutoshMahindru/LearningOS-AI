from __future__ import annotations

import json

from tests.api.helpers import assert_no_forbidden_payload, assert_typed_error
from tests.evidence.conftest import MISSION_A, MISSION_B, seed_mission


def _unsupported_execute(self, code, parameters=None):
    return {"status": "UNSUPPORTED", "job_id": "job_evidence", "reason": "generic"}


def test_evidence_and_next_action_routes(client, auth_headers, conn) -> None:
    from app.core.worker_client import WorkerClient

    seed_mission(conn, MISSION_A, order_index=1, two_criteria=True)
    seed_mission(conn, MISSION_B, order_index=2, two_criteria=False)

    learner = client.post(
        "/api/v1/learners",
        json={"username": "route-learner", "display_name": "Route"},
        headers=auth_headers,
    )
    assert learner.status_code == 200, learner.text
    learner_id = learner.json()["learner_id"]

    missing = client.get("/api/v1/learners/missing/evidence", headers=auth_headers)
    assert_typed_error(missing, 404, "NOT_FOUND")

    unauth = client.get(f"/api/v1/learners/{learner_id}/evidence")
    assert_typed_error(unauth, 401, "UNAUTHORIZED")

    today = client.get(f"/api/v1/learners/{learner_id}/next-action", headers=auth_headers)
    assert today.status_code == 200, today.text
    assert_no_forbidden_payload(today)
    body = today.json()
    assert body["action"] == "START_MISSION"
    assert body["mission_id"] == MISSION_A
    assert body["today"]["mission_id"] == MISSION_A
    assert "M01" not in json.dumps(body)

    session = client.post(
        "/api/v1/sessions",
        json={"mission_id": MISSION_A, "learner_id": learner_id},
        headers=auth_headers,
    )
    assert session.status_code == 200, session.text
    session_id = session.json()["session_id"]

    assert client.post(
        f"/api/v1/sessions/{session_id}/stages/s1_orient/enter",
        headers=auth_headers,
    ).status_code == 200
    submitted = client.post(
        f"/api/v1/sessions/{session_id}/stages/s1_orient/submit",
        json={"explanation": "framed the problem"},
        headers=auth_headers,
    )
    assert submitted.status_code == 200, submitted.text

    evidence = client.get(f"/api/v1/learners/{learner_id}/evidence", headers=auth_headers)
    assert evidence.status_code == 200, evidence.text
    assert_no_forbidden_payload(evidence)
    payload = evidence.json()
    assert payload["learner_id"] == learner_id
    assert payload["evidence"]
    claim = payload["evidence"][0]
    provenance = claim["provenance"]
    assert provenance["learner_id"] == learner_id
    assert provenance["stage_attempt_id"]
    assert provenance["artifact_hash"] != "dummy_hash"
    assert provenance["runner_hash"] != "dummy_hash"
    assert provenance["curriculum_sha"] != "dummy_hash"
    assert len(provenance["artifact_hash"]) == 64
    assert "dummy_hash" not in json.dumps(payload)

    gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=auth_headers)
    assert gate.status_code == 200, gate.text
    assert_no_forbidden_payload(gate)
    gate_body = gate.json()
    assert gate_body["status"] == "REPAIR_REQUIRED"
    assert gate_body["repair_plan"]["failed_knowledge_nodes"]
    assert "kn.generic.beta" in gate_body["repair_plan"]["failed_knowledge_nodes"]
    assert gate_body["reason"] != "GENERIC_PLACEHOLDER_NO_WP400_ASSESSMENT"

    assert client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/enter",
        headers=auth_headers,
    ).status_code == 200
    predict = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/predict",
        json={"hypothesis": "output grows", "expected_values": {"y": 2}},
        headers=auth_headers,
    )
    assert predict.status_code == 200, predict.text
    from unittest.mock import patch

    with patch.object(WorkerClient, "execute", _unsupported_execute):
        executed = client.post(
            f"/api/v1/sessions/{session_id}/stages/s2_lab/execute",
            json={"code": "print(1)", "parameters": {}},
            headers=auth_headers,
        )
    assert executed.status_code in {200, 202}, executed.text
    lab = client.post(
        f"/api/v1/sessions/{session_id}/stages/s2_lab/submit",
        json={"explanation": "observed", "artifacts": [{"artifact_type": "trace"}]},
        headers=auth_headers,
    )
    assert lab.status_code == 200, lab.text

    passed = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=auth_headers)
    assert passed.status_code == 200, passed.text
    assert passed.json()["status"] == "PASSED"
    assert passed.json()["reason"] == "GATE_CRITERIA_MET"
    increments = passed.json().get("competency_increments") or []
    assert {item["competency_id"] for item in increments} == {"comp.generic.alpha", "comp.generic.beta"}

    following = client.get(f"/api/v1/learners/{learner_id}/next-action", headers=auth_headers)
    assert following.status_code == 200, following.text
    assert following.json()["action"] == "START_MISSION"
    assert following.json()["mission_id"] == MISSION_B
    assert "dummy_hash" not in following.text
    assert "M01" not in following.text
