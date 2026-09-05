from __future__ import annotations

import json

from app.core.flagship import record_adr, record_git_artifact
from tests.platform.flagship.conftest import complete_mission, insert_learner, seed_mission


def test_get_flagship_requires_auth(client) -> None:
    response = client.get("/api/v1/flagship")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_get_flagship_returns_v00_v12_index(client, auth_headers) -> None:
    response = client.get("/api/v1/flagship", headers=auth_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema"] == "learningos.flagship.v1"
    assert body["id"] == "g6.flagship.index"
    assert body["version_count"] == 13
    assert body["mission_count"] == 42
    ids = [item["id"] for item in body["versions"]]
    assert ids == [f"V{i:02d}" for i in range(0, 13)]
    assert body["versions"][0]["missions"] == ["M01", "M02"]
    assert body["versions"][-1]["missions"] == ["M42"]
    assert "M07" in body["missions"]
    assert body["missions"]["M07"]["prerequisites"]["blocking"] == ["M04", "M05", "M06"]
    assert not any(path.endswith("/missions/M42") for path in ("/flagship",))


def test_get_flagship_with_learner_progress_and_artifacts(client, auth_headers, conn) -> None:
    seed_mission(conn, "M01", order_index=1, flagship_version="V00")
    seed_mission(conn, "M02", order_index=2, flagship_version="V00")
    learner_id = insert_learner(conn, "api-flagship")
    complete_mission(conn, learner_id, "M01")
    record_adr(
        conn,
        learner_id=learner_id,
        mission_id="M01",
        title="Start with the useful whole",
        context_text="Layer slogans hide training vs inference.",
        decision_text="Trace a toy system before naming layers.",
        consequences_text="Gate requires a map and a sealed experiment.",
    )
    record_git_artifact(
        conn,
        learner_id=learner_id,
        mission_id="M01",
        git_sha="b" * 40,
        message="system map notes",
    )
    conn.commit()

    missing = client.get("/api/v1/flagship?learner_id=missing", headers=auth_headers)
    assert missing.status_code == 404

    response = client.get(f"/api/v1/flagship?learner_id={learner_id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["learner_id"] == learner_id
    by_id = {item["id"]: item for item in body["progress"]}
    assert by_id["V00"]["status"] == "IN_PROGRESS"
    assert body["artifacts"]["counts"]["adr"] == 1
    assert body["artifacts"]["counts"]["git"] == 1
    assert body["artifacts"]["adr"][0]["title"] == "Start with the useful whole"
    assert body["artifacts"]["git"][0]["git_sha"] == "b" * 40
    assert "dummy_hash" not in json.dumps(body)

    today = client.get(f"/api/v1/learners/{learner_id}/next-action", headers=auth_headers)
    assert today.status_code == 200, today.text
    action = today.json()
    assert action["action"] == "START_MISSION"
    assert action["mission_id"] == "M02"
    assert action["flagship_version"] == "V00"


def test_no_mission_specific_flagship_route(client, auth_headers) -> None:
    response = client.get("/api/v1/missions/M42", headers=auth_headers)
    assert response.status_code == 404
    listed = client.get("/api/v1/flagship", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["versions"][-1]["id"] == "V12"
