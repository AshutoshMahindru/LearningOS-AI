from __future__ import annotations

import json

from app.core.flagship import (
    ARTIFACT_ADR,
    ARTIFACT_GIT,
    FlagshipError,
    list_learner_artifacts,
    record_adr,
    record_git_artifact,
)
from tests.platform.flagship.conftest import insert_learner, seed_mission


def test_record_and_list_adr(conn) -> None:
    seed_mission(conn, "M07", order_index=7, flagship_version="V01")
    learner_id = insert_learner(conn, "adr-learner")
    record = record_adr(
        conn,
        learner_id=learner_id,
        mission_id="M07",
        title="Reuse one pipeline object",
        context_text="Duplicated preprocessing drifted at inference.",
        decision_text="Fit preprocessing and estimator behind one interface.",
        consequences_text="Reload tests must fail when state diverges.",
        status="ACCEPTED",
    )
    conn.commit()
    assert record["artifact_type"] == ARTIFACT_ADR
    assert record["flagship_version"] == "V01"
    assert record["id"].startswith("ADR-")
    listed = list_learner_artifacts(conn, learner_id)
    assert len(listed[ARTIFACT_ADR]) == 1
    adr = listed[ARTIFACT_ADR][0]
    assert adr["title"] == "Reuse one pipeline object"
    assert adr["mission_id"] == "M07"
    assert adr["flagship_version"] == "V01"
    row = conn.execute("SELECT id, status FROM adrs WHERE learner_id = ?", (learner_id,)).fetchone()
    assert row is not None
    events = conn.execute(
        "SELECT event_type FROM learning_events WHERE learner_id = ?",
        (learner_id,),
    ).fetchall()
    assert any(item["event_type"] == "learner_artifact.adr" for item in events)


def test_record_and_list_git_artifact(conn) -> None:
    seed_mission(conn, "M03", order_index=3, flagship_version="V01")
    learner_id = insert_learner(conn, "git-learner")
    git_sha = "a" * 40
    record = record_git_artifact(
        conn,
        learner_id=learner_id,
        mission_id="M03",
        git_sha=git_sha,
        message="modify working program before learning syntax",
    )
    conn.commit()
    assert record["artifact_type"] == ARTIFACT_GIT
    assert record["git_sha"] == git_sha
    assert record["flagship_version"] == "V01"
    assert len(record["artifact_hash"]) == 64
    listed = list_learner_artifacts(conn, learner_id)
    assert len(listed[ARTIFACT_GIT]) == 1
    assert listed[ARTIFACT_GIT][0]["git_sha"] == git_sha
    events = conn.execute(
        "SELECT payload_json FROM learning_events WHERE learner_id = ? AND event_type = ?",
        (learner_id, "learner_artifact.git"),
    ).fetchall()
    assert events
    body = json.loads(events[0]["payload_json"])
    assert body["git_sha"] == git_sha
    assert "dummy_hash" not in json.dumps(body)


def test_git_sha_must_be_hex_digest(conn) -> None:
    learner_id = insert_learner(conn, "bad-git")
    try:
        record_git_artifact(conn, learner_id=learner_id, mission_id="M01", git_sha="not-a-sha")
    except FlagshipError as exc:
        assert exc.code == "BAD_GIT_SHA"
    else:
        raise AssertionError("expected FlagshipError")
