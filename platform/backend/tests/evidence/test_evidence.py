from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evidence.conftest import (
    MISSION_A,
    canonical_sha256,
    insert_learner,
    insert_session,
    seed_mission,
)


def _blob(value) -> str:
    return json.dumps(value, sort_keys=True)


def test_libraries_have_no_mission_special_cases() -> None:
    core = Path(__file__).resolve().parents[2] / "app" / "core"
    for name in ("evidence.py", "gates.py", "projection.py"):
        text = (core / name).read_text(encoding="utf-8")
        assert "M01" not in text
        assert "comp.sys." not in text


def test_record_activity_requires_learner_id(conn) -> None:
    from app.core.evidence import record_activity

    with pytest.raises(ValueError, match="learner_id"):
        record_activity(conn, "", "ping", {"ok": True})


def test_evidence_round_trip_and_ledger(conn, data_home) -> None:
    from app.core.evidence import (
        build_claim,
        insert_claim,
        is_real_sha256,
        list_evidence,
        persist_submission_evidence,
        record_activity,
        runner_harness_hash,
    )

    spec = seed_mission(conn, MISSION_A, two_criteria=True)
    learner_id = insert_learner(conn)
    session_id = insert_session(conn, learner_id, MISSION_A, spec)
    from app.api import runtime

    session = runtime.load_session(conn, session_id)
    entered = runtime.enter_stage(conn, session_id, "s1_orient")
    attempt = {"id": entered["attempt_id"], "assistance_level": "UNASSISTED"}
    payload_hash = canonical_sha256({"explanation": "framed", "artifacts": []})
    claims = persist_submission_evidence(
        conn,
        session,
        attempt,
        "s1_orient",
        spec,
        "framed",
        [{"artifact_type": "markdown", "competency_id": "comp.generic.alpha"}],
        payload_hash,
    )
    conn.commit()
    assert claims
    claim = claims[0]
    for field in ("learner_id", "stage_attempt_id", "artifact_hash", "runner_hash", "curriculum_sha"):
        assert claim["provenance"][field]
        assert claim["provenance"][field] != "dummy_hash"
    assert is_real_sha256(claim["artifact_hash"])
    assert is_real_sha256(claim["runner_hash"])
    assert is_real_sha256(claim["curriculum_sha"])
    assert claim["harness_hash"] == claim["runner_hash"]
    assert claim["learner_id"] == learner_id
    assert claim["stage_attempt_id"] == entered["attempt_id"]
    assert claim["competency_id"] == "comp.generic.alpha"
    assert claim["knowledge_node_id"] == "kn.generic.alpha"

    event_id = record_activity(
        conn,
        learner_id,
        "evidence_recorded",
        {"claim_id": claim["id"], "artifact_hash": claim["artifact_hash"]},
    )
    assert event_id

    listed = list_evidence(conn, learner_id, mission_id=MISSION_A, spec=spec)
    assert len(listed) >= 1
    assert listed[0]["id"] == claim["id"]
    assert listed[0]["artifact_hash"] == claim["artifact_hash"]
    assert "dummy_hash" not in _blob(listed)

    rows = conn.execute(
        "SELECT learner_id, event_type, payload_json, event_hash FROM learning_events WHERE learner_id = ?",
        (learner_id,),
    ).fetchall()
    assert rows
    assert all(row["learner_id"] == learner_id for row in rows)
    assert any(row["event_type"] == "evidence_recorded" for row in rows)
    for row in rows:
        assert is_real_sha256(row["event_hash"])
        assert "dummy_hash" not in row["payload_json"]

    dummy = build_claim(
        learner_id=learner_id,
        mission_id=MISSION_A,
        stage_id="s1_orient",
        stage_attempt_id=entered["attempt_id"],
        competency_id="comp.generic.alpha",
        knowledge_node_id="kn.generic.alpha",
        artifact_type="markdown",
        artifact_hash="dummy_hash",
        runner_hash="dummy_hash",
        curriculum_sha="dummy_hash",
    )
    assert dummy["artifact_hash"] != "dummy_hash"
    assert dummy["runner_hash"] != "dummy_hash"
    assert dummy["curriculum_sha"] != "dummy_hash"
    assert is_real_sha256(dummy["artifact_hash"])
    stored = insert_claim(conn, dummy)
    conn.commit()
    assert stored["artifact_hash"] != "dummy_hash"
    assert data_home.name == "learningos-home"
    harness = runner_harness_hash(conn, entered["attempt_id"], spec, "s2_lab")
    assert is_real_sha256(harness)
    assert harness != "dummy_hash"
