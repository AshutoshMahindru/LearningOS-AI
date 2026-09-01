from __future__ import annotations

import json

from app.core.projection import next_action, project_competencies
from tests.evidence.conftest import (
    MISSION_A,
    MISSION_B,
    insert_learner,
    insert_session,
    seed_mission,
)


def test_next_action_start_then_stage_then_next_mission(conn) -> None:
    spec_a = seed_mission(conn, MISSION_A, order_index=1, two_criteria=False)
    seed_mission(conn, MISSION_B, order_index=2, two_criteria=False)
    learner_id = insert_learner(conn, "proj-learner")

    idle_start = next_action(conn, learner_id)
    assert idle_start["action"] == "START_MISSION"
    assert idle_start["mission_id"] == MISSION_A
    assert idle_start["stage_id"] == "s1_orient"
    assert idle_start["today"]["mission_id"] == MISSION_A
    assert idle_start["reason"] == "NEXT_MISSION"
    assert "M01" not in json.dumps(idle_start)

    session_id = insert_session(conn, learner_id, MISSION_A, spec_a)
    current = next_action(conn, learner_id)
    assert current["action"] == "ENTER_STAGE"
    assert current["session_id"] == session_id
    assert current["mission_id"] == MISSION_A
    assert current["stage_id"] == "s1_orient"

    from app.api import runtime

    runtime.enter_stage(conn, session_id, "s1_orient")
    in_progress = next_action(conn, learner_id)
    assert in_progress["action"] == "CONTINUE_STAGE"
    assert in_progress["stage_id"] == "s1_orient"

    runtime.submit_stage(conn, session_id, "s1_orient", "framed", [])
    after_submit = next_action(conn, learner_id)
    assert after_submit["stage_id"] == "s2_lab"
    assert after_submit["action"] == "ENTER_STAGE"

    runtime.enter_stage(conn, session_id, "s2_lab")
    runtime.submit_stage(conn, session_id, "s2_lab", "lab", [{"artifact_type": "trace"}])
    runtime.enter_stage(conn, session_id, "s3_wrap")
    runtime.submit_stage(conn, session_id, "s3_wrap", "ready", [])
    session = runtime.load_session(conn, session_id)
    runtime.evaluate_gate(conn, session_id)
    session = runtime.load_session(conn, session_id)
    assert session["status"] == "COMPLETED"

    following = next_action(conn, learner_id)
    assert following["action"] == "START_MISSION"
    assert following["mission_id"] == MISSION_B
    assert following["stage_id"] == "s1_orient"
    assert following["reason"] == "NEXT_MISSION"
    assert following["today"]["action"] == "START_MISSION"

    comps = project_competencies(conn, learner_id)
    assert any(item["competency_id"] == "comp.generic.alpha" for item in comps)
    assert all("comp.sys." not in item["competency_id"] for item in comps)
    assert "dummy_hash" not in json.dumps(following)
    assert "dummy_hash" not in json.dumps(comps)


def test_next_action_idle_without_catalog(conn) -> None:
    learner_id = insert_learner(conn, "proj-empty")
    result = next_action(conn, learner_id)
    assert result["action"] == "IDLE"
    assert result["reason"] == "NO_AVAILABLE_MISSIONS"
    assert result["mission_id"] is None
    assert result["today"]["action"] == "IDLE"


def test_next_action_evaluate_gate_when_last_stage_submitted(conn) -> None:
    spec = seed_mission(conn, MISSION_A, two_criteria=False)
    learner_id = insert_learner(conn, "proj-gate")
    session_id = insert_session(conn, learner_id, MISSION_A, spec)
    from app.api import runtime

    runtime.enter_stage(conn, session_id, "s1_orient")
    runtime.submit_stage(conn, session_id, "s1_orient", "framed", [])
    runtime.enter_stage(conn, session_id, "s2_lab")
    runtime.submit_stage(conn, session_id, "s2_lab", "lab", [])
    runtime.enter_stage(conn, session_id, "s3_wrap")
    runtime.submit_stage(conn, session_id, "s3_wrap", "ready", [])
    pending = next_action(conn, learner_id)
    assert pending["action"] == "EVALUATE_GATE"
    assert pending["session_id"] == session_id
    assert pending["mission_id"] == MISSION_A
