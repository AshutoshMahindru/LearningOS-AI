from __future__ import annotations

import json

from app.core.evidence import is_real_sha256, persist_submission_evidence
from app.core.gates import (
    GATE_CONTRACT_ABSENT,
    GATE_CRITERIA_MET,
    GATE_CRITERIA_UNMET,
    evaluate_contract,
    evaluate_session_gate,
)
from tests.evidence.conftest import (
    MISSION_A,
    canonical_sha256,
    insert_learner,
    insert_session,
    seed_mission,
)


def test_evaluate_contract_absent_keeps_typed_placeholder() -> None:
    result = evaluate_contract(None, [])
    assert result["status"] == "EVALUATED"
    assert result["reason"] == GATE_CONTRACT_ABSENT
    assert result["reason"] == "GENERIC_PLACEHOLDER_NO_WP400_ASSESSMENT"
    assert "dummy_hash" not in json.dumps(result)


def test_evaluate_contract_pass_and_repair_from_contract_nodes() -> None:
    spec = {
        "competencies": ["comp.generic.alpha", "comp.generic.beta"],
        "knowledge_nodes": ["kn.generic.alpha", "kn.generic.beta"],
        "gate_contract": {
            "required_evidence": [
                {
                    "competency_id": "comp.generic.alpha",
                    "stage_id": "s1_orient",
                    "artifact_type": "markdown",
                    "knowledge_node_id": "kn.generic.alpha",
                },
                {
                    "competency_id": "comp.generic.beta",
                    "stage_id": "s2_lab",
                    "artifact_type": "trace",
                    "knowledge_node_id": "kn.generic.beta",
                },
            ],
            "pass_threshold": 1.0,
            "repair_policy": {"allow_targeted_repair": True},
        },
    }
    digest = canonical_sha256({"ok": True})
    alpha_claim = {
        "competency_id": "comp.generic.alpha",
        "stage_id": "s1_orient",
        "artifact_type": "markdown",
        "artifact_hash": digest,
        "assessment_status": "ACCEPTED",
    }
    repair = evaluate_contract(spec["gate_contract"], [alpha_claim], spec=spec)
    assert repair["status"] == "REPAIR_REQUIRED"
    assert repair["reason"] == GATE_CRITERIA_UNMET
    assert repair["passed_criteria"] == 1
    assert repair["required_criteria"] == 2
    plan = repair["repair_plan"]
    assert "kn.generic.beta" in plan["failed_knowledge_nodes"]
    assert plan["drills"][0]["knowledge_node_id"] == "kn.generic.beta"
    assert plan["drills"][0]["competency_id"] == "comp.generic.beta"
    assert all(item.get("competency_id") != "comp.sys.hypothesis_testing" for item in plan["drills"])

    both = [
        alpha_claim,
        {
            "competency_id": "comp.generic.beta",
            "stage_id": "s2_lab",
            "artifact_type": "trace",
            "artifact_hash": digest,
            "assessment_status": "ACCEPTED",
        },
    ]
    passed = evaluate_contract(spec["gate_contract"], both, spec=spec)
    assert passed["status"] == "PASSED"
    assert passed["reason"] == GATE_CRITERIA_MET
    ids = [item["competency_id"] for item in passed["competency_increments"]]
    assert ids == ["comp.generic.alpha", "comp.generic.beta"]
    assert "comp.sys." not in json.dumps(passed)
    assert "dummy_hash" not in json.dumps(passed)


def test_session_gate_pass_increments_contract_competencies(conn) -> None:
    spec = seed_mission(conn, MISSION_A, two_criteria=False)
    learner_id = insert_learner(conn, "gate-pass")
    session_id = insert_session(conn, learner_id, MISSION_A, spec)
    from app.api import runtime

    runtime.enter_stage(conn, session_id, "s1_orient")
    submitted = runtime.submit_stage(conn, session_id, "s1_orient", "framed", [])
    assert submitted["status"] == "SUBMITTED"
    session = runtime.load_session(conn, session_id)
    result = evaluate_session_gate(conn, session, spec)
    assert result["status"] == "PASSED"
    assert result["reason"] == GATE_CRITERIA_MET
    assert result["session_status"] == "COMPLETED"
    increments = result["competency_increments"]
    assert increments
    assert {item["competency_id"] for item in increments} == {"comp.generic.alpha"}
    row = conn.execute(
        "SELECT competency_id, level FROM competency_mastery WHERE learner_id = ?",
        (learner_id,),
    ).fetchone()
    assert row["competency_id"] == "comp.generic.alpha"
    assert int(row["level"]) >= 1
    refreshed = runtime.load_session(conn, session_id)
    assert refreshed["status"] == "COMPLETED"
    assert "dummy_hash" not in json.dumps(result)


def test_session_gate_repair_plan_from_failed_nodes(conn) -> None:
    spec = seed_mission(conn, MISSION_A, two_criteria=True)
    learner_id = insert_learner(conn, "gate-repair")
    session_id = insert_session(conn, learner_id, MISSION_A, spec)
    from app.api import runtime

    runtime.enter_stage(conn, session_id, "s1_orient")
    runtime.submit_stage(conn, session_id, "s1_orient", "framed", [])
    session = runtime.load_session(conn, session_id)
    result = evaluate_session_gate(conn, session, spec)
    assert result["status"] == "REPAIR_REQUIRED"
    assert result["reason"] == GATE_CRITERIA_UNMET
    plan = result["repair_plan"]
    assert "kn.generic.beta" in plan["failed_knowledge_nodes"]
    assert any(drill.get("stage_id") == "s2_lab" for drill in plan["drills"])
    mastery = conn.execute(
        "SELECT COUNT(*) AS n FROM competency_mastery WHERE learner_id = ?",
        (learner_id,),
    ).fetchone()
    assert int(mastery["n"]) == 0

    runtime.enter_stage(conn, session_id, "s2_lab")
    runtime.submit_stage(
        conn,
        session_id,
        "s2_lab",
        "observed",
        [{"artifact_type": "trace"}],
    )
    session = runtime.load_session(conn, session_id)
    passed = evaluate_session_gate(conn, session, spec)
    assert passed["status"] == "PASSED"
    ids = {item["competency_id"] for item in passed["competency_increments"]}
    assert ids == {"comp.generic.alpha", "comp.generic.beta"}
    assert "dummy_hash" not in json.dumps(passed)


def test_gate_without_contract_via_runtime(conn) -> None:
    spec = seed_mission(conn, MISSION_A, include_gate=False)
    learner_id = insert_learner(conn, "gate-absent")
    session_id = insert_session(conn, learner_id, MISSION_A, spec)
    from app.api import runtime

    body = runtime.evaluate_gate(conn, session_id)
    assert body["status"] == "EVALUATED"
    assert body["reason"] == "GENERIC_PLACEHOLDER_NO_WP400_ASSESSMENT"
    assert "dummy_hash" not in json.dumps(body)


def test_dummy_hash_claim_does_not_satisfy_criterion(conn) -> None:
    spec = seed_mission(conn, MISSION_A, two_criteria=False)
    learner_id = insert_learner(conn, "gate-dummy")
    session_id = insert_session(conn, learner_id, MISSION_A, spec)
    from app.api import runtime

    session = runtime.load_session(conn, session_id)
    entered = runtime.enter_stage(conn, session_id, "s1_orient")
    persist_submission_evidence(
        conn,
        session,
        {"id": entered["attempt_id"], "assistance_level": "UNASSISTED"},
        "s1_orient",
        spec,
        "framed",
        [],
        canonical_sha256({"explanation": "framed"}),
    )
    conn.commit()
    claims = conn.execute("SELECT artifact_hash FROM evidence_items WHERE learner_id = ?", (learner_id,)).fetchall()
    assert claims
    assert all(is_real_sha256(row["artifact_hash"]) for row in claims)
    assert all(row["artifact_hash"] != "dummy_hash" for row in claims)
