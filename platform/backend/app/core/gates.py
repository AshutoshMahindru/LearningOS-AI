"""Generic gate_contract evaluation and targeted repair (WP-138 / WP-126)."""

from __future__ import annotations

import sqlite3
from typing import Any

from app.core.evidence import (
    is_real_sha256,
    knowledge_node_for,
    list_evidence,
    record_activity,
    row_dict,
    utc_now,
)

GATE_CONTRACT_ABSENT = "GENERIC_PLACEHOLDER_NO_WP400_ASSESSMENT"
GATE_CONTRACT_INCOMPLETE = "GATE_CONTRACT_INCOMPLETE"
GATE_CRITERIA_MET = "GATE_CRITERIA_MET"
GATE_CRITERIA_UNMET = "GATE_CRITERIA_UNMET"
SESSION_COMPLETED = "COMPLETED"
SESSION_ACTIVE = "ACTIVE"
ATTEMPT_SUBMITTED = "SUBMITTED"


def _criteria_from_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    required = contract.get("required_evidence")
    if not isinstance(required, list):
        return []
    return [item for item in required if isinstance(item, dict)]


def _parse_threshold(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def meets_pass_threshold(passed_criteria: int, required_criteria: int, pass_threshold: float) -> bool:
    if required_criteria <= 0:
        return False
    if 0.0 <= pass_threshold <= 1.0:
        return (passed_criteria / required_criteria) >= pass_threshold
    return passed_criteria >= pass_threshold


def criterion_satisfied(
    criterion: dict[str, Any],
    claims: list[dict[str, Any]],
    submitted_stage_ids: set[str] | None = None,
) -> bool:
    stage_id = criterion.get("stage_id")
    competency_id = criterion.get("competency_id")
    artifact_type = criterion.get("artifact_type")
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        status = str(claim.get("assessment_status") or "ACCEPTED")
        if status not in {"ACCEPTED", "PASSED"}:
            continue
        artifact_hash = claim.get("artifact_hash")
        if artifact_hash is not None and not is_real_sha256(artifact_hash):
            continue
        if isinstance(stage_id, str) and stage_id and str(claim.get("stage_id") or "") != stage_id:
            continue
        if isinstance(competency_id, str) and competency_id and str(claim.get("competency_id") or "") != competency_id:
            continue
        if isinstance(artifact_type, str) and artifact_type and str(claim.get("artifact_type") or "") != artifact_type:
            continue
        return True
    if isinstance(stage_id, str) and stage_id and submitted_stage_ids and stage_id in submitted_stage_ids:
        return True
    return False


def _repair_item(spec: dict[str, Any], criterion: dict[str, Any]) -> dict[str, Any]:
    competency_id = criterion.get("competency_id")
    node = knowledge_node_for(
        spec,
        competency_id=str(competency_id) if isinstance(competency_id, str) else None,
        explicit=criterion.get("knowledge_node_id"),
    )
    item: dict[str, Any] = {
        "action": "targeted_repair",
        "stage_id": criterion.get("stage_id"),
        "artifact_type": criterion.get("artifact_type"),
        "competency_id": competency_id,
    }
    if node:
        item["knowledge_node_id"] = node
    return item


def build_repair_plan(spec: dict[str, Any], failed: list[dict[str, Any]]) -> dict[str, Any]:
    drills = [_repair_item(spec, item) for item in failed]
    nodes: list[str] = []
    seen: set[str] = set()
    for drill in drills:
        node = drill.get("knowledge_node_id")
        if isinstance(node, str) and node and node not in seen:
            seen.add(node)
            nodes.append(node)
    stages = [
        str(item.get("stage_id"))
        for item in drills
        if isinstance(item.get("stage_id"), str) and item.get("stage_id")
    ]
    return {
        "failed_knowledge_nodes": nodes,
        "failed_evidence": [
            {
                key: item.get(key)
                for key in ("stage_id", "artifact_type", "competency_id", "knowledge_node_id")
                if key in item and item.get(key) is not None
            }
            for item in failed
        ],
        "target_stage_ids": stages,
        "drills": drills,
    }


def evaluate_contract(
    contract: dict[str, Any] | None,
    claims: list[dict[str, Any]],
    *,
    spec: dict[str, Any] | None = None,
    submitted_stage_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Pure gate_contract evaluation. No mission-id branches."""
    spec = spec if isinstance(spec, dict) else {}
    if not isinstance(contract, dict):
        return {
            "status": "EVALUATED",
            "reason": GATE_CONTRACT_ABSENT,
            "passed_criteria": 0,
            "required_criteria": 0,
            "pass_threshold": None,
            "failed_evidence": [],
            "competency_increments": [],
        }

    criteria = _criteria_from_contract(contract)
    threshold = _parse_threshold(contract.get("pass_threshold"))
    if not criteria or threshold is None:
        return {
            "status": "EVALUATED",
            "reason": GATE_CONTRACT_INCOMPLETE,
            "passed_criteria": 0,
            "required_criteria": len(criteria),
            "pass_threshold": threshold,
            "failed_evidence": [],
            "competency_increments": [],
        }

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in criteria:
        if criterion_satisfied(item, claims, submitted_stage_ids):
            passed.append(item)
        else:
            failed.append(item)

    required_criteria = len(passed) + len(failed)
    passed_criteria = len(passed)
    repair_policy = contract.get("repair_policy") if isinstance(contract.get("repair_policy"), dict) else {}
    allow_repair = True
    if repair_policy:
        allow_repair = bool(repair_policy.get("allow_targeted_repair", True))

    increments: list[dict[str, Any]] = []
    seen_comps: set[str] = set()
    result: dict[str, Any] = {
        "passed_criteria": passed_criteria,
        "required_criteria": required_criteria,
        "pass_threshold": threshold,
        "score": (passed_criteria / required_criteria) if required_criteria else 0.0,
        "failed_evidence": [
            {
                key: item.get(key)
                for key in ("stage_id", "artifact_type", "competency_id", "knowledge_node_id")
                if key in item
            }
            for item in failed
        ],
        "competency_increments": increments,
    }

    if meets_pass_threshold(passed_criteria, required_criteria, threshold):
        for item in passed:
            competency_id = item.get("competency_id")
            if not isinstance(competency_id, str) or not competency_id.strip():
                continue
            cid = competency_id.strip()
            if cid in seen_comps:
                continue
            seen_comps.add(cid)
            increments.append({"competency_id": cid, "delta": 1})
        result["status"] = "PASSED"
        result["reason"] = GATE_CRITERIA_MET
        result["competency_increments"] = increments
        return result

    if allow_repair:
        result["status"] = "REPAIR_REQUIRED"
    else:
        result["status"] = "FAILED"
    result["reason"] = GATE_CRITERIA_UNMET
    result["repair_plan"] = build_repair_plan(spec, failed)
    return result


def _submitted_stage_ids(conn: sqlite3.Connection, session_id: str) -> set[str]:
    try:
        rows = conn.execute(
            """
            SELECT stage_id FROM stage_attempts
            WHERE session_id = ? AND status = ?
            """,
            (session_id, ATTEMPT_SUBMITTED),
        ).fetchall()
    except sqlite3.Error:
        return set()
    ids: set[str] = set()
    for row in rows:
        stage_id = row_dict(row).get("stage_id")
        if isinstance(stage_id, str) and stage_id:
            ids.add(stage_id)
    return ids


def _latest_evidence_id(conn: sqlite3.Connection, learner_id: str, competency_id: str) -> str | None:
    try:
        row = conn.execute(
            """
            SELECT id FROM evidence_items
            WHERE learner_id = ? AND competency_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (learner_id, competency_id),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    value = row_dict(row).get("id")
    return str(value) if value else None


def increment_competencies(
    conn: sqlite3.Connection,
    learner_id: str,
    increments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    now = utc_now()
    for item in increments:
        competency_id = item.get("competency_id")
        if not isinstance(competency_id, str) or not competency_id.strip():
            continue
        cid = competency_id.strip()
        delta = int(item.get("delta") or 1)
        evidence_id = _latest_evidence_id(conn, learner_id, cid)
        existing = conn.execute(
            """
            SELECT level FROM competency_mastery
            WHERE learner_id = ? AND competency_id = ?
            """,
            (learner_id, cid),
        ).fetchone()
        if existing is None:
            level = min(5, max(1, delta))
            conn.execute(
                """
                INSERT INTO competency_mastery (
                    learner_id, competency_id, level, decay_score,
                    last_evaluated_at, last_evidence_item_id
                ) VALUES (?, ?, ?, 1.0, ?, ?)
                """,
                (learner_id, cid, level, now, evidence_id),
            )
        else:
            current = int(row_dict(existing).get("level") or 0)
            level = min(5, current + max(1, delta))
            conn.execute(
                """
                UPDATE competency_mastery
                SET level = ?, decay_score = 1.0, last_evaluated_at = ?, last_evidence_item_id = ?
                WHERE learner_id = ? AND competency_id = ?
                """,
                (level, now, evidence_id, learner_id, cid),
            )
        applied.append({"competency_id": cid, "level": level, "delta": delta})
    return applied


def evaluate_session_gate(
    conn: sqlite3.Connection,
    session: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    session_id = str(session.get("session_id") or session.get("id") or "")
    learner_id = str(session.get("learner_id") or "")
    mission_id = str(session.get("mission_id") or "")
    contract = spec.get("gate_contract") if isinstance(spec.get("gate_contract"), dict) else None
    claims = list_evidence(conn, learner_id, mission_id=mission_id, spec=spec) if learner_id else []
    submitted = _submitted_stage_ids(conn, session_id) if session_id else set()
    result = evaluate_contract(
        contract,
        claims,
        spec=spec,
        submitted_stage_ids=submitted,
    )
    result["session_id"] = session_id
    session_status = str(session.get("status") or SESSION_ACTIVE)

    if result.get("status") == "PASSED" and learner_id:
        applied = increment_competencies(conn, learner_id, result.get("competency_increments") or [])
        result["competency_increments"] = applied
        conn.execute(
            "UPDATE mission_sessions SET status = ?, completed_at = ? WHERE id = ?",
            (SESSION_COMPLETED, utc_now(), session_id),
        )
        conn.commit()
        session_status = SESSION_COMPLETED
    else:
        try:
            conn.commit()
        except sqlite3.Error:
            pass

    result["session_status"] = session_status
    if learner_id:
        record_activity(
            conn,
            learner_id,
            "gate_evaluated",
            {
                "session_id": session_id,
                "mission_id": mission_id,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "passed_criteria": result.get("passed_criteria"),
                "required_criteria": result.get("required_criteria"),
            },
        )
    return result
