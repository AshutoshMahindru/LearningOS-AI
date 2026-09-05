"""Competency projection and generic Today / next-action sequencing."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.core.evidence import row_dict

SESSION_ACTIVE = "ACTIVE"
SESSION_COMPLETED = "COMPLETED"
ATTEMPT_READY = "READY"
ATTEMPT_ACTIVE = "ACTIVE"
ATTEMPT_SUBMITTED = "SUBMITTED"
UNSTARTED = frozenset({"", "start"})


def _spec_from_value(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def _stage_ids(spec: dict[str, Any]) -> list[str]:
    stages = spec.get("stages") if isinstance(spec.get("stages"), list) else []
    ids: list[str] = []
    for stage in stages:
        if isinstance(stage, dict) and stage.get("id"):
            ids.append(str(stage["id"]))
    return ids


def _stage_type(spec: dict[str, Any], stage_id: str) -> str:
    stages = spec.get("stages") if isinstance(spec.get("stages"), list) else []
    for stage in stages:
        if isinstance(stage, dict) and str(stage.get("id") or "") == stage_id:
            raw = stage.get("type")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return "unspecified"


def project_competencies(conn: sqlite3.Connection, learner_id: str) -> list[dict[str, Any]]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    rows = conn.execute(
        """
        SELECT competency_id, level, decay_score, last_evaluated_at, last_evidence_item_id
        FROM competency_mastery
        WHERE learner_id = ?
        ORDER BY competency_id
        """,
        (learner_id.strip(),),
    ).fetchall()
    projected: list[dict[str, Any]] = []
    for row in rows:
        data = row_dict(row)
        projected.append(
            {
                "competency_id": data.get("competency_id"),
                "level": int(data.get("level") or 0),
                "decay_score": float(data.get("decay_score") or 0.0),
                "last_evaluated_at": data.get("last_evaluated_at"),
                "last_evidence_item_id": data.get("last_evidence_item_id"),
            }
        )
    return projected


def _missions_ordered(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT id, title, order_index, spec_json
            FROM missions
            ORDER BY order_index ASC, id ASC
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    missions: list[dict[str, Any]] = []
    for row in rows:
        data = row_dict(row)
        spec = _spec_from_value(data.get("spec_json"))
        missions.append(
            {
                "id": data.get("id"),
                "title": data.get("title"),
                "order_index": data.get("order_index"),
                "spec": spec,
            }
        )
    return missions


def _sessions_for_learner(conn: sqlite3.Connection, learner_id: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT * FROM mission_sessions
            WHERE learner_id = ?
            ORDER BY rowid DESC
            """,
            (learner_id,),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [row_dict(row) for row in rows]


def _latest_attempt(conn: sqlite3.Connection, session_id: str, stage_id: str) -> dict[str, Any] | None:
    try:
        row = conn.execute(
            """
            SELECT * FROM stage_attempts
            WHERE session_id = ? AND stage_id = ?
            ORDER BY attempt_number DESC, rowid DESC
            LIMIT 1
            """,
            (session_id, stage_id),
        ).fetchone()
    except sqlite3.Error:
        return None
    return row_dict(row) if row is not None else None


def _action_payload(
    *,
    learner_id: str,
    action: str,
    reason: str,
    mission_id: str | None = None,
    session_id: str | None = None,
    stage_id: str | None = None,
    competencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    today = {
        "action": action,
        "reason": reason,
        "mission_id": mission_id,
        "session_id": session_id,
        "stage_id": stage_id,
    }
    return {
        "learner_id": learner_id,
        "action": action,
        "reason": reason,
        "mission_id": mission_id,
        "session_id": session_id,
        "stage_id": stage_id,
        "today": today,
        "competencies": competencies or [],
    }


def _flagship_mod() -> Any | None:
    try:
        from app.core import flagship as flagship_mod
    except ImportError:
        return None
    return flagship_mod


def _finalize_action(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
    completed_missions: set[str],
    catalog_ids: set[str],
) -> dict[str, Any]:
    flagship_mod = _flagship_mod()
    if flagship_mod is None:
        return payload
    return flagship_mod.annotate_next_action(
        payload,
        conn=conn,
        completed_missions=completed_missions,
        catalog_ids=catalog_ids,
    )


def next_action(conn: sqlite3.Connection, learner_id: str) -> dict[str, Any]:
    """Next stage or next mission from catalog sequencing. No mission-id special case."""
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    lid = learner_id.strip()
    competencies = project_competencies(conn, lid)
    missions = _missions_ordered(conn)
    sessions = _sessions_for_learner(conn, lid)
    completed_missions = {
        str(item.get("mission_id"))
        for item in sessions
        if str(item.get("status") or "") == SESSION_COMPLETED and item.get("mission_id")
    }
    catalog_ids = {str(item.get("id")) for item in missions if item.get("id")}
    flagship_mod = _flagship_mod()

    active = [
        item
        for item in sessions
        if str(item.get("status") or SESSION_ACTIVE) == SESSION_ACTIVE
    ]
    if active:
        session = active[0]
        session_id = str(session.get("id") or session.get("session_id") or "")
        mission_id = str(session.get("mission_id") or "")
        spec = {}
        for mission in missions:
            if str(mission.get("id")) == mission_id:
                spec = mission.get("spec") or {}
                break
        if not spec and mission_id:
            try:
                row = conn.execute(
                    "SELECT spec_json FROM missions WHERE id = ?",
                    (mission_id,),
                ).fetchone()
            except sqlite3.Error:
                row = None
            if row is not None:
                spec = _spec_from_value(row_dict(row).get("spec_json"))
        ids = _stage_ids(spec)
        current = str(session.get("current_stage_id") or "")
        if current.strip().lower() in UNSTARTED:
            current = ids[0] if ids else ""
        attempt = _latest_attempt(conn, session_id, current) if current else None
        attempt_status = str(attempt.get("status") or "") if attempt else ""
        if current and attempt_status == ATTEMPT_SUBMITTED:
            try:
                index = ids.index(current)
            except ValueError:
                index = -1
            if index >= 0 and index + 1 < len(ids):
                nxt = ids[index + 1]
                return _finalize_action(
                    conn,
                    _action_payload(
                        learner_id=lid,
                        action="ENTER_STAGE",
                        reason="NEXT_STAGE",
                        mission_id=mission_id,
                        session_id=session_id,
                        stage_id=nxt,
                        competencies=competencies,
                    ),
                    completed_missions,
                    catalog_ids,
                )
            return _finalize_action(
                conn,
                _action_payload(
                    learner_id=lid,
                    action="EVALUATE_GATE",
                    reason="GATE_PENDING",
                    mission_id=mission_id,
                    session_id=session_id,
                    stage_id=current,
                    competencies=competencies,
                ),
                completed_missions,
                catalog_ids,
            )
        if current and attempt_status == ATTEMPT_ACTIVE:
            action = "CONTINUE_STAGE"
            reason = "STAGE_IN_PROGRESS"
        else:
            action = "ENTER_STAGE"
            reason = "CURRENT_SESSION_STAGE"
        return _finalize_action(
            conn,
            _action_payload(
                learner_id=lid,
                action=action,
                reason=reason,
                mission_id=mission_id,
                session_id=session_id,
                stage_id=current or None,
                competencies=competencies,
            ),
            completed_missions,
            catalog_ids,
        )

    blocked = False
    for mission in missions:
        mission_id = str(mission.get("id") or "")
        if not mission_id or mission_id in completed_missions:
            continue
        if flagship_mod is not None and not flagship_mod.mission_unlocked(
            mission_id, completed_missions, catalog_ids
        ):
            blocked = True
            continue
        spec = mission.get("spec") or {}
        ids = _stage_ids(spec)
        return _finalize_action(
            conn,
            _action_payload(
                learner_id=lid,
                action="START_MISSION",
                reason="NEXT_MISSION",
                mission_id=mission_id,
                stage_id=ids[0] if ids else None,
                competencies=competencies,
            ),
            completed_missions,
            catalog_ids,
        )

    if blocked:
        return _finalize_action(
            conn,
            _action_payload(
                learner_id=lid,
                action="IDLE",
                reason="PREREQUISITES_UNMET",
                competencies=competencies,
            ),
            completed_missions,
            catalog_ids,
        )
    if missions:
        return _finalize_action(
            conn,
            _action_payload(
                learner_id=lid,
                action="IDLE",
                reason="ALL_MISSIONS_COMPLETE",
                competencies=competencies,
            ),
            completed_missions,
            catalog_ids,
        )
    return _finalize_action(
        conn,
        _action_payload(
            learner_id=lid,
            action="IDLE",
            reason="NO_AVAILABLE_MISSIONS",
            competencies=competencies,
        ),
        completed_missions,
        catalog_ids,
    )
