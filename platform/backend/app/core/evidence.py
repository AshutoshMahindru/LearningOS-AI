"""Evidence claims, provenance hashes, and activity events (WP-138)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
DUMMY_HASH = "dummy_hash"
ASSESSMENT_ACCEPTED = "ACCEPTED"


def canonical_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {"value": row}


def is_real_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    if not digest or digest == DUMMY_HASH or "dummy" in digest:
        return False
    return SHA256_HEX_RE.fullmatch(digest) is not None


def coerce_sha256(value: Any, fallback: str | bytes) -> str:
    """Return a real SHA-256 hex digest; never dummy_hash."""
    if is_real_sha256(value):
        return str(value).strip().lower()
    if is_real_sha256(fallback):
        return str(fallback).strip().lower()
    return sha256_hex(fallback)


def record_activity(
    conn: sqlite3.Connection,
    learner_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> str | None:
    """Append to EventLedger when importable. learner_id is required."""
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("event_type is required")
    body = dict(payload) if isinstance(payload, dict) else {"value": payload}
    try:
        from app.db.ledger import EventLedger
    except ImportError:
        return None
    try:
        return EventLedger(conn).append(learner_id.strip(), event_type, body)
    except Exception:
        return None


def curriculum_identity(
    conn: sqlite3.Connection | None,
    mission_id: str,
    spec: dict[str, Any],
) -> str:
    digest: Any = None
    if conn is not None and mission_id:
        try:
            row = conn.execute(
                """
                SELECT p.git_commit_sha
                FROM missions m
                JOIN curriculum_packages p ON p.id = m.package_id
                WHERE m.id = ?
                """,
                (mission_id,),
            ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None:
            digest = row_dict(row).get("git_commit_sha")
    if is_real_sha256(digest):
        return str(digest).strip().lower()
    if isinstance(digest, str) and digest.strip() and digest.strip().lower() != DUMMY_HASH:
        text = digest.strip()
        if GIT_SHA1_RE.fullmatch(text.lower()):
            return sha256_hex(text.lower())
        return sha256_hex(text)
    return sha256_hex(canonical_dumps(spec))


def _stage_from_spec(spec: dict[str, Any], stage_id: str) -> dict[str, Any] | None:
    stages = spec.get("stages") if isinstance(spec.get("stages"), list) else []
    for stage in stages:
        if isinstance(stage, dict) and str(stage.get("id") or "") == stage_id:
            return stage
    return None


def runner_harness_hash(
    conn: sqlite3.Connection | None,
    attempt_id: str | None,
    spec: dict[str, Any] | None = None,
    stage_id: str | None = None,
) -> str:
    if conn is not None and attempt_id:
        try:
            row = conn.execute(
                """
                SELECT runner_id, code_hash
                FROM executions
                WHERE stage_attempt_id = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (attempt_id,),
            ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None:
            data = row_dict(row)
            runner_id = data.get("runner_id")
            code_hash = data.get("code_hash")
            envelope = {
                "code_hash": code_hash if is_real_sha256(code_hash) else None,
                "runner_id": runner_id if isinstance(runner_id, str) else None,
            }
            if envelope["code_hash"] or envelope["runner_id"]:
                return sha256_hex(canonical_dumps(envelope))
    stage = _stage_from_spec(spec or {}, str(stage_id or "")) if spec and stage_id else None
    runner = stage.get("runner") if isinstance(stage, dict) and isinstance(stage.get("runner"), dict) else None
    if runner:
        return sha256_hex(canonical_dumps(runner))
    return sha256_hex(canonical_dumps({"harness": "unspecified"}))


def knowledge_node_for(
    spec: dict[str, Any],
    *,
    competency_id: str | None = None,
    explicit: Any = None,
) -> str | None:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    nodes = spec.get("knowledge_nodes") if isinstance(spec.get("knowledge_nodes"), list) else []
    comps = spec.get("competencies") if isinstance(spec.get("competencies"), list) else []
    node_ids = [str(node) for node in nodes if isinstance(node, str) and node.strip()]
    if competency_id and isinstance(comps, list) and competency_id in comps:
        index = comps.index(competency_id)
        if index < len(node_ids):
            return node_ids[index]
    if node_ids:
        return node_ids[0]
    return None


def claim_public(claim: dict[str, Any]) -> dict[str, Any]:
    learner_id = claim.get("learner_id")
    attempt_id = claim.get("stage_attempt_id")
    artifact_hash = claim.get("artifact_hash")
    runner_hash = claim.get("runner_hash") or claim.get("harness_hash")
    curriculum_sha = claim.get("curriculum_sha")
    return {
        "id": claim.get("id"),
        "learner_id": learner_id,
        "mission_id": claim.get("mission_id"),
        "stage_id": claim.get("stage_id"),
        "stage_attempt_id": attempt_id,
        "competency_id": claim.get("competency_id"),
        "knowledge_node_id": claim.get("knowledge_node_id"),
        "artifact_type": claim.get("artifact_type"),
        "artifact_path": claim.get("artifact_path"),
        "artifact_hash": artifact_hash,
        "runner_hash": runner_hash,
        "harness_hash": runner_hash,
        "curriculum_sha": curriculum_sha,
        "assessment_status": claim.get("assessment_status") or ASSESSMENT_ACCEPTED,
        "assistance_level": claim.get("assistance_level"),
        "created_at": claim.get("created_at"),
        "provenance": {
            "learner_id": learner_id,
            "stage_attempt_id": attempt_id,
            "artifact_hash": artifact_hash,
            "runner_hash": runner_hash,
            "curriculum_sha": curriculum_sha,
        },
    }


def build_claim(
    *,
    learner_id: str,
    mission_id: str,
    stage_id: str,
    stage_attempt_id: str,
    competency_id: str,
    knowledge_node_id: str,
    artifact_type: str,
    artifact_hash: str,
    runner_hash: str,
    curriculum_sha: str,
    assistance_level: str = "UNASSISTED",
    artifact_path: str | None = None,
    assessment_status: str = ASSESSMENT_ACCEPTED,
    claim_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    material = canonical_dumps(
        {
            "artifact_type": artifact_type,
            "competency_id": competency_id,
            "learner_id": learner_id,
            "mission_id": mission_id,
            "stage_attempt_id": stage_attempt_id,
            "stage_id": stage_id,
        }
    )
    claim = {
        "id": claim_id or str(uuid.uuid4()),
        "learner_id": learner_id.strip(),
        "mission_id": mission_id,
        "stage_id": stage_id,
        "stage_attempt_id": stage_attempt_id,
        "competency_id": competency_id,
        "knowledge_node_id": knowledge_node_id,
        "artifact_type": artifact_type,
        "artifact_path": artifact_path,
        "artifact_hash": coerce_sha256(artifact_hash, material),
        "runner_hash": coerce_sha256(runner_hash, canonical_dumps({"attempt": stage_attempt_id})),
        "curriculum_sha": coerce_sha256(curriculum_sha, canonical_dumps({"mission_id": mission_id})),
        "assessment_status": assessment_status or ASSESSMENT_ACCEPTED,
        "assistance_level": assistance_level or "UNASSISTED",
        "created_at": created_at,
    }
    claim["harness_hash"] = claim["runner_hash"]
    return claim_public(claim)


def insert_claim(conn: sqlite3.Connection, claim: dict[str, Any]) -> dict[str, Any]:
    payload = build_claim(
        learner_id=str(claim.get("learner_id") or ""),
        mission_id=str(claim.get("mission_id") or ""),
        stage_id=str(claim.get("stage_id") or ""),
        stage_attempt_id=str(claim.get("stage_attempt_id") or ""),
        competency_id=str(claim.get("competency_id") or ""),
        knowledge_node_id=str(claim.get("knowledge_node_id") or ""),
        artifact_type=str(claim.get("artifact_type") or "explanation"),
        artifact_hash=str(claim.get("artifact_hash") or ""),
        runner_hash=str(claim.get("runner_hash") or claim.get("harness_hash") or ""),
        curriculum_sha=str(claim.get("curriculum_sha") or ""),
        assistance_level=str(claim.get("assistance_level") or "UNASSISTED"),
        artifact_path=claim.get("artifact_path") if isinstance(claim.get("artifact_path"), str) else None,
        assessment_status=str(claim.get("assessment_status") or ASSESSMENT_ACCEPTED),
        claim_id=str(claim["id"]) if claim.get("id") else None,
        created_at=claim.get("created_at") if isinstance(claim.get("created_at"), str) else None,
    )
    conn.execute(
        """
        INSERT INTO evidence_items (
            id, learner_id, mission_id, stage_id, stage_attempt_id,
            competency_id, knowledge_node_id, artifact_type, artifact_path,
            artifact_hash, assessment_status, assistance_level, curriculum_sha
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["id"],
            payload["learner_id"],
            payload["mission_id"],
            payload["stage_id"],
            payload["stage_attempt_id"],
            payload["competency_id"],
            payload["knowledge_node_id"],
            payload["artifact_type"],
            payload.get("artifact_path"),
            payload["artifact_hash"],
            payload["assessment_status"],
            payload["assistance_level"],
            payload["curriculum_sha"],
        ),
    )
    return payload


def _enrich_row(
    conn: sqlite3.Connection,
    data: dict[str, Any],
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner_hash = runner_harness_hash(
        conn,
        str(data.get("stage_attempt_id") or "") or None,
        spec,
        str(data.get("stage_id") or "") or None,
    )
    data = dict(data)
    data["runner_hash"] = runner_hash
    data["harness_hash"] = runner_hash
    return claim_public(data)


def list_evidence(
    conn: sqlite3.Connection,
    learner_id: str,
    *,
    mission_id: str | None = None,
    spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    clauses = ["learner_id = ?"]
    params: list[Any] = [learner_id.strip()]
    if isinstance(mission_id, str) and mission_id:
        clauses.append("mission_id = ?")
        params.append(mission_id)
    rows = conn.execute(
        f"""
        SELECT * FROM evidence_items
        WHERE {' AND '.join(clauses)}
        ORDER BY rowid ASC
        """,
        params,
    ).fetchall()
    return [_enrich_row(conn, row_dict(row), spec) for row in rows]


def persist_submission_evidence(
    conn: sqlite3.Connection,
    session: dict[str, Any],
    attempt: dict[str, Any],
    stage_id: str,
    spec: dict[str, Any],
    explanation: str | None,
    artifacts: list[dict[str, Any]],
    payload_hash: str,
) -> list[dict[str, Any]]:
    """Persist evidence claims for a stage submission. Does not commit."""
    learner_id = str(session.get("learner_id") or "")
    if not learner_id.strip():
        raise ValueError("learner_id is required")
    mission_id = str(session.get("mission_id") or "")
    attempt_id = str(attempt.get("id") or "")
    assistance = str(attempt.get("assistance_level") or "UNASSISTED")
    contract = spec.get("gate_contract") if isinstance(spec.get("gate_contract"), dict) else {}
    required = contract.get("required_evidence") if isinstance(contract.get("required_evidence"), list) else []
    matching = [
        item
        for item in required
        if isinstance(item, dict) and str(item.get("stage_id") or "") == stage_id
    ]
    artifact_list = [item for item in artifacts if isinstance(item, dict)]
    curriculum_sha = curriculum_identity(conn, mission_id, spec)
    runner_hash = runner_harness_hash(conn, attempt_id, spec, stage_id)
    digest = coerce_sha256(payload_hash, canonical_dumps({"explanation": explanation, "stage_id": stage_id}))

    records: list[dict[str, Any]] = list(matching)
    records.extend(artifact_list)

    seen: set[tuple[str, str, str]] = set()
    claims: list[dict[str, Any]] = []
    for record in records:
        competency_id = record.get("competency_id")
        if not isinstance(competency_id, str) or not competency_id.strip():
            continue
        knowledge_node_id = knowledge_node_for(
            spec,
            competency_id=competency_id,
            explicit=record.get("knowledge_node_id"),
        )
        if not isinstance(knowledge_node_id, str) or not knowledge_node_id:
            continue
        artifact_type = record.get("artifact_type") or record.get("type") or "explanation"
        raw_hash = record.get("artifact_hash") or digest
        artifact_hash = coerce_sha256(raw_hash, digest)
        key = (competency_id, str(artifact_type), artifact_hash)
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            insert_claim(
                conn,
                {
                    "learner_id": learner_id,
                    "mission_id": mission_id,
                    "stage_id": stage_id,
                    "stage_attempt_id": attempt_id,
                    "competency_id": competency_id.strip(),
                    "knowledge_node_id": knowledge_node_id,
                    "artifact_type": str(artifact_type),
                    "artifact_path": record.get("artifact_path"),
                    "artifact_hash": artifact_hash,
                    "runner_hash": runner_hash,
                    "curriculum_sha": curriculum_sha,
                    "assistance_level": assistance,
                },
            )
        )
    return claims
