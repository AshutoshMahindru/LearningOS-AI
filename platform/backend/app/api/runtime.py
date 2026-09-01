"""Generic session/stage runtime. Sequencing comes from mission spec_json only."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.errors import (
    AppError,
    ConflictError,
    InternalError,
    NotFoundError,
    ValidationAppError,
)
from app.core.gates import GATE_CONTRACT_ABSENT as GENERIC_GATE_PLACEHOLDER

SESSION_ACTIVE = "ACTIVE"
SESSION_COMPLETED = "COMPLETED"
ATTEMPT_READY = "READY"
ATTEMPT_ACTIVE = "ACTIVE"
ATTEMPT_SUBMITTED = "SUBMITTED"
EXPERIMENT_STAGE_TYPE = "experiment"
UNSTARTED_STAGE_SENTINELS = frozenset({"", "start"})
PREDICTION_SENSITIVE_TYPES = frozenset({EXPERIMENT_STAGE_TYPE})
WP137_STATUSES = frozenset({"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"})

_ASSISTANCE_LEVELS = {
    "NO_AI_REQUIRED": "NO_AI_CERTIFIED",
    "SOCRATIC_ONLY": "SOCRATIC",
    "RESTRICTED_HINTS": "SOCRATIC",
    "UNRESTRICTED": "UNASSISTED",
}


def canonical_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_dumps(value))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _row(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {"value": row}


def validation_error_details(exc: BaseException) -> dict[str, Any]:
    details: dict[str, Any] = dict(getattr(exc, "details", None) or {})
    path = getattr(exc, "path", None)
    if isinstance(path, str) and path:
        details.setdefault("path", path)
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        details.setdefault("code", code)
    errors = getattr(exc, "errors", None)
    if isinstance(errors, list) and errors:
        details.setdefault("errors", errors)
    return details


def _maybe_validate_spec(spec: dict[str, Any]) -> None:
    """Validate complete M## specs via 21A; skip G3/legacy and incomplete payloads."""
    try:
        from app.core import mdl_validator
        from app.core.mdl_types import MISSION_ID_PATTERN, MISSION_REQUIRED_FIELDS
    except ImportError:
        return
    mission_id = str(spec.get("id") or "")
    if not re.fullmatch(MISSION_ID_PATTERN, mission_id):
        return
    if any(field not in spec for field in MISSION_REQUIRED_FIELDS):
        return
    validator = None
    for name in ("validate_mission", "validate_spec", "validate"):
        candidate = getattr(mdl_validator, name, None)
        if callable(candidate):
            validator = candidate
            break
    if validator is None:
        return
    try:
        result = validator(spec)
    except ImportError:
        return
    except AppError:
        raise
    except (ValueError, TypeError) as exc:
        raise ValidationAppError(
            str(exc) or "Mission spec failed validation",
            details=validation_error_details(exc),
        ) from exc
    except Exception:
        return
    if result is False:
        raise ValidationAppError("Mission spec failed validation")


def spec_from_value(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        spec = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationAppError("Mission spec_json is not valid JSON") from exc
        if not isinstance(loaded, dict):
            raise ValidationAppError("Mission spec_json must be an object")
        spec = loaded
    else:
        raise ValidationAppError("Mission spec_json is missing")
    _maybe_validate_spec(spec)
    return spec


def load_mission_spec(conn: sqlite3.Connection, mission_id: str) -> dict[str, Any]:
    try:
        row = conn.execute(
            "SELECT spec_json FROM missions WHERE id = ?",
            (mission_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        raise InternalError("Failed to load mission spec", status_code=503) from exc
    if row is None:
        raise NotFoundError("Mission not found", details={"mission_id": mission_id})
    data = _row(row)
    return spec_from_value(data.get("spec_json"))


def stage_list(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw = spec.get("stages")
    if not isinstance(raw, list) or not raw:
        raise ValidationAppError("Mission spec must declare a stages list")
    stages: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("id"):
            stages.append(item)
    if not stages:
        raise ValidationAppError("Mission spec stages are missing ids")
    return stages


def first_stage_id(spec: dict[str, Any]) -> str:
    return str(stage_list(spec)[0]["id"])


def get_stage(spec: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in stage_list(spec):
        if str(stage.get("id")) == stage_id:
            return stage
    raise NotFoundError("Stage not found in mission spec", details={"stage_id": stage_id})


def next_stage_id(spec: dict[str, Any], stage_id: str) -> str | None:
    ids = [str(stage["id"]) for stage in stage_list(spec)]
    try:
        index = ids.index(stage_id)
    except ValueError:
        return None
    if index + 1 >= len(ids):
        return None
    return ids[index + 1]


def stage_type_of(stage: dict[str, Any]) -> str:
    raw = stage.get("type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "unspecified"


def assistance_level_of(stage: dict[str, Any]) -> str:
    policy = str(stage.get("assistance_policy") or "").strip().upper()
    return _ASSISTANCE_LEVELS.get(policy, "UNASSISTED")


def is_prediction_sensitive(stage: dict[str, Any]) -> bool:
    return stage_type_of(stage).lower() in PREDICTION_SENSITIVE_TYPES


def load_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    try:
        row = conn.execute(
            "SELECT * FROM mission_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        raise InternalError("Failed to load session", status_code=503) from exc
    if row is None:
        raise NotFoundError("Session not found", details={"session_id": session_id})
    data = _row(row)
    data["session_id"] = data.get("id") or session_id
    return data


def _require_session_mutable(session: dict[str, Any]) -> None:
    status = str(session.get("status") or SESSION_ACTIVE)
    if status == SESSION_COMPLETED:
        raise ConflictError(
            "Session is already completed",
            details={"session_id": session.get("session_id"), "status": status},
        )


def _unstarted(current_stage_id: Any) -> bool:
    current = "" if current_stage_id is None else str(current_stage_id).strip()
    return current in UNSTARTED_STAGE_SENTINELS


def assert_stage_is_current(session: dict[str, Any], spec: dict[str, Any], stage_id: str) -> dict[str, Any]:
    stage = get_stage(spec, stage_id)
    ids = [str(item["id"]) for item in stage_list(spec)]
    current = str(session.get("current_stage_id") or "")
    if _unstarted(current):
        expected = ids[0]
    else:
        expected = current
    if stage_id != expected:
        raise ConflictError(
            "Stage is not the current session stage",
            details={
                "stage_id": stage_id,
                "current_stage_id": expected,
            },
        )
    return stage


def latest_attempt(
    conn: sqlite3.Connection, session_id: str, stage_id: str
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT * FROM stage_attempts
        WHERE session_id = ? AND stage_id = ?
        ORDER BY attempt_number DESC, rowid DESC
        LIMIT 1
        """,
        (session_id, stage_id),
    ).fetchone()
    return _row(row) if row is not None else None


def _next_attempt_number(conn: sqlite3.Connection, session_id: str, stage_id: str) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(attempt_number), 0) AS n
        FROM stage_attempts
        WHERE session_id = ? AND stage_id = ?
        """,
        (session_id, stage_id),
    ).fetchone()
    current = int(_row(row).get("n") or 0)
    return current + 1


def insert_attempt(
    conn: sqlite3.Connection,
    session_id: str,
    stage: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    attempt_id = str(uuid.uuid4())
    stage_id = str(stage["id"])
    attempt_number = _next_attempt_number(conn, session_id, stage_id)
    conn.execute(
        """
        INSERT INTO stage_attempts (
            id, session_id, stage_id, stage_type, attempt_number, status, assistance_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            attempt_id,
            session_id,
            stage_id,
            stage_type_of(stage),
            attempt_number,
            status,
            assistance_level_of(stage),
        ),
    )
    row = conn.execute("SELECT * FROM stage_attempts WHERE id = ?", (attempt_id,)).fetchone()
    return _row(row)


def ensure_ready_attempt(
    conn: sqlite3.Connection, session_id: str, stage: dict[str, Any]
) -> dict[str, Any]:
    existing = latest_attempt(conn, session_id, str(stage["id"]))
    if existing is not None:
        return existing
    return insert_attempt(conn, session_id, stage, ATTEMPT_READY)


def _stage_public(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(stage.get("id")),
        "title": stage.get("title"),
        "type": stage_type_of(stage),
        "assistance_policy": stage.get("assistance_policy"),
    }


def _attempt_public(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    if not attempt:
        return None
    return {
        "attempt_id": attempt.get("id"),
        "stage_id": attempt.get("stage_id"),
        "stage_type": attempt.get("stage_type"),
        "attempt_number": attempt.get("attempt_number"),
        "status": attempt.get("status"),
    }


def enrich_session(conn: sqlite3.Connection, session: dict[str, Any]) -> dict[str, Any]:
    data = dict(session)
    data["session_id"] = data.get("session_id") or data.get("id")
    current_id = data.get("current_stage_id")
    spec: dict[str, Any] | None = None
    mission_id = data.get("mission_id")
    if mission_id:
        try:
            spec = load_mission_spec(conn, str(mission_id))
        except AppError:
            spec = None
    if spec is not None and current_id and not _unstarted(current_id):
        try:
            stage = get_stage(spec, str(current_id))
            data["current_stage"] = _stage_public(stage)
        except AppError:
            data["current_stage"] = {"id": current_id}
        data["current_attempt"] = _attempt_public(latest_attempt(conn, str(data["session_id"]), str(current_id)))
    else:
        data["current_stage"] = None
        data["current_attempt"] = None
    return data


def _require_attempt(
    conn: sqlite3.Connection, session_id: str, stage_id: str
) -> dict[str, Any]:
    attempt = latest_attempt(conn, session_id, stage_id)
    if attempt is None:
        raise ConflictError(
            "Stage has not been entered",
            details={"session_id": session_id, "stage_id": stage_id},
        )
    return attempt


def _latest_prediction(conn: sqlite3.Connection, attempt_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT * FROM predictions
        WHERE stage_attempt_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (attempt_id,),
    ).fetchone()
    return _row(row) if row is not None else None


def _prediction_is_sealed(prediction: dict[str, Any] | None) -> bool:
    if not prediction:
        return False
    sealed = prediction.get("is_sealed")
    if sealed in (0, "0", False, "false", "FALSE"):
        return False
    return bool(prediction.get("prediction_hash"))


def enter_stage(conn: sqlite3.Connection, session_id: str, stage_id: str) -> dict[str, Any]:
    session = load_session(conn, session_id)
    _require_session_mutable(session)
    spec = load_mission_spec(conn, str(session["mission_id"]))
    stage = assert_stage_is_current(session, spec, stage_id)
    current_id = str(stage["id"])
    if _unstarted(session.get("current_stage_id")):
        conn.execute(
            "UPDATE mission_sessions SET current_stage_id = ?, status = ? WHERE id = ?",
            (current_id, SESSION_ACTIVE, session_id),
        )
        session["current_stage_id"] = current_id
        session["status"] = SESSION_ACTIVE

    attempt = latest_attempt(conn, session_id, current_id)
    resumed = False
    if attempt is None:
        attempt = insert_attempt(conn, session_id, stage, ATTEMPT_ACTIVE)
    elif str(attempt.get("status") or "") == ATTEMPT_READY:
        conn.execute(
            "UPDATE stage_attempts SET status = ? WHERE id = ?",
            (ATTEMPT_ACTIVE, attempt["id"]),
        )
        attempt["status"] = ATTEMPT_ACTIVE
    else:
        resumed = True

    conn.commit()
    return {
        "session_id": session_id,
        "stage_id": current_id,
        "stage_type": stage_type_of(stage),
        "attempt_id": attempt["id"],
        "attempt_number": attempt.get("attempt_number"),
        "status": attempt.get("status"),
        "session_status": session.get("status") or SESSION_ACTIVE,
        "current_stage_id": current_id,
        "resumed": resumed,
    }


def commit_prediction(
    conn: sqlite3.Connection,
    session_id: str,
    stage_id: str,
    hypothesis: str,
    expected_values: dict[str, Any],
) -> dict[str, Any]:
    session = load_session(conn, session_id)
    _require_session_mutable(session)
    spec = load_mission_spec(conn, str(session["mission_id"]))
    assert_stage_is_current(session, spec, stage_id)
    attempt = _require_attempt(conn, session_id, stage_id)
    if str(attempt.get("status") or "") != ATTEMPT_ACTIVE:
        raise ConflictError(
            "Predictions can only be committed on an ACTIVE stage attempt",
            details={"attempt_status": attempt.get("status")},
        )

    envelope = {"expected_values": expected_values, "hypothesis": hypothesis}
    prediction_hash = canonical_sha256(envelope)
    existing = _latest_prediction(conn, str(attempt["id"]))
    if existing is not None and _prediction_is_sealed(existing):
        if str(existing.get("prediction_hash")) == prediction_hash:
            committed_at = existing.get("committed_at")
            return {
                "prediction_id": existing.get("id"),
                "stage_attempt_id": attempt["id"],
                "session_id": session_id,
                "stage_id": stage_id,
                "hypothesis": hypothesis,
                "expected_values": expected_values,
                "prediction_hash": prediction_hash,
                "is_sealed": True,
                "committed_at": committed_at,
                "resumed": True,
            }
        raise ConflictError(
            "A sealed prediction already exists for this attempt",
            details={"prediction_id": existing.get("id")},
        )

    prediction_id = str(uuid.uuid4())
    committed_at = _now()
    conn.execute(
        """
        INSERT INTO predictions (
            id, stage_attempt_id, hypothesis_text, expected_values_json,
            prediction_hash, committed_at, is_sealed
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            attempt["id"],
            hypothesis,
            canonical_dumps(expected_values),
            prediction_hash,
            committed_at,
            1,
        ),
    )
    conn.commit()
    return {
        "prediction_id": prediction_id,
        "stage_attempt_id": attempt["id"],
        "session_id": session_id,
        "stage_id": stage_id,
        "hypothesis": hypothesis,
        "expected_values": expected_values,
        "prediction_hash": prediction_hash,
        "is_sealed": True,
        "committed_at": committed_at,
        "resumed": False,
    }


def _call_worker_execute(code: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
    try:
        from app.core.worker_client import WorkerClient
    except ImportError:
        return {
            "status": "UNSUPPORTED",
            "reason": "WorkerClient is not available; execution sandbox is WP400",
        }
    try:
        client = WorkerClient()
        result = client.execute(code, parameters)
    except Exception as exc:
        return {
            "status": "WORKER_UNAVAILABLE",
            "reason": "Worker execute failed",
            "error": {
                "code": "WORKER_UNAVAILABLE",
                "message": str(exc) or "Worker execute failed",
                "details": {},
            },
        }
    if isinstance(result, dict):
        return result
    return {"status": "UNSUPPORTED", "result": result}


def _execution_status(worker_result: dict[str, Any]) -> str:
    status = worker_result.get("status")
    if isinstance(status, str) and status:
        return status
    error = worker_result.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
        return "WORKER_UNAVAILABLE"
    return "UNSUPPORTED"


def _execution_exit_code(worker_result: dict[str, Any], status: str) -> int:
    raw = worker_result.get("exit_code")
    if isinstance(raw, int):
        return raw
    if status in {"SUCCESS", "ACCEPTED", "UNSUPPORTED"}:
        return 0
    if status in {"WORKER_UNAVAILABLE", "TIMEOUT", "CRASHED"}:
        return -1
    return 1


def _wp137_status(status: str) -> str:
    if status in WP137_STATUSES:
        return status
    if status == "CANCELLED":
        return "TIMEOUT"
    if status in {"WORKER_UNAVAILABLE"}:
        return "CRASHED"
    return "FAILED"


def _structured_from_worker(
    worker_result: dict[str, Any],
    *,
    execution_id: str,
    duration_ms: int,
    status: str,
    exit_code: int,
) -> dict[str, Any]:
    """Map 31B sandbox / 31A runner output onto the WP-137 envelope."""
    raw = worker_result if isinstance(worker_result, dict) else {}
    exec_id = raw.get("execution_id")
    if not isinstance(exec_id, str) or not exec_id.strip():
        exec_id = execution_id
    dur = raw.get("duration_ms")
    if not isinstance(dur, int) or dur < 0:
        dur = duration_ms
    ex = raw.get("exit_code")
    if not isinstance(ex, int):
        ex = exit_code
    blocks = raw.get("blocks") if isinstance(raw.get("blocks"), list) else []
    diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
    stdout = diagnostics.get("stdout") if isinstance(diagnostics.get("stdout"), str) else None
    if stdout is None:
        stdout = str(raw.get("stdout") or "")
    stderr = diagnostics.get("stderr") if isinstance(diagnostics.get("stderr"), str) else None
    if stderr is None:
        stderr = str(raw.get("stderr") or "")
    diag: dict[str, Any] = {"stdout": stdout, "stderr": stderr}
    metrics = diagnostics.get("system_metrics")
    if isinstance(metrics, dict) and metrics:
        diag["system_metrics"] = metrics
    structured: dict[str, Any] = {
        "execution_id": exec_id,
        "status": _wp137_status(str(raw.get("status") or status)),
        "exit_code": ex,
        "duration_ms": dur,
        "blocks": blocks,
        "diagnostics": diag,
    }
    reproducibility = raw.get("reproducibility")
    if isinstance(reproducibility, dict):
        structured["reproducibility"] = reproducibility
    return structured


def _runner_parameters(
    runner: dict[str, Any],
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    params = dict(parameters or {})
    if "limits" in params:
        return params
    limits: dict[str, Any] = {}
    if runner.get("timeout_sec") is not None:
        limits["timeout_sec"] = runner.get("timeout_sec")
    if runner.get("memory_mb") is not None:
        limits["memory_mb"] = runner.get("memory_mb")
    if limits:
        params["limits"] = limits
    return params


def execute_stage(
    conn: sqlite3.Connection,
    session_id: str,
    stage_id: str,
    code: str | None,
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    session = load_session(conn, session_id)
    _require_session_mutable(session)
    spec = load_mission_spec(conn, str(session["mission_id"]))
    stage = assert_stage_is_current(session, spec, stage_id)
    attempt = _require_attempt(conn, session_id, stage_id)
    if str(attempt.get("status") or "") != ATTEMPT_ACTIVE:
        raise ConflictError(
            "Execute requires an ACTIVE stage attempt",
            details={"attempt_status": attempt.get("status")},
        )
    if is_prediction_sensitive(stage):
        sealed = _latest_prediction(conn, str(attempt["id"]))
        if not _prediction_is_sealed(sealed):
            raise ConflictError(
                "Experiment stage requires a sealed prediction before execute",
                details={
                    "reason": "PREDICTION_REQUIRED",
                    "stage_id": stage_id,
                    "stage_type": stage_type_of(stage),
                },
            )

    source = code or ""
    code_hash = sha256_hex(source)
    runner = stage.get("runner") if isinstance(stage.get("runner"), dict) else {}
    runner_id = str(runner.get("module") or "worker")
    started = time.perf_counter()
    worker_result = _call_worker_execute(source, _runner_parameters(runner, parameters))
    duration_ms = max(0, int((time.perf_counter() - started) * 1000))
    status = _execution_status(worker_result)
    exit_code = _execution_exit_code(worker_result, status)
    execution_id = str(uuid.uuid4())
    structured = _structured_from_worker(
        worker_result,
        execution_id=execution_id,
        duration_ms=duration_ms,
        status=status,
        exit_code=exit_code,
    )
    execution_id = str(structured["execution_id"])
    exit_code = int(structured["exit_code"])
    duration_ms = int(structured["duration_ms"])
    diagnostics = structured.get("diagnostics") if isinstance(structured.get("diagnostics"), dict) else None
    conn.execute(
        """
        INSERT INTO executions (
            id, stage_attempt_id, runner_id, input_code, code_hash,
            exit_code, duration_ms, structured_result_json, diagnostics_log
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            execution_id,
            attempt["id"],
            runner_id,
            source,
            code_hash,
            exit_code,
            duration_ms,
            canonical_dumps(structured),
            canonical_dumps(diagnostics) if diagnostics is not None else (
                canonical_dumps(worker_result.get("error"))
                if isinstance(worker_result.get("error"), dict)
                else None
            ),
        ),
    )
    conn.commit()
    return {
        "execution_id": execution_id,
        "stage_attempt_id": attempt["id"],
        "session_id": session_id,
        "stage_id": stage_id,
        "status": status,
        "code_hash": code_hash,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "runner_id": runner_id,
        "structured_result": structured,
        "blocks": structured.get("blocks") if isinstance(structured.get("blocks"), list) else [],
        "diagnostics": diagnostics or {},
    }


def _persist_submission_evidence(
    conn: sqlite3.Connection,
    session: dict[str, Any],
    attempt: dict[str, Any],
    stage_id: str,
    spec: dict[str, Any],
    explanation: str | None,
    artifacts: list[dict[str, Any]],
    payload_hash: str,
) -> list[dict[str, Any]]:
    from app.core.evidence import persist_submission_evidence

    return persist_submission_evidence(
        conn,
        session,
        attempt,
        stage_id,
        spec,
        explanation,
        artifacts,
        payload_hash,
    )


def submit_stage(
    conn: sqlite3.Connection,
    session_id: str,
    stage_id: str,
    explanation: str | None,
    artifacts: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    session = load_session(conn, session_id)
    _require_session_mutable(session)
    spec = load_mission_spec(conn, str(session["mission_id"]))
    assert_stage_is_current(session, spec, stage_id)
    attempt = _require_attempt(conn, session_id, stage_id)
    artifact_list = [item for item in (artifacts or []) if isinstance(item, dict)]
    payload = {"artifacts": artifact_list, "explanation": explanation}
    payload_hash = canonical_sha256(payload)

    already = str(attempt.get("status") or "") == ATTEMPT_SUBMITTED
    if already:
        refreshed = load_session(conn, session_id)
        nxt = next_stage_id(spec, stage_id)
        current_stage_id = str(refreshed.get("current_stage_id") or stage_id)
        return {
            "session_id": session_id,
            "stage_id": stage_id,
            "attempt_id": attempt["id"],
            "status": ATTEMPT_SUBMITTED,
            "current_stage_id": current_stage_id,
            "next_stage_id": nxt,
            "session_status": refreshed.get("status") or SESSION_ACTIVE,
            "payload_hash": payload_hash,
            "resumed": True,
        }

    if str(attempt.get("status") or "") != ATTEMPT_ACTIVE:
        raise ConflictError(
            "Submit requires an ACTIVE stage attempt",
            details={"attempt_status": attempt.get("status")},
        )
    completed_at = _now()
    conn.execute(
        """
        UPDATE stage_attempts
        SET status = ?, completed_at = ?
        WHERE id = ?
        """,
        (ATTEMPT_SUBMITTED, completed_at, attempt["id"]),
    )
    attempt["status"] = ATTEMPT_SUBMITTED
    claims = _persist_submission_evidence(
        conn, session, attempt, stage_id, spec, explanation, artifact_list, payload_hash
    )
    nxt = next_stage_id(spec, stage_id)
    if nxt:
        conn.execute(
            "UPDATE mission_sessions SET current_stage_id = ? WHERE id = ?",
            (nxt, session_id),
        )
        session["current_stage_id"] = nxt
        ensure_ready_attempt(conn, session_id, get_stage(spec, nxt))
    conn.commit()

    from app.core.evidence import record_activity

    record_activity(
        conn,
        str(session["learner_id"]),
        "stage_submitted",
        {
            "session_id": session_id,
            "stage_id": stage_id,
            "attempt_id": attempt["id"],
            "explanation": explanation,
            "artifacts": artifact_list,
            "payload_hash": payload_hash,
            "evidence_ids": [claim.get("id") for claim in claims],
        },
    )

    refreshed = load_session(conn, session_id)
    current_stage_id = str(refreshed.get("current_stage_id") or stage_id)
    return {
        "session_id": session_id,
        "stage_id": stage_id,
        "attempt_id": attempt["id"],
        "status": ATTEMPT_SUBMITTED,
        "current_stage_id": current_stage_id,
        "next_stage_id": nxt,
        "session_status": refreshed.get("status") or SESSION_ACTIVE,
        "payload_hash": payload_hash,
        "resumed": False,
    }


def evaluate_gate(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    session = load_session(conn, session_id)
    spec = load_mission_spec(conn, str(session["mission_id"]))
    from app.core.gates import evaluate_session_gate

    return evaluate_session_gate(conn, session, spec)
