"""Flagship V00–V12 index, prerequisite sequencing, and generic ADR/git artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

from app.core.evidence import record_activity, row_dict, sha256_hex, utc_now

SCHEMA = "learningos.flagship.v1"
INDEX_ID = "g6.flagship.index"
ARTIFACT_ADR = "adr"
ARTIFACT_GIT = "git"
EVENT_ADR = "learner_artifact.adr"
EVENT_GIT = "learner_artifact.git"
SESSION_COMPLETED = "COMPLETED"
MISSION_ID_RE = re.compile(r"^M[0-9]{2}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_IDS: tuple[str, ...] = tuple(f"V{i:02d}" for i in range(0, 13))
MISSION_IDS: tuple[str, ...] = tuple(f"M{i:02d}" for i in range(1, 43))

_INDEX_CACHE: dict[str, Any] | None = None
_INDEX_PATH_CACHE: str | None = None


class FlagshipError(ValueError):
    def __init__(self, message: str, *, code: str = "FLAGSHIP_ERROR") -> None:
        super().__init__(message)
        self.code = code


def fixture_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "fixtures" / "flagship"


def _parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        entries[rel.strip()] = digest.strip().lower()
    return entries


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_index(index: dict[str, Any]) -> None:
    if index.get("schema") != SCHEMA:
        raise FlagshipError("Unsupported flagship schema", code="BAD_SCHEMA")
    versions = index.get("versions")
    missions = index.get("missions")
    if not isinstance(versions, list) or len(versions) != len(VERSION_IDS):
        raise FlagshipError("Flagship index must declare V00–V12", code="BAD_INDEX")
    if not isinstance(missions, dict):
        raise FlagshipError("Flagship index missions map is required", code="BAD_INDEX")
    seen_versions: list[str] = []
    assigned: dict[str, str] = {}
    for position, version in enumerate(versions):
        if not isinstance(version, dict):
            raise FlagshipError("Flagship version entries must be objects", code="BAD_INDEX")
        version_id = str(version.get("id") or "")
        if version_id != VERSION_IDS[position]:
            raise FlagshipError("Flagship versions must be V00–V12 in order", code="BAD_INDEX")
        seen_versions.append(version_id)
        prereqs = version.get("prerequisites") if isinstance(version.get("prerequisites"), list) else []
        expected = [] if position == 0 else [VERSION_IDS[position - 1]]
        if list(prereqs) != expected:
            raise FlagshipError(
                f"{version_id} spine prerequisites must be {expected}",
                code="BAD_PREREQ",
            )
        mission_ids = version.get("missions") if isinstance(version.get("missions"), list) else []
        if not mission_ids:
            raise FlagshipError(f"{version_id} must list missions", code="BAD_INDEX")
        for mission_id in mission_ids:
            if not isinstance(mission_id, str) or not MISSION_ID_RE.fullmatch(mission_id):
                raise FlagshipError(f"Invalid mission id in {version_id}", code="BAD_INDEX")
            if mission_id in assigned:
                raise FlagshipError(
                    f"{mission_id} assigned to both {assigned[mission_id]} and {version_id}",
                    code="BAD_INDEX",
                )
            assigned[mission_id] = version_id
    if tuple(assigned.keys()) != MISSION_IDS or set(missions) != set(MISSION_IDS):
        missing = [mid for mid in MISSION_IDS if mid not in assigned]
        extra = sorted(set(assigned) | set(missions) - set(MISSION_IDS))
        raise FlagshipError(
            f"Flagship missions must be M01–M42 once; missing={missing} extra={extra}",
            code="BAD_INDEX",
        )
    order_by_id = {
        str(item.get("id")): int(item.get("order_index") or 0)
        for item in missions.values()
        if isinstance(item, dict)
    }
    for mission_id in MISSION_IDS:
        record = missions.get(mission_id)
        if not isinstance(record, dict):
            raise FlagshipError(f"Missing mission record {mission_id}", code="BAD_INDEX")
        version_id = str(record.get("flagship_version") or "")
        if version_id != assigned[mission_id]:
            raise FlagshipError(f"{mission_id} version mismatch", code="BAD_INDEX")
        prereq_block = record.get("prerequisites") if isinstance(record.get("prerequisites"), dict) else {}
        blocking = prereq_block.get("blocking") if isinstance(prereq_block.get("blocking"), list) else []
        helpful = prereq_block.get("helpful") if isinstance(prereq_block.get("helpful"), list) else []
        self_order = int(record.get("order_index") or 0)
        for dep in blocking:
            if dep not in missions:
                raise FlagshipError(f"{mission_id} blocking prereq {dep} is unknown", code="BAD_PREREQ")
            if dep == mission_id:
                raise FlagshipError(f"{mission_id} cannot block on itself", code="BAD_PREREQ")
            dep_order = int(order_by_id.get(str(dep)) or 0)
            if dep_order >= self_order:
                raise FlagshipError(
                    f"{mission_id} blocking prereq {dep} is not earlier in the spine",
                    code="BAD_PREREQ",
                )
        for dep in helpful:
            if dep not in missions:
                raise FlagshipError(f"{mission_id} helpful prereq {dep} is unknown", code="BAD_PREREQ")


def load_index(path: Path | str | None = None, *, reload: bool = False) -> dict[str, Any]:
    global _INDEX_CACHE, _INDEX_PATH_CACHE
    directory = Path(path) if path is not None else fixture_dir()
    directory = directory.resolve()
    cache_key = str(directory)
    if not reload and _INDEX_CACHE is not None and _INDEX_PATH_CACHE == cache_key:
        return _INDEX_CACHE
    index_path = directory / "index.json" if directory.is_dir() else directory
    if directory.is_dir():
        sums_path = directory / "SHA256SUMS"
        if sums_path.is_file():
            expected = _parse_sha256sums(sums_path)
            for rel, digest in expected.items():
                actual = _file_sha256(directory / rel)
                if actual != digest:
                    raise FlagshipError(f"SHA256 mismatch for {rel}", code="BAD_DIGEST")
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = manifest.get("digest")
            if isinstance(digest, str) and digest.lower() != _file_sha256(index_path):
                raise FlagshipError("manifest digest does not match index.json", code="BAD_DIGEST")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FlagshipError("Flagship index must be an object", code="BAD_INDEX")
    _validate_index(payload)
    _INDEX_CACHE = payload
    _INDEX_PATH_CACHE = cache_key
    return payload


def versions(index: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = index if index is not None else load_index()
    items = payload.get("versions")
    return [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def mission_record(mission_id: str, index: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not isinstance(mission_id, str) or not mission_id:
        return None
    payload = index if index is not None else load_index()
    missions = payload.get("missions") if isinstance(payload.get("missions"), dict) else {}
    record = missions.get(mission_id)
    return dict(record) if isinstance(record, dict) else None


def version_for_mission(mission_id: str, index: dict[str, Any] | None = None) -> dict[str, Any] | None:
    record = mission_record(mission_id, index)
    if record is None:
        return None
    version_id = str(record.get("flagship_version") or "")
    for version in versions(index):
        if str(version.get("id") or "") == version_id:
            return version
    return None


def version_by_id(version_id: str, index: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not isinstance(version_id, str):
        return None
    for version in versions(index):
        if str(version.get("id") or "") == version_id:
            return version
    return None


def blocking_prerequisites(mission_id: str, index: dict[str, Any] | None = None) -> list[str]:
    record = mission_record(mission_id, index)
    if record is None:
        return []
    block = record.get("prerequisites") if isinstance(record.get("prerequisites"), dict) else {}
    raw = block.get("blocking") if isinstance(block.get("blocking"), list) else []
    return [str(item) for item in raw if isinstance(item, str) and item]


def helpful_prerequisites(mission_id: str, index: dict[str, Any] | None = None) -> list[str]:
    record = mission_record(mission_id, index)
    if record is None:
        return []
    block = record.get("prerequisites") if isinstance(record.get("prerequisites"), dict) else {}
    raw = block.get("helpful") if isinstance(block.get("helpful"), list) else []
    return [str(item) for item in raw if isinstance(item, str) and item]


def unmet_blocking(
    mission_id: str,
    completed_missions: Iterable[str],
    catalog_ids: Iterable[str] | None = None,
    index: dict[str, Any] | None = None,
) -> list[str]:
    completed = {str(item) for item in completed_missions}
    catalog = {str(item) for item in catalog_ids} if catalog_ids is not None else None
    unmet: list[str] = []
    for dep in blocking_prerequisites(mission_id, index):
        if catalog is not None and dep not in catalog:
            continue
        if dep not in completed:
            unmet.append(dep)
    return unmet


def mission_unlocked(
    mission_id: str,
    completed_missions: Iterable[str],
    catalog_ids: Iterable[str] | None = None,
    index: dict[str, Any] | None = None,
) -> bool:
    return not unmet_blocking(mission_id, completed_missions, catalog_ids, index)


def _completed_missions(conn: sqlite3.Connection, learner_id: str) -> set[str]:
    try:
        rows = conn.execute(
            """
            SELECT mission_id FROM mission_sessions
            WHERE learner_id = ? AND status = ?
            """,
            (learner_id, SESSION_COMPLETED),
        ).fetchall()
    except sqlite3.Error:
        return set()
    return {str(row_dict(row).get("mission_id")) for row in rows if row_dict(row).get("mission_id")}


def _catalog_ids(conn: sqlite3.Connection) -> set[str]:
    try:
        rows = conn.execute("SELECT id FROM missions").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row_dict(row).get("id")) for row in rows if row_dict(row).get("id")}


def version_status(
    version: dict[str, Any],
    completed_missions: Iterable[str],
    catalog_ids: Iterable[str] | None = None,
    index: dict[str, Any] | None = None,
) -> str:
    completed = {str(item) for item in completed_missions}
    catalog = {str(item) for item in catalog_ids} if catalog_ids is not None else None
    mission_ids = [str(item) for item in version.get("missions") or [] if isinstance(item, str)]
    relevant = [mid for mid in mission_ids if catalog is None or mid in catalog]
    if not relevant:
        return "ABSENT"
    done = [mid for mid in relevant if mid in completed]
    if done and len(done) == len(relevant):
        return "COMPLETE"
    if done:
        return "IN_PROGRESS"
    if any(mission_unlocked(mid, completed, catalog, index) for mid in relevant):
        return "AVAILABLE"
    return "LOCKED"


def learner_progress(
    conn: sqlite3.Connection,
    learner_id: str,
    *,
    index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    payload = index if index is not None else load_index()
    completed = _completed_missions(conn, learner_id.strip())
    catalog = _catalog_ids(conn)
    progress: list[dict[str, Any]] = []
    for version in versions(payload):
        mission_ids = [str(item) for item in version.get("missions") or [] if isinstance(item, str)]
        relevant = [mid for mid in mission_ids if mid in catalog] if catalog else mission_ids
        done = [mid for mid in relevant if mid in completed]
        progress.append(
            {
                "id": version.get("id"),
                "name": version.get("name"),
                "release_tag": version.get("release_tag"),
                "architecture_step": version.get("architecture_step"),
                "status": version_status(version, completed, catalog or None, payload),
                "missions": mission_ids,
                "completed_count": len(done),
                "mission_count": len(relevant) if relevant else len(mission_ids),
            }
        )
    return progress


def current_version_id(progress: list[dict[str, Any]]) -> str | None:
    current: str | None = None
    for item in progress:
        status = str(item.get("status") or "")
        version_id = item.get("id") if isinstance(item.get("id"), str) else None
        if status == "COMPLETE":
            current = version_id
            continue
        if status in {"IN_PROGRESS", "AVAILABLE"}:
            return version_id
        if status == "LOCKED" and current:
            return current
    return current


def _normalize_git_sha(value: str) -> str:
    digest = value.strip().lower()
    if GIT_SHA_RE.fullmatch(digest) or SHA256_RE.fullmatch(digest):
        return digest
    raise FlagshipError("git_sha must be a 40-char git SHA-1 or 64-char SHA-256", code="BAD_GIT_SHA")


def record_adr(
    conn: sqlite3.Connection,
    *,
    learner_id: str,
    mission_id: str,
    title: str,
    context_text: str,
    decision_text: str,
    consequences_text: str,
    status: str = "PROPOSED",
    adr_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    if not isinstance(mission_id, str) or not mission_id.strip():
        raise ValueError("mission_id is required")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    lid = learner_id.strip()
    mid = mission_id.strip()
    record_id = adr_id or f"ADR-{uuid.uuid4()}"
    created_at = utc_now()
    version = version_for_mission(mid)
    payload = {
        "id": record_id,
        "learner_id": lid,
        "mission_id": mid,
        "title": title.strip(),
        "status": status or "PROPOSED",
        "context_text": context_text or "",
        "decision_text": decision_text or "",
        "consequences_text": consequences_text or "",
        "artifact_type": ARTIFACT_ADR,
        "flagship_version": version.get("id") if version else None,
        "created_at": created_at,
    }
    conn.execute(
        """
        INSERT INTO adrs (
            id, learner_id, mission_id, title, status,
            context_text, decision_text, consequences_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["id"],
            payload["learner_id"],
            payload["mission_id"],
            payload["title"],
            payload["status"],
            payload["context_text"],
            payload["decision_text"],
            payload["consequences_text"],
            created_at,
        ),
    )
    record_activity(conn, lid, EVENT_ADR, payload)
    return payload


def record_git_artifact(
    conn: sqlite3.Connection,
    *,
    learner_id: str,
    mission_id: str,
    git_sha: str,
    message: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    if not isinstance(mission_id, str) or not mission_id.strip():
        raise ValueError("mission_id is required")
    lid = learner_id.strip()
    mid = mission_id.strip()
    sha = _normalize_git_sha(git_sha)
    version = version_for_mission(mid)
    payload = {
        "id": artifact_id or f"GIT-{uuid.uuid4()}",
        "learner_id": lid,
        "mission_id": mid,
        "git_sha": sha,
        "artifact_hash": sha if SHA256_RE.fullmatch(sha) else sha256_hex(sha),
        "message": message or "",
        "artifact_type": ARTIFACT_GIT,
        "flagship_version": version.get("id") if version else None,
        "created_at": utc_now(),
    }
    record_activity(conn, lid, EVENT_GIT, payload)
    return payload


def _events_of_type(conn: sqlite3.Connection, learner_id: str, event_type: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT id, payload_json, created_at, event_hash
            FROM learning_events
            WHERE learner_id = ? AND event_type = ?
            ORDER BY rowid ASC
            """,
            (learner_id, event_type),
        ).fetchall()
    except sqlite3.Error:
        return []
    items: list[dict[str, Any]] = []
    for row in rows:
        data = row_dict(row)
        body: dict[str, Any] = {}
        raw = data.get("payload_json")
        if isinstance(raw, str) and raw.strip():
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                body = loaded
        body.setdefault("id", data.get("id"))
        body.setdefault("created_at", data.get("created_at"))
        body["event_hash"] = data.get("event_hash")
        items.append(body)
    return items


def list_adrs(conn: sqlite3.Connection, learner_id: str) -> list[dict[str, Any]]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    lid = learner_id.strip()
    items: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            """
            SELECT id, learner_id, mission_id, title, status,
                   context_text, decision_text, consequences_text, created_at
            FROM adrs
            WHERE learner_id = ?
            ORDER BY rowid ASC
            """,
            (lid,),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    for row in rows:
        data = row_dict(row)
        data["artifact_type"] = ARTIFACT_ADR
        version = version_for_mission(str(data.get("mission_id") or ""))
        data["flagship_version"] = version.get("id") if version else None
        items.append(data)
    seen = {str(item.get("id")) for item in items if item.get("id")}
    for event in _events_of_type(conn, lid, EVENT_ADR):
        if str(event.get("id") or "") in seen:
            continue
        event.setdefault("artifact_type", ARTIFACT_ADR)
        items.append(event)
    try:
        evidence_rows = conn.execute(
            """
            SELECT * FROM evidence_items
            WHERE learner_id = ? AND artifact_type = ?
            ORDER BY rowid ASC
            """,
            (lid, ARTIFACT_ADR),
        ).fetchall()
    except sqlite3.Error:
        evidence_rows = []
    for row in evidence_rows:
        data = row_dict(row)
        key = str(data.get("id") or "")
        if key in seen:
            continue
        data["artifact_type"] = ARTIFACT_ADR
        version = version_for_mission(str(data.get("mission_id") or ""))
        data["flagship_version"] = version.get("id") if version else None
        items.append(data)
    return items


def list_git_artifacts(conn: sqlite3.Connection, learner_id: str) -> list[dict[str, Any]]:
    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError("learner_id is required")
    lid = learner_id.strip()
    items = _events_of_type(conn, lid, EVENT_GIT)
    for item in items:
        item.setdefault("artifact_type", ARTIFACT_GIT)
    seen = {str(item.get("id")) for item in items if item.get("id")}
    try:
        rows = conn.execute(
            """
            SELECT * FROM evidence_items
            WHERE learner_id = ? AND artifact_type = ?
            ORDER BY rowid ASC
            """,
            (lid, ARTIFACT_GIT),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    for row in rows:
        data = row_dict(row)
        if str(data.get("id") or "") in seen:
            continue
        data["artifact_type"] = ARTIFACT_GIT
        version = version_for_mission(str(data.get("mission_id") or ""))
        data["flagship_version"] = version.get("id") if version else None
        git_sha = data.get("artifact_path")
        if isinstance(git_sha, str) and git_sha.startswith("git:"):
            data["git_sha"] = git_sha[4:]
        items.append(data)
    return items


def list_learner_artifacts(conn: sqlite3.Connection, learner_id: str) -> dict[str, list[dict[str, Any]]]:
    return {
        ARTIFACT_ADR: list_adrs(conn, learner_id),
        ARTIFACT_GIT: list_git_artifacts(conn, learner_id),
    }


def public_index(index: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = index if index is not None else load_index()
    version_list = versions(payload)
    return {
        "schema": payload.get("schema"),
        "id": payload.get("id") or INDEX_ID,
        "system": payload.get("system"),
        "title": payload.get("title"),
        "learner_artifact_types": payload.get("learner_artifact_types") or [ARTIFACT_ADR, ARTIFACT_GIT],
        "versions": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "architecture_step": item.get("architecture_step"),
                "release_tag": item.get("release_tag"),
                "phase_ids": item.get("phase_ids") or [],
                "missions": item.get("missions") or [],
                "prerequisites": item.get("prerequisites") or [],
                "unlocks": item.get("unlocks"),
                "order_index": item.get("order_index"),
            }
            for item in version_list
        ],
        "missions": payload.get("missions") or {},
        "version_count": len(version_list),
        "mission_count": len(payload.get("missions") or {}),
    }


def flagship_payload(
    conn: sqlite3.Connection | None = None,
    learner_id: str | None = None,
    *,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = public_index(index)
    if conn is None or not learner_id:
        return body
    progress = learner_progress(conn, learner_id, index=index)
    artifacts = list_learner_artifacts(conn, learner_id)
    body["learner_id"] = learner_id
    body["progress"] = progress
    body["current_version_id"] = current_version_id(progress)
    body["artifacts"] = {
        ARTIFACT_ADR: artifacts[ARTIFACT_ADR],
        ARTIFACT_GIT: artifacts[ARTIFACT_GIT],
        "counts": {
            ARTIFACT_ADR: len(artifacts[ARTIFACT_ADR]),
            ARTIFACT_GIT: len(artifacts[ARTIFACT_GIT]),
        },
    }
    return body


def annotate_next_action(
    payload: dict[str, Any],
    *,
    conn: sqlite3.Connection | None = None,
    completed_missions: Iterable[str] | None = None,
    catalog_ids: Iterable[str] | None = None,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach flagship version metadata. Never special-cases a mission id."""
    _ = completed_missions, catalog_ids
    mission_id = payload.get("mission_id")
    version = version_for_mission(str(mission_id), index) if isinstance(mission_id, str) else None
    if version is None:
        return payload
    version_id = str(version.get("id") or "")
    flagship = {
        "version_id": version_id,
        "name": version.get("name"),
        "release_tag": version.get("release_tag"),
        "architecture_step": version.get("architecture_step"),
    }
    if conn is not None:
        learner_id = payload.get("learner_id")
        if isinstance(learner_id, str) and learner_id.strip():
            artifacts = list_learner_artifacts(conn, learner_id.strip())
            flagship["artifact_counts"] = {
                ARTIFACT_ADR: len(artifacts[ARTIFACT_ADR]),
                ARTIFACT_GIT: len(artifacts[ARTIFACT_GIT]),
            }
    payload["flagship_version"] = version_id
    payload["flagship"] = flagship
    today = payload.get("today")
    if isinstance(today, dict):
        today["flagship_version"] = version_id
    return payload
