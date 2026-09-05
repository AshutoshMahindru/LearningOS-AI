#!/usr/bin/env python3
"""Dry-run or import V2 prototype learner state into V3 LEARNINGOS_HOME.

V2 freeze (tag v2-prototype-freeze, SHA 35293455ff769855014a49fa56315b22829e74b1)
stores a single learner as JSON under tracking/ and has no portable export.
This tool reads that layout (a V2 repo root, a tracking/ directory, or a fixture
that mirrors it). It never merges V2 application code and never writes learner
state into the Git worktree.

Counts
    learners   V2 learner bundles discovered
    progress   session records + mission_status entries
    artifacts  files under artifacts/ plus evidence records with type=artifact

--dry-run inspects only (exit 0 if readable, non-zero on garbage) and writes
nothing under LEARNINGOS_HOME. Import is additive: colliding V3 learner rows
are not overwritten unless --force, which snapshots first.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


V2_FREEZE_SHA = "35293455ff769855014a49fa56315b22829e74b1"
V2_FREEZE_TAG = "v2-prototype-freeze"
LEARNER_ID_PREFIX = "v2:"
USERNAME_PREFIX = "v2_"
BUNDLE_STATE = "learner_state.json"
BUNDLE_MODEL = "learner_model.json"
STATE_KEYS = ("learner_id", "current_mission", "mission_status", "blockers", "display_name")
MODEL_KEYS = ("learner_id", "autonomy_level", "competencies", "gate_history", "evidence_summary")
TYPED_ARTIFACT = "artifact"

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"


class V2MigrateError(Exception):
    """Unreadable V2 source, unsafe destination, or refused overwrite."""


@dataclass
class V2Bundle:
    path: Path
    learner_id: str
    display_name: str
    autonomy_tier: int
    learner_state: dict[str, Any]
    learner_model: dict[str, Any]
    evidence: list[Any]
    sessions: list[Any]
    competencies: dict[str, Any]
    mission_status: dict[str, Any]
    artifact_files: list[Path]

    @property
    def progress_count(self) -> int:
        return len(self.sessions) + len(self.mission_status)

    @property
    def artifact_count(self) -> int:
        typed = sum(
            1
            for item in self.evidence
            if isinstance(item, Mapping) and str(item.get("type", "")).lower() == TYPED_ARTIFACT
        )
        return len(self.artifact_files) + typed


@dataclass
class Inspection:
    source: Path
    bundles: list[V2Bundle]

    @property
    def learners(self) -> int:
        return len(self.bundles)

    @property
    def progress(self) -> int:
        return sum(bundle.progress_count for bundle in self.bundles)

    @property
    def artifacts(self) -> int:
        return sum(bundle.artifact_count for bundle in self.bundles)


@dataclass
class ImportResult:
    source: Path
    home: Path
    inspection: Inspection
    id_map: dict[str, str] = field(default_factory=dict)
    overwritten_ids: list[str] = field(default_factory=list)
    snapshot: Path | None = None


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise V2MigrateError(f"unreadable V2 file: {path}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise V2MigrateError(f"invalid JSON: {path}") from exc


def _require_type(path: Path, value: Any, expected: type) -> Any:
    if not isinstance(value, expected):
        raise V2MigrateError(f"{path} must be a {expected.__name__}")
    return value


def _load_optional(path: Path, expected: type, default: Any) -> Any:
    if not path.is_file():
        return default
    return _require_type(path, _read_json(path), expected)


def _looks_like_state(data: Any) -> bool:
    return isinstance(data, dict) and any(key in data for key in STATE_KEYS)


def _looks_like_model(data: Any) -> bool:
    return isinstance(data, dict) and any(key in data for key in MODEL_KEYS)


def _is_bundle_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    state = path / BUNDLE_STATE
    model = path / BUNDLE_MODEL
    return state.is_file() or model.is_file()


def discover_bundle_dirs(source: Path) -> list[Path]:
    source = source.expanduser()
    if not source.exists():
        raise V2MigrateError(f"V2 source does not exist: {source}")
    if not source.is_dir():
        raise V2MigrateError(f"V2 source is not a directory: {source}")
    source = source.resolve()

    if _is_bundle_dir(source):
        return [source]
    tracking = source / "tracking"
    if _is_bundle_dir(tracking):
        return [tracking]

    found: list[Path] = []
    try:
        children = sorted(source.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise V2MigrateError(f"unreadable V2 source: {source}") from exc
    for child in children:
        if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
            continue
        if _is_bundle_dir(child):
            found.append(child)
        elif _is_bundle_dir(child / "tracking"):
            found.append(child / "tracking")
    if not found:
        raise V2MigrateError(f"no V2 tracking layout under {source}")
    return found


def _artifact_roots(bundle_dir: Path) -> list[Path]:
    roots = [bundle_dir / "artifacts"]
    if bundle_dir.name == "tracking":
        roots.append(bundle_dir.parent / "artifacts")
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not root.is_dir():
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def list_artifact_files(bundle_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root in _artifact_roots(bundle_dir):
        for current, directories, filenames in os.walk(root):
            directories[:] = [name for name in directories if not name.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                files.append(Path(current) / name)
    files.sort()
    return files


def _load_evidence(bundle_dir: Path) -> list[Any]:
    records = list(_load_optional(bundle_dir / "evidence.json", list, []))
    ledger = bundle_dir / "evidence_ledger.jsonl"
    if ledger.is_file():
        try:
            lines = ledger.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            raise V2MigrateError(f"unreadable V2 file: {ledger}") from exc
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise V2MigrateError(f"invalid JSONL at {ledger}:{index}") from exc
    return records


def _mission_status(state: dict[str, Any], bundle_dir: Path) -> dict[str, Any]:
    status = state.get("mission_status") or {}
    if not isinstance(status, dict):
        raise V2MigrateError(f"{bundle_dir / BUNDLE_STATE} mission_status must be an object")
    progress_path = bundle_dir / "progress.json"
    if progress_path.is_file():
        extra = _read_json(progress_path)
        if isinstance(extra, dict):
            nested = extra.get("mission_status")
            if isinstance(nested, dict):
                extra = nested
            if all(isinstance(key, str) for key in extra):
                merged = dict(status)
                merged.update(extra)
                return merged
        raise V2MigrateError(f"{progress_path} must be an object of mission statuses")
    return dict(status)


def _autonomy_tier(model: Mapping[str, Any], state: Mapping[str, Any]) -> int:
    raw = model.get("autonomy_level", state.get("autonomy_level", state.get("autonomy_tier", 1)))
    if isinstance(raw, bool) or raw is None:
        return 1
    if isinstance(raw, int):
        return max(0, min(4, raw))
    text = str(raw).strip().upper()
    match = re.fullmatch(r"A([0-5])", text)
    if match:
        return min(4, int(match.group(1)))
    try:
        return max(0, min(4, int(text)))
    except ValueError:
        return 1


def _display_name(learner_id: str, state: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    for candidate in (state.get("display_name"), model.get("display_name"), state.get("username")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return learner_id


def _learner_id(state: Mapping[str, Any], model: Mapping[str, Any], bundle_dir: Path) -> str:
    for candidate in (state.get("learner_id"), model.get("learner_id")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise V2MigrateError(f"V2 bundle at {bundle_dir} is missing learner_id")


def load_bundle(bundle_dir: Path) -> V2Bundle:
    bundle_dir = bundle_dir.resolve()
    state_path = bundle_dir / BUNDLE_STATE
    model_path = bundle_dir / BUNDLE_MODEL
    state: dict[str, Any] = {}
    model: dict[str, Any] = {}
    if state_path.is_file():
        state = _require_type(state_path, _read_json(state_path), dict)
        if not _looks_like_state(state):
            raise V2MigrateError(f"{state_path} is not V2 learner_state")
    if model_path.is_file():
        model = _require_type(model_path, _read_json(model_path), dict)
        if not _looks_like_model(model):
            raise V2MigrateError(f"{model_path} is not V2 learner_model")
    if not state and not model:
        raise V2MigrateError(f"no V2 learner JSON in {bundle_dir}")

    evidence = _load_evidence(bundle_dir)
    sessions = _load_optional(bundle_dir / "sessions.json", list, [])
    competencies = _load_optional(bundle_dir / "competencies.json", dict, {})
    if not competencies and isinstance(model.get("competencies"), dict):
        competencies = dict(model["competencies"])
    learner_id = _learner_id(state, model, bundle_dir)
    return V2Bundle(
        path=bundle_dir,
        learner_id=learner_id,
        display_name=_display_name(learner_id, state, model),
        autonomy_tier=_autonomy_tier(model, state),
        learner_state=state,
        learner_model=model,
        evidence=evidence,
        sessions=sessions,
        competencies=competencies,
        mission_status=_mission_status(state, bundle_dir),
        artifact_files=list_artifact_files(bundle_dir),
    )


def inspect_source(source: Path) -> Inspection:
    bundles = [load_bundle(path) for path in discover_bundle_dirs(source)]
    if not bundles:
        raise V2MigrateError(f"no V2 learners in {source}")
    return Inspection(source=source.expanduser().resolve(), bundles=bundles)


def freeze_commit_available(repo: Path | None = None) -> bool:
    root = repo or REPO_ROOT
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{V2_FREEZE_SHA}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def extract_freeze_tracking(dest: Path, repo: Path | None = None) -> Path:
    """Materialise freeze tracking/ via git archive (does not checkout the freeze)."""
    root = (repo or REPO_ROOT).resolve()
    if not freeze_commit_available(root):
        raise V2MigrateError(
            f"V2 freeze commit {V2_FREEZE_SHA} is not in {root}; fetch origin/feature/learningos-app-v1"
        )
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "archive", "--format=tar", V2_FREEZE_SHA, "tracking"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip() or "git archive failed"
        raise V2MigrateError(detail)
    _safe_extract_tar(completed.stdout, dest)
    tracking = dest / "tracking"
    if not tracking.is_dir():
        raise V2MigrateError("freeze archive did not contain tracking/")
    return tracking


def _safe_extract_tar(data: bytes, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise V2MigrateError(f"unsafe archive member: {member.name}")
            target = (dest / member.name).resolve()
            if target != dest and not str(target).startswith(str(dest) + os.sep):
                raise V2MigrateError(f"unsafe archive member: {member.name}")
        try:
            archive.extractall(dest, filter="data")
        except TypeError:
            archive.extractall(dest)


def _ensure_backend_path() -> None:
    backend = str(BACKEND_ROOT)
    if backend not in sys.path:
        sys.path.insert(0, backend)


def reject_home_inside_worktree(home: Path, repo_root: Path | None = None) -> Path:
    resolved = home.expanduser().resolve()
    root = (repo_root or REPO_ROOT).resolve()
    if resolved == root or is_within(resolved, root):
        raise V2MigrateError(
            f"LEARNINGOS_HOME must not be inside the Git worktree ({resolved} is under {root})"
        )
    return resolved


def _sanitize_username(learner_id: str) -> str:
    raw = f"{USERNAME_PREFIX}{learner_id.strip().lower()}"
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_") or "v2_learner"
    return cleaned[:80]


def _unique_username(conn: sqlite3.Connection, base: str, *, allow_id: str | None = None) -> str:
    candidate = base
    suffix = 2
    while True:
        row = conn.execute("SELECT id FROM learners WHERE username = ?", (candidate,)).fetchone()
        if row is None or (allow_id is not None and row["id"] == allow_id):
            return candidate
        candidate = f"{base}_{suffix}"[:80]
        suffix += 1


def _namespaced_id(source_id: str) -> str:
    return f"{LEARNER_ID_PREFIX}{source_id}"


def _competency_level(item: Any) -> int:
    if isinstance(item, Mapping):
        raw = item.get("level", 0)
    else:
        raw = item
    try:
        level = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, level))


def _bind_home(home: Path) -> str | None:
    previous = os.environ.get("LEARNINGOS_HOME")
    os.environ["LEARNINGOS_HOME"] = str(home)
    return previous


def _restore_home(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("LEARNINGOS_HOME", None)
    else:
        os.environ["LEARNINGOS_HOME"] = previous


def _open_v3(home: Path) -> sqlite3.Connection:
    _ensure_backend_path()
    from app.db.database import get_connection, init_db

    init_db()
    return get_connection()


def _snapshot(home: Path) -> Path:
    _ensure_backend_path()
    from app.db.backup import create_backup

    dest = home / "backups"
    dest.mkdir(parents=True, exist_ok=True)
    return create_backup(dest, label="pre_v2_import")


def _allocate_identity(
    conn: sqlite3.Connection,
    bundle: V2Bundle,
    *,
    force: bool,
) -> tuple[str, str, bool]:
    desired_id = _namespaced_id(bundle.learner_id)
    desired_username = _sanitize_username(bundle.learner_id)
    existing = conn.execute("SELECT id, username FROM learners WHERE id = ?", (desired_id,)).fetchone()
    if existing is None:
        username = _unique_username(conn, desired_username)
        return desired_id, username, False
    if force:
        username = _unique_username(conn, desired_username, allow_id=desired_id)
        return desired_id, username, True
    minted = f"{desired_id}:{uuid.uuid4().hex[:8]}"
    username = _unique_username(conn, desired_username)
    return minted, username, False


def _upsert_learner(
    conn: sqlite3.Connection,
    *,
    learner_id: str,
    username: str,
    display_name: str,
    autonomy_tier: int,
    overwrite: bool,
) -> None:
    if overwrite:
        conn.execute(
            """
            UPDATE learners
            SET username = ?, display_name = ?, autonomy_tier = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (username, display_name, autonomy_tier, learner_id),
        )
        return
    conn.execute(
        """
        INSERT INTO learners (id, username, display_name, autonomy_tier)
        VALUES (?, ?, ?, ?)
        """,
        (learner_id, username, display_name, autonomy_tier),
    )


def _upsert_competencies(conn: sqlite3.Connection, learner_id: str, competencies: Mapping[str, Any]) -> None:
    for competency_id, item in competencies.items():
        if not isinstance(competency_id, str) or not competency_id.strip():
            continue
        level = _competency_level(item)
        conn.execute(
            """
            INSERT INTO competency_mastery (learner_id, competency_id, level, decay_score)
            VALUES (?, ?, ?, 1.0)
            ON CONFLICT(learner_id, competency_id) DO UPDATE SET
                level = MAX(competency_mastery.level, excluded.level),
                last_evaluated_at = CURRENT_TIMESTAMP
            """,
            (learner_id, competency_id.strip(), level),
        )


def _append_event(conn: sqlite3.Connection, learner_id: str, event_type: str, payload: dict[str, Any]) -> None:
    _ensure_backend_path()
    from app.db.ledger import EventLedger

    EventLedger(conn).append(learner_id, event_type, payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _put_artifacts(bundle: V2Bundle, learner_id: str) -> list[dict[str, str]]:
    _ensure_backend_path()
    from app.core.artifact_store import ArtifactStore

    store = ArtifactStore()
    stored: list[dict[str, str]] = []
    for path in bundle.artifact_files:
        digest = store.put(path.read_bytes())
        stored.append({"path": str(path), "sha256": digest, "name": path.name})
    snapshot = {
        "learner_id": bundle.learner_id,
        "imported_learner_id": learner_id,
        "learner_state": bundle.learner_state,
        "mission_status": bundle.mission_status,
        "sessions": bundle.sessions,
        "evidence": bundle.evidence,
        "competencies": bundle.competencies,
    }
    payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    digest = store.put(payload)
    stored.append({"path": "v2-bundle.json", "sha256": digest, "name": "v2-bundle.json"})
    return stored


def import_source(
    source: Path,
    home: Path,
    *,
    force: bool = False,
    repo_root: Path | None = None,
) -> ImportResult:
    inspection = inspect_source(source)
    resolved_home = reject_home_inside_worktree(home, repo_root)
    previous = _bind_home(resolved_home)
    conn: sqlite3.Connection | None = None
    try:
        conn = _open_v3(resolved_home)
        allocations: list[tuple[V2Bundle, str, str, bool]] = []
        for bundle in inspection.bundles:
            learner_id, username, overwrite = _allocate_identity(conn, bundle, force=force)
            allocations.append((bundle, learner_id, username, overwrite))
        overwritten = [learner_id for _, learner_id, _, overwrite in allocations if overwrite]
        snapshot: Path | None = None
        if overwritten:
            if not force:
                raise V2MigrateError("refusing to overwrite V3 learners without --force")
            snapshot = _snapshot(resolved_home)
        id_map: dict[str, str] = {}
        for bundle, learner_id, username, overwrite in allocations:
            _upsert_learner(
                conn,
                learner_id=learner_id,
                username=username,
                display_name=bundle.display_name,
                autonomy_tier=bundle.autonomy_tier,
                overwrite=overwrite,
            )
            _upsert_competencies(conn, learner_id, bundle.competencies)
            conn.commit()
            stored = _put_artifacts(bundle, learner_id)
            _append_event(
                conn,
                learner_id,
                "v2_import",
                {
                    "source_learner_id": bundle.learner_id,
                    "source_path": str(bundle.path),
                    "imported_at": _utc_now(),
                    "overwrite": overwrite,
                    "progress": bundle.progress_count,
                    "artifacts": bundle.artifact_count,
                    "freeze_sha": V2_FREEZE_SHA,
                },
            )
            if bundle.mission_status:
                _append_event(conn, learner_id, "v2_progress", _jsonable(bundle.mission_status))
            for session in bundle.sessions:
                payload = session if isinstance(session, dict) else {"session": session}
                _append_event(conn, learner_id, "v2_session", _jsonable(payload))
            for record in bundle.evidence:
                payload = record if isinstance(record, dict) else {"evidence": record}
                _append_event(conn, learner_id, "v2_evidence", _jsonable(payload))
            if stored:
                _append_event(
                    conn,
                    learner_id,
                    "v2_artifacts",
                    {"items": stored},
                )
            id_map[bundle.learner_id] = learner_id
        return ImportResult(
            source=inspection.source,
            home=resolved_home,
            inspection=inspection,
            id_map=id_map,
            overwritten_ids=overwritten,
            snapshot=snapshot,
        )
    finally:
        if conn is not None:
            conn.close()
        _restore_home(previous)


def format_inspection(inspection: Inspection, *, mode: str = "dry-run") -> str:
    lines = [
        f"mode: {mode}",
        "ok: true",
        f"source: {inspection.source}",
        f"learners: {inspection.learners}",
        f"progress: {inspection.progress}",
        f"artifacts: {inspection.artifacts}",
    ]
    for bundle in inspection.bundles:
        lines.append(
            f"bundle: id={bundle.learner_id} progress={bundle.progress_count} artifacts={bundle.artifact_count}"
        )
    return "\n".join(lines) + "\n"


def format_import(result: ImportResult) -> str:
    text = format_inspection(result.inspection, mode="import")
    extra = [
        f"home: {result.home}",
        f"imported_learners: {len(result.id_map)}",
        f"overwritten_learners: {len(result.overwritten_ids)}",
        f"snapshot: {result.snapshot if result.snapshot is not None else '-'}",
    ]
    for source_id, dest_id in result.id_map.items():
        extra.append(f"map: {source_id} -> {dest_id}")
    return text + "\n".join(extra) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="V2 data directory, tracking/, or fixture")
    parser.add_argument(
        "--from-freeze",
        action="store_true",
        help=f"git archive tracking/ from {V2_FREEZE_SHA} (no checkout, no network)",
    )
    parser.add_argument(
        "--home",
        "--dest",
        dest="home",
        type=Path,
        help="V3 LEARNINGOS_HOME (import only; must be outside the Git worktree)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect source and print counts; write nothing to LEARNINGOS_HOME",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite colliding namespaced V3 learner rows after a snapshot",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="repository root used for freeze archive and worktree guards",
    )
    return parser.parse_args(argv)


def _resolve_source(args: argparse.Namespace) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.from_freeze and args.source is not None:
        raise V2MigrateError("use either --source or --from-freeze, not both")
    if args.from_freeze:
        holding = tempfile.TemporaryDirectory(prefix="learningos-v2-freeze-")
        tracking = extract_freeze_tracking(Path(holding.name), repo=Path(args.repo))
        return tracking, holding
    if args.source is None:
        raise V2MigrateError("pass --source PATH or --from-freeze")
    return args.source, None


def _resolve_home(args: argparse.Namespace) -> Path:
    if args.home is not None:
        return args.home
    raw = os.environ.get("LEARNINGOS_HOME")
    if raw:
        return Path(raw)
    raise V2MigrateError("import requires --home or LEARNINGOS_HOME (refusing implicit ~/.learningos)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    holding: tempfile.TemporaryDirectory[str] | None = None
    try:
        source, holding = _resolve_source(args)
        if args.dry_run:
            inspection = inspect_source(source)
            sys.stdout.write(format_inspection(inspection))
            return 0
        home = _resolve_home(args)
        result = import_source(source, home, force=args.force, repo_root=Path(args.repo))
        sys.stdout.write(format_import(result))
        return 0
    except V2MigrateError as exc:
        print(f"v2_migrate: {exc}", file=sys.stderr)
        return 1
    finally:
        if holding is not None:
            holding.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
