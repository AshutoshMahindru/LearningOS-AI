from __future__ import annotations

import base64
import importlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from app.api import runtime
from app.core.config import get_settings, public_config, worker_alive
from app.core.errors import (
    AppError,
    ConflictError,
    CurriculumUnavailableError,
    InternalError,
    NotFoundError,
    StorageUnavailableError,
    TutorUnavailableError,
    ValidationAppError,
)
from app.core.security import ensure_auth_token, require_auth, require_loopback, rotate_auth_token
from app.core.version import PLATFORM_VERSION
from app.models.schemas import (
    ArtifactCreateRequest,
    CurriculumLoadRequest,
    ExecuteStageRequest,
    LearnerCreateRequest,
    PredictCommitRequest,
    RestoreRequest,
    SessionCreateRequest,
    SubmitStageRequest,
    TutorChatRequest,
)

router = APIRouter()
public_router = APIRouter()
protected_router = APIRouter(dependencies=[Depends(require_auth)])


def _open_connection() -> sqlite3.Connection:
    try:
        from app.db import database as dbmod
    except ImportError as exc:
        raise StorageUnavailableError("Storage module is not available") from exc
    settings = get_settings()
    from app.core.config import storage_honors_settings

    if not storage_honors_settings(dbmod, settings):
        raise StorageUnavailableError("Database module does not honor LEARNINGOS_HOME")
    get_connection = getattr(dbmod, "get_connection", None)
    if not callable(get_connection):
        raise StorageUnavailableError("get_connection is not available")
    try:
        return get_connection()
    except Exception as exc:
        raise InternalError("Failed to open database connection", status_code=503) from exc


@contextmanager
def _db_conn() -> Iterator[sqlite3.Connection]:
    conn = _open_connection()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {"value": row}


def _map_sqlite_write_error(exc: Exception, action: str) -> AppError:
    if isinstance(exc, sqlite3.IntegrityError):
        return InternalError(f"{action} failed", status_code=500, details={"reason": "integrity"})
    if isinstance(exc, sqlite3.OperationalError):
        return InternalError(f"{action} failed", status_code=503, details={"reason": "schema"})
    return InternalError(f"{action} failed", status_code=500)


def _get_artifact_store() -> Any:
    try:
        from app.core.artifact_store import ArtifactStore
    except ImportError as exc:
        raise StorageUnavailableError("Artifact store is not available") from exc
    settings = get_settings()
    root = settings.data_home / "artifacts"
    store: Any
    try:
        store = ArtifactStore(root)
    except TypeError:
        try:
            store = ArtifactStore(root=root)
        except TypeError:
            store = ArtifactStore()
    put = getattr(store, "put", None)
    get = getattr(store, "get", None)
    if not callable(put) or not callable(get):
        raise StorageUnavailableError("Artifact store does not expose put/get")
    return store


def _call_first(module: Any, names: tuple[str, ...], *args: Any) -> tuple[bool, Any]:
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return True, fn(*args)
    cls = getattr(module, "CurriculumRegistry", None) or getattr(module, "MissionLoader", None)
    if cls is None:
        return False, None
    try:
        inst = cls() if callable(cls) else cls
    except TypeError:
        inst = cls
    for name in names:
        fn = getattr(inst, name, None)
        if callable(fn):
            return True, fn(*args)
    return False, None


def _normalize_package(result: Any) -> dict[str, Any]:
    if result is None:
        return {"status": "LOADED"}
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        return {"packages": list(result)}
    if hasattr(result, "__dict__"):
        data = {k: v for k, v in vars(result).items() if not k.startswith("_")}
        if data:
            return data
    return {"status": "LOADED"}


def _as_package_list(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return list(result)
    if isinstance(result, dict):
        if "packages" in result and isinstance(result["packages"], list):
            return result["packages"]
        if "missions" in result and isinstance(result["missions"], list):
            return result["missions"]
        return [result]
    return [result]


def _import_backup_fn(name: str) -> Any | None:
    for modname in ("app.core.backup", "app.db.backup", "app.db.database", "app.db"):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _home_empty_of_db_and_artifacts(home: Path) -> bool:
    db = home / "learningos.db"
    if db.exists() or Path(str(db) + "-wal").exists() or Path(str(db) + "-shm").exists():
        return False
    artifacts = home / "artifacts"
    if artifacts.is_dir():
        for path in artifacts.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                return False
    return True


def _append_ledger_event(
    conn: sqlite3.Connection, learner_id: str, event_type: str, payload: dict[str, Any]
) -> None:
    try:
        from app.db.ledger import EventLedger
    except ImportError:
        return
    try:
        EventLedger(conn).append(learner_id, event_type, payload)
    except Exception:
        return


def _maybe_validate_imported_missions(result: Any) -> None:
    """Validate imported M## missions with 21A; G3-style ids stay G3-loader-only."""
    try:
        from app.core.mdl_types import MISSION_ID_PATTERN, ValidationError
        from app.core.mdl_validator import validate_mission
    except ImportError:
        return
    missions = getattr(result, "missions", None)
    if missions is None and isinstance(result, dict):
        missions = result.get("missions")
    if not missions:
        return
    pattern = re.compile(MISSION_ID_PATTERN)
    for spec in missions:
        if not isinstance(spec, dict):
            continue
        if not pattern.fullmatch(str(spec.get("id") or "")):
            continue
        try:
            validate_mission(spec)
        except ValidationError as exc:
            raise ValidationAppError(
                str(exc),
                details=runtime.validation_error_details(exc),
            ) from exc


def _register_loaded_package(result: Any) -> None:
    try:
        registry = importlib.import_module("app.core.registry")
    except ImportError:
        return
    _call_first(registry, ("register_package",), result)


def _decode_artifact_bytes(raw_b64: str) -> bytes:
    try:
        return base64.b64decode(raw_b64, validate=True)
    except Exception:
        try:
            return base64.b64decode(raw_b64)
        except Exception as exc:
            raise ValidationAppError("Invalid base64 artifact payload") from exc


@public_router.get("/system/health")
async def health_check() -> dict[str, Any]:
    settings = get_settings()
    alive = False
    try:
        alive = bool(worker_alive())
    except Exception:
        alive = False
    return {
        "status": "HEALTHY",
        "version": PLATFORM_VERSION,
        "worker_alive": alive,
        "database_path": str(settings.database_path),
    }


@public_router.get("/system/version")
async def system_version() -> dict[str, str]:
    return {"version": PLATFORM_VERSION}


@public_router.get("/system/config")
async def system_config() -> dict[str, str]:
    return public_config()


@public_router.post("/auth/bootstrap")
async def auth_bootstrap(request: Request) -> dict[str, str]:
    require_loopback(request)
    token = ensure_auth_token()
    return {"token": token, "token_type": "bearer"}


@protected_router.post("/auth/logout")
async def auth_logout() -> dict[str, str]:
    rotate_auth_token()
    return {"status": "LOGGED_OUT"}


@protected_router.post("/learners")
async def create_learner(payload: LearnerCreateRequest) -> dict[str, Any]:
    learner_id = str(uuid.uuid4())
    display_name = payload.display_name or payload.username
    with _db_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
                (learner_id, payload.username, display_name),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise ConflictError("Learner username already exists") from exc
        except Exception as exc:
            conn.rollback()
            raise _map_sqlite_write_error(exc, "Learner create") from exc
        _append_ledger_event(
            conn,
            learner_id,
            "learner_created",
            {"username": payload.username, "display_name": display_name},
        )
    return {"learner_id": learner_id, "username": payload.username, "display_name": display_name}


@protected_router.get("/learners/{learner_id}")
async def get_learner(learner_id: str) -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, display_name, autonomy_tier, created_at, updated_at FROM learners WHERE id = ?",
                (learner_id,),
            )
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learners WHERE id = ?", (learner_id,))
            row = cursor.fetchone()
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Learner get") from exc
    if not row:
        raise NotFoundError("Learner not found", details={"learner_id": learner_id})
    data = _row_dict(row)
    if "id" in data and "learner_id" not in data:
        data["learner_id"] = data["id"]
    return data


@protected_router.post("/sessions")
async def create_session(payload: SessionCreateRequest) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    current_stage_id = None
    with _db_conn() as conn:
        try:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT id FROM learners WHERE id = ?", (payload.learner_id,))
                if cursor.fetchone() is None:
                    raise NotFoundError("Learner not found", details={"learner_id": payload.learner_id})
                cursor.execute("SELECT id FROM missions WHERE id = ?", (payload.mission_id,))
                if cursor.fetchone() is None:
                    raise NotFoundError("Mission not found", details={"mission_id": payload.mission_id})
            except NotFoundError:
                raise
            except sqlite3.OperationalError:
                pass
            spec = runtime.load_mission_spec(conn, payload.mission_id)
            current_stage_id = runtime.first_stage_id(spec)
            cursor.execute(
                """
                INSERT INTO mission_sessions (id, learner_id, mission_id, status, current_stage_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, payload.learner_id, payload.mission_id, runtime.SESSION_ACTIVE, current_stage_id),
            )
            runtime.ensure_ready_attempt(conn, session_id, runtime.get_stage(spec, current_stage_id))
            conn.commit()
        except AppError:
            conn.rollback()
            raise
        except Exception as exc:
            conn.rollback()
            raise _map_sqlite_write_error(exc, "Session create") from exc
    return {
        "session_id": session_id,
        "mission_id": payload.mission_id,
        "learner_id": payload.learner_id,
        "status": runtime.SESSION_ACTIVE,
        "current_stage_id": current_stage_id,
    }


@protected_router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            session = runtime.load_session(conn, session_id)
            return runtime.enrich_session(conn, session)
        except AppError:
            raise
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Session get") from exc


@protected_router.get("/missions")
async def list_missions() -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT id, title, phase_id, order_index FROM missions ORDER BY order_index"
                )
            except sqlite3.OperationalError:
                cursor.execute("SELECT * FROM missions")
            rows = cursor.fetchall()
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Mission list") from exc
    return {"missions": [_row_dict(row) for row in rows]}


@protected_router.get("/missions/{mission_id}")
async def get_mission(mission_id: str) -> Any:
    with _db_conn() as conn:
        try:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT spec_json FROM missions WHERE id = ?", (mission_id,))
                row = cursor.fetchone()
            except sqlite3.OperationalError:
                cursor.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
                row = cursor.fetchone()
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Mission get") from exc
    if not row:
        raise NotFoundError("Mission not found", details={"mission_id": mission_id})
    data = _row_dict(row)
    spec_raw = data.get("spec_json")
    if isinstance(spec_raw, str) and spec_raw:
        try:
            return json.loads(spec_raw)
        except json.JSONDecodeError:
            pass
    return data


@protected_router.post("/artifacts")
async def create_artifact(request: Request) -> dict[str, Any]:
    store = _get_artifact_store()
    content_type = (request.headers.get("content-type") or "").lower()
    payload: bytes
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            payload = await upload.read()
        else:
            raw_b64 = form.get("bytes_b64")
            if not isinstance(raw_b64, str) or not raw_b64:
                raise ValidationAppError("file or bytes_b64 is required")
            payload = _decode_artifact_bytes(raw_b64)
    else:
        try:
            body = await request.json()
        except Exception as exc:
            raise ValidationAppError("JSON body is required") from exc
        parsed = ArtifactCreateRequest.model_validate(body)
        payload = _decode_artifact_bytes(parsed.bytes_b64)
    if not isinstance(payload, (bytes, bytearray)):
        raise ValidationAppError("Artifact payload must be bytes")
    data = bytes(payload)
    artifact_hash = store.put(data)
    return {"artifact_hash": str(artifact_hash), "size": len(data)}


@protected_router.get("/artifacts/{artifact_hash}")
async def get_artifact(artifact_hash: str) -> Response:
    store = _get_artifact_store()
    try:
        data = store.get(artifact_hash)
    except FileNotFoundError as exc:
        raise NotFoundError("Artifact not found", details={"artifact_hash": artifact_hash}) from exc
    except Exception as exc:
        message = str(exc).lower()
        if "not found" in message:
            raise NotFoundError("Artifact not found", details={"artifact_hash": artifact_hash}) from exc
        raise StorageUnavailableError("Failed to read artifact") from exc
    if data is None:
        raise NotFoundError("Artifact not found", details={"artifact_hash": artifact_hash})
    body = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return Response(
        content=bytes(body),
        media_type="application/octet-stream",
        headers={"X-Checksum-SHA256": artifact_hash},
    )


@protected_router.post("/curriculum/packages/load")
async def load_curriculum_package(payload: CurriculumLoadRequest) -> dict[str, Any]:
    package_dir = payload.package_dir
    loaded = False
    result: Any = None
    try:
        mission_loader = importlib.import_module("app.core.mission_loader")
    except ImportError:
        mission_loader = None
    try:
        if mission_loader is not None:
            loaded, result = _call_first(
                mission_loader,
                ("load_package", "load", "install_package"),
                package_dir,
            )
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code or isinstance(exc, ValueError):
            details = dict(getattr(exc, "details", None) or {})
            if code:
                details["code"] = code
            raise ValidationAppError(str(exc), details=details) from exc
        raise
    if not loaded:
        try:
            registry = importlib.import_module("app.core.registry")
        except ImportError:
            registry = None
        if registry is not None:
            loaded, result = _call_first(
                registry,
                ("load_package", "load", "install_package"),
                package_dir,
            )
    if not loaded:
        raise CurriculumUnavailableError("Curriculum loader is not available")
    _maybe_validate_imported_missions(result)
    _register_loaded_package(result)
    return _normalize_package(result)


@protected_router.get("/curriculum/packages")
async def list_curriculum_packages() -> dict[str, Any]:
    listed = False
    result: Any = None
    try:
        registry = importlib.import_module("app.core.registry")
    except ImportError:
        registry = None
    if registry is not None:
        listed, result = _call_first(
            registry,
            ("list_packages", "list_curriculum_packages", "list_all", "list_missions"),
        )
    if not listed:
        try:
            mission_loader = importlib.import_module("app.core.mission_loader")
        except ImportError:
            mission_loader = None
        if mission_loader is not None:
            listed, result = _call_first(
                mission_loader,
                ("list_packages", "list_curriculum_packages", "list_all", "list_missions"),
            )
    if not listed:
        raise CurriculumUnavailableError("Curriculum registry is not available")
    return {"packages": _as_package_list(result)}


@protected_router.post("/system/backup")
async def create_system_backup() -> dict[str, Any]:
    fn = _import_backup_fn("create_backup")
    if fn is None:
        raise StorageUnavailableError("Backup is not available")
    settings = get_settings()
    dest = settings.data_home / "backups"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        archive = fn(dest)
    except TypeError:
        archive = fn()
    except Exception as exc:
        raise StorageUnavailableError("Backup failed") from exc
    path = Path(str(archive))
    return {"backup_id": path.name, "path": str(path)}


@protected_router.post("/system/restore")
async def restore_system_backup(payload: RestoreRequest) -> dict[str, Any]:
    fn = _import_backup_fn("restore_backup")
    if fn is None:
        raise StorageUnavailableError("Restore is not available")
    settings = get_settings()
    archive = payload.path
    if payload.backup_id and not archive:
        archive = str(settings.data_home / "backups" / payload.backup_id)
    if not archive:
        raise ValidationAppError("backup_id or path is required")
    archive_path = Path(archive).expanduser()
    if payload.dest_home:
        dest = Path(payload.dest_home).expanduser().resolve()
    else:
        dest = settings.data_home
        if not _home_empty_of_db_and_artifacts(dest):
            raise ValidationAppError(
                "Restore destination is not empty of db/artifacts; provide dest_home pointing at a clean directory"
            )
    try:
        fn(archive_path, dest)
    except TypeError:
        try:
            fn(archive_path)
        except FileExistsError as exc:
            raise ValidationAppError(str(exc) or "Restore destination is not empty") from exc
        except FileNotFoundError as exc:
            raise NotFoundError("Backup archive not found") from exc
        except Exception as exc:
            raise StorageUnavailableError("Restore failed") from exc
    except FileExistsError as exc:
        raise ValidationAppError(str(exc) or "Restore destination is not empty") from exc
    except FileNotFoundError as exc:
        raise NotFoundError("Backup archive not found") from exc
    except (NotADirectoryError, ValueError) as exc:
        raise ValidationAppError(str(exc)) from exc
    except Exception as exc:
        raise StorageUnavailableError("Restore failed") from exc
    return {"status": "RESTORED", "path": str(archive_path), "dest_home": str(dest)}


@protected_router.post("/sessions/{session_id}/stages/{stage_id}/enter")
async def enter_stage(session_id: str, stage_id: str) -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            return runtime.enter_stage(conn, session_id, stage_id)
        except AppError:
            raise
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Stage enter") from exc


@protected_router.post("/sessions/{session_id}/stages/{stage_id}/predict")
async def predict_stage(session_id: str, stage_id: str, payload: PredictCommitRequest) -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            return runtime.commit_prediction(
                conn,
                session_id,
                stage_id,
                payload.hypothesis,
                payload.expected_values,
            )
        except AppError:
            raise
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Predict-commit") from exc


@protected_router.post("/sessions/{session_id}/stages/{stage_id}/execute")
async def execute_stage(session_id: str, stage_id: str, payload: ExecuteStageRequest) -> JSONResponse:
    with _db_conn() as conn:
        try:
            body = runtime.execute_stage(
                conn,
                session_id,
                stage_id,
                payload.code,
                payload.parameters,
            )
        except AppError:
            raise
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Stage execute") from exc
    status_code = 202 if body.get("status") == "ACCEPTED" else 200
    return JSONResponse(content=body, status_code=status_code)


@protected_router.post("/sessions/{session_id}/stages/{stage_id}/submit")
async def submit_stage(session_id: str, stage_id: str, payload: SubmitStageRequest) -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            return runtime.submit_stage(
                conn,
                session_id,
                stage_id,
                payload.explanation,
                payload.artifacts,
            )
        except AppError:
            raise
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Stage submit") from exc


@protected_router.post("/sessions/{session_id}/gates/evaluate")
async def evaluate_gate(session_id: str) -> dict[str, Any]:
    with _db_conn() as conn:
        try:
            return runtime.evaluate_gate(conn, session_id)
        except AppError:
            raise
        except Exception as exc:
            raise _map_sqlite_write_error(exc, "Gate evaluate") from exc


@protected_router.post("/tutor/chat")
async def tutor_chat(_payload: TutorChatRequest) -> None:
    raise TutorUnavailableError("Tutor is not available in G3")


router.include_router(public_router)
router.include_router(protected_router)
