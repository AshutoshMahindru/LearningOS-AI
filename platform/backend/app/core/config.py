from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path


BIND_HOST = "127.0.0.1"
API_PREFIX = "/api/v1"
DATABASE_FILENAME = "learningos.db"
HISTORICAL_WORKER_SOCKET = Path("/tmp/learningos_worker.sock")

_PYTEST_HOME: Path | None = None


def resolve_data_home() -> Path:
    """Resolve LEARNINGOS_HOME. Exact algorithm from the G3 lane contract.

    Pytest sessions without an explicit LEARNINGOS_HOME use a process temp dir so
    the seed TestClient cannot write to the developer's real ~/.learningos.
    """
    raw = os.environ.get("LEARNINGOS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    if "pytest" in sys.modules:
        global _PYTEST_HOME
        if _PYTEST_HOME is None:
            import tempfile

            _PYTEST_HOME = Path(tempfile.mkdtemp(prefix="learningos-pytest-"))
        return _PYTEST_HOME.resolve()
    raw = os.environ.get("LEARNINGOS_HOME") or "~/.learningos"
    return Path(raw).expanduser().resolve()


def resolve_worker_socket(data_home: Path | None = None) -> Path:
    raw = os.environ.get("LEARNINGOS_WORKER_SOCKET")
    if raw:
        return Path(raw).expanduser().resolve()
    home = data_home if data_home is not None else resolve_data_home()
    return (home / "run" / "worker.sock").resolve()


@dataclass(frozen=True)
class Settings:
    data_home: Path
    database_path: Path
    worker_socket: Path
    bind_host: str = BIND_HOST
    api_prefix: str = API_PREFIX


def get_settings() -> Settings:
    data_home = resolve_data_home()
    return Settings(
        data_home=data_home,
        database_path=(data_home / DATABASE_FILENAME).resolve(),
        worker_socket=resolve_worker_socket(data_home),
        bind_host=BIND_HOST,
        api_prefix=API_PREFIX,
    )


def ensure_data_layout(settings: Settings | None = None) -> Settings:
    current = settings or get_settings()
    current.data_home.mkdir(parents=True, exist_ok=True)
    (current.data_home / "artifacts").mkdir(parents=True, exist_ok=True)
    (current.data_home / "backups").mkdir(parents=True, exist_ok=True)
    (current.data_home / "run").mkdir(parents=True, exist_ok=True)
    return current


def public_config() -> dict[str, str]:
    """Non-secret runtime configuration. Never includes provider keys or tokens."""
    settings = get_settings()
    return {
        "data_home": str(settings.data_home),
        "database_path": str(settings.database_path),
        "worker_socket": str(settings.worker_socket),
        "bind_host": settings.bind_host,
        "api_prefix": settings.api_prefix,
    }


def _interpret_worker_health(result: object) -> bool | None:
    if result is None:
        return False
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        error = result.get("error")
        if error is not None:
            text = str(error).lower()
            if "not found" in text or "not running" in text:
                return False
        if "alive" in result:
            return bool(result.get("alive"))
        if result.get("status") in {"ok", "HEALTHY", "alive"}:
            return True
        if "pid" in result:
            return True
        return True
    return bool(result)


def _unix_socket_probe(path: Path) -> bool:
    try:
        if not path.exists():
            return False
    except OSError:
        return False
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.4)
        sock.connect(os.fspath(path))
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def worker_alive() -> bool:
    """Probe the worker via WorkerClient.health() or a unix-socket connect."""
    settings = get_settings()
    path = settings.worker_socket
    try:
        from app.core.worker_client import WorkerClient

        try:
            client = WorkerClient(socket_path=str(path))
        except TypeError:
            client = WorkerClient()
        health = getattr(client, "health", None)
        if callable(health):
            interpreted = _interpret_worker_health(health())
            if interpreted is not None:
                return interpreted
    except Exception:
        pass
    if _unix_socket_probe(path):
        return True
    # Historical WP-140 path only when neither env var is set.
    if os.environ.get("LEARNINGOS_HOME") or os.environ.get("LEARNINGOS_WORKER_SOCKET"):
        return False
    try:
        if path.resolve() != HISTORICAL_WORKER_SOCKET.resolve():
            return _unix_socket_probe(HISTORICAL_WORKER_SOCKET)
    except OSError:
        return False
    return False


def storage_honors_settings(dbmod: object, settings: Settings) -> bool:
    get_data_home = getattr(dbmod, "get_data_home", None)
    if callable(get_data_home):
        try:
            return Path(get_data_home()).expanduser().resolve() == settings.data_home
        except Exception:
            return False
    db_path = getattr(dbmod, "DB_PATH", None)
    if db_path is not None:
        try:
            return Path(db_path).expanduser().resolve() == settings.database_path
        except Exception:
            return False
    get_database_path = getattr(dbmod, "get_database_path", None)
    if callable(get_database_path):
        try:
            return Path(get_database_path()).expanduser().resolve() == settings.database_path
        except Exception:
            return False
    return True


def attempt_storage_init(settings: Settings | None = None) -> None:
    current = settings or get_settings()
    try:
        from app.db import database as dbmod
    except ImportError:
        return
    if not storage_honors_settings(dbmod, current):
        return
    init_db = getattr(dbmod, "init_db", None)
    run_migrations = getattr(dbmod, "run_migrations", None)
    try:
        if callable(init_db):
            init_db()
        elif callable(run_migrations):
            run_migrations()
    except Exception:
        return
