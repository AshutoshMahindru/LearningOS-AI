"""In-memory (and optional SQLite) curriculum package registry."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.mission_loader import CurriculumPackage

ConnectionFactory = Callable[[], Any]


def _connection_is_safe(db_path: Path, env_home: Path) -> bool:
    try:
        db_path.expanduser().resolve().relative_to(env_home)
        return True
    except ValueError:
        return False


def _try_get_connection() -> Any | None:
    """Use app.db.database.get_connection only when it honors LEARNINGOS_HOME."""
    env_home_raw = os.environ.get("LEARNINGOS_HOME")
    if not env_home_raw:
        return None
    env_home = Path(env_home_raw).expanduser().resolve()
    try:
        from app.db import database as dbmod
    except ImportError:
        return None
    get_connection = getattr(dbmod, "get_connection", None)
    if get_connection is None:
        return None
    db_path = getattr(dbmod, "DB_PATH", None)
    if db_path is not None and not _connection_is_safe(Path(db_path), env_home):
        return None
    try:
        conn = get_connection()
    except Exception:
        return None
    try:
        conn.execute("SELECT 1 FROM curriculum_packages LIMIT 0")
        conn.execute("SELECT 1 FROM missions LIMIT 0")
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn


def _persist(conn: Any, package: CurriculumPackage) -> None:
    conn.execute(
        """
        INSERT INTO curriculum_packages (id, version, git_commit_sha, manifest_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            version = excluded.version,
            git_commit_sha = excluded.git_commit_sha,
            manifest_json = excluded.manifest_json
        """,
        (
            package.id,
            package.version,
            package.digest,
            json.dumps(package.manifest, sort_keys=True, separators=(",", ":")),
        ),
    )
    for index, spec in enumerate(package.missions):
        phase = spec.get("phase") if isinstance(spec.get("phase"), dict) else {}
        phase_id = str(phase.get("id") or "g3.fixture")
        title = str(spec.get("title") or spec["id"])
        order_index = spec.get("order_index", index + 1)
        schema_version = str(spec.get("schema_version") or package.version)
        conn.execute(
            """
            INSERT INTO missions (id, package_id, title, phase_id, order_index, schema_version, spec_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                package_id = excluded.package_id,
                title = excluded.title,
                phase_id = excluded.phase_id,
                order_index = excluded.order_index,
                schema_version = excluded.schema_version,
                spec_json = excluded.spec_json
            """,
            (
                spec["id"],
                package.id,
                title,
                phase_id,
                int(order_index),
                schema_version,
                json.dumps(spec, sort_keys=True, separators=(",", ":")),
            ),
        )
    conn.commit()


class CurriculumRegistry:
    def __init__(self, connection_factory: ConnectionFactory | None = None) -> None:
        self._packages: dict[tuple[str, str], CurriculumPackage] = {}
        self._connection_factory = connection_factory

    def register_package(self, package: CurriculumPackage) -> CurriculumPackage:
        self._packages[package.identity] = package
        conn: Any | None = None
        close = False
        if self._connection_factory is not None:
            conn = self._connection_factory()
        else:
            conn = _try_get_connection()
            close = conn is not None
        if conn is None:
            return package
        try:
            _persist(conn, package)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            if close:
                try:
                    conn.close()
                except Exception:
                    pass
        return package

    def get_package(self, package_id: str, version: str | None = None) -> CurriculumPackage | None:
        if version is not None:
            return self._packages.get((package_id, version))
        matches = [pkg for (pid, _ver), pkg in self._packages.items() if pid == package_id]
        return matches[-1] if matches else None

    def list_packages(self) -> list[CurriculumPackage]:
        return list(self._packages.values())

    def identities(self) -> list[tuple[str, str]]:
        return list(self._packages.keys())


_default_registry = CurriculumRegistry()


def register_package(package: CurriculumPackage) -> CurriculumPackage:
    return _default_registry.register_package(package)


def get_default_registry() -> CurriculumRegistry:
    return _default_registry
