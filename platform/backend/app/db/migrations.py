"""Versioned SQLite migrations applied in a single transaction."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.db.database import get_connection, get_data_home, get_db_path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_SCHEMA_MIGRATIONS_SQL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def _discover_migrations() -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    if not MIGRATIONS_DIR.is_dir():
        return found
    for path in MIGRATIONS_DIR.glob("*.sql"):
        prefix = path.name.split("_", 1)[0]
        try:
            version = int(prefix)
        except ValueError as exc:
            raise ValueError(f"migration filename must start with an integer: {path.name}") from exc
        found.append((version, path))
    found.sort(key=lambda item: item[0])
    versions = [version for version, _ in found]
    if len(versions) != len(set(versions)):
        raise ValueError("duplicate migration versions in migrations/")
    return found


def _iter_sql_statements(sql: str):
    buf: list[str] = []
    for line in sql.splitlines():
        if not buf:
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
        buf.append(line)
        candidate = "\n".join(buf)
        if sqlite3.complete_statement(candidate):
            stmt = candidate.strip()
            buf.clear()
            if stmt:
                yield stmt
    leftover = "\n".join(buf).strip()
    if leftover:
        yield leftover


def _is_pragma(statement: str) -> bool:
    return statement.lstrip().upper().startswith("PRAGMA")


def _write_pre_migration_snapshot() -> Path:
    from app.db.backup import create_backup

    dest = get_data_home() / "backups"
    dest.mkdir(parents=True, exist_ok=True)
    return create_backup(dest, label="pre_migration")


def run_migrations(conn: sqlite3.Connection | None = None) -> list[int]:
    """Apply pending numbered SQL files in order inside one transaction.

    If the database file already exists and at least one new migration will
    apply, a pre-migration snapshot is written under $LEARNINGOS_HOME/backups/.
    """
    db_path = get_db_path()
    db_existed = db_path.is_file() and db_path.stat().st_size > 0
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    applied_now: list[int] = []
    previous_isolation = conn.isolation_level
    try:
        conn.isolation_level = None
        conn.execute(_SCHEMA_MIGRATIONS_SQL)
        applied = {
            int(row["version"])
            for row in conn.execute("SELECT version FROM _schema_migrations")
        }
        pending = [(version, path) for version, path in _discover_migrations() if version not in applied]
        if not pending:
            return applied_now
        if db_existed:
            _write_pre_migration_snapshot()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for version, path in pending:
                sql = path.read_text(encoding="utf-8")
                for statement in _iter_sql_statements(sql):
                    if _is_pragma(statement):
                        continue
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO _schema_migrations (version) VALUES (?)",
                    (version,),
                )
                applied_now.append(version)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return applied_now
    finally:
        conn.isolation_level = previous_isolation
        if close_conn:
            conn.close()
