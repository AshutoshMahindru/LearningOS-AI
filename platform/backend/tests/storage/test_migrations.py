from __future__ import annotations

from pathlib import Path

from app.db.database import get_connection, init_db
from app.db.migrations import run_migrations
from app.db.seed import seed_database


def _applied_versions(conn) -> list[int]:
    rows = conn.execute("SELECT version FROM _schema_migrations ORDER BY version").fetchall()
    return [int(row["version"]) for row in rows]


def test_migrations_apply_and_are_idempotent(data_home: Path) -> None:
    init_db()
    conn = get_connection()
    try:
        versions = _applied_versions(conn)
        assert versions == [1, 2]
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "learners" in tables
        assert "learning_events" in tables
        assert "_schema_migrations" in tables
        assert "missions" in tables
    finally:
        conn.close()

    again = run_migrations()
    assert again == []
    conn = get_connection()
    try:
        assert _applied_versions(conn) == [1, 2]
    finally:
        conn.close()


def test_pending_migration_writes_pre_migration_snapshot(
    data_home: Path, tmp_path: Path, monkeypatch
) -> None:
    import app.db.migrations as migrations_mod

    init_db()
    alt = tmp_path / "alt-migrations"
    alt.mkdir()
    for src in Path(migrations_mod.MIGRATIONS_DIR).glob("*.sql"):
        (alt / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (alt / "0003_probe.sql").write_text(
        "CREATE TABLE IF NOT EXISTS migration_probe (id INTEGER PRIMARY KEY);\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(migrations_mod, "MIGRATIONS_DIR", alt)

    applied = run_migrations()
    assert applied == [3]
    snapshots = list((data_home / "backups").glob("backup_pre_migration_*.tar.gz"))
    assert snapshots, "expected a pre-migration snapshot in backups/"

    conn = get_connection()
    try:
        assert _applied_versions(conn) == [1, 2, 3]
        probe = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_probe'"
        ).fetchone()
        assert probe is not None
    finally:
        conn.close()


def test_seed_is_noop(data_home: Path) -> None:
    init_db()
    seed_database()
    conn = get_connection()
    try:
        packages = conn.execute("SELECT COUNT(*) AS n FROM curriculum_packages").fetchone()["n"]
        missions = conn.execute("SELECT COUNT(*) AS n FROM missions").fetchone()["n"]
        learners = conn.execute("SELECT COUNT(*) AS n FROM learners").fetchone()["n"]
        assert packages == 0
        assert missions == 0
        assert learners == 0
    finally:
        conn.close()
