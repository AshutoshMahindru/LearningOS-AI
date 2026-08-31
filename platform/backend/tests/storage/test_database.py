from __future__ import annotations

from pathlib import Path

import pytest




def test_data_home_uses_env_and_expands(data_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.db.database import get_data_home, get_db_path

    assert get_data_home() == data_home.resolve()
    assert get_db_path() == data_home.resolve() / "learningos.db"

    tilde_target = tmp_path / "from-tilde"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LEARNINGOS_HOME", "~/from-tilde")
    assert get_data_home() == tilde_target.resolve()


def test_init_creates_home_and_wal_pragmas(data_home: Path) -> None:
    from app.db.database import get_connection, get_db_path, init_db

    init_db()
    assert data_home.is_dir()
    assert get_db_path().is_file()
    assert (data_home / "artifacts").is_dir()
    assert (data_home / "backups").is_dir()

    conn = get_connection()
    try:
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
        busy = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert str(journal).lower() == "wal"
        assert int(foreign_keys) == 1
        assert int(synchronous) == 1  # NORMAL
        assert int(busy) == 5000
        assert conn.row_factory is __import__("sqlite3").Row
    finally:
        conn.close()


def test_learner_survives_reconnect(data_home: Path) -> None:
    from app.db.database import get_connection, init_db

    init_db()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            ("learner_1", "alice", "Alice"),
        )
        conn.commit()
    finally:
        conn.close()

    conn2 = get_connection()
    try:
        row = conn2.execute(
            "SELECT username, display_name FROM learners WHERE id = ?",
            ("learner_1",),
        ).fetchone()
        assert row["username"] == "alice"
        assert row["display_name"] == "Alice"
    finally:
        conn2.close()


def test_init_refuses_home_inside_repo(monkeypatch: pytest.MonkeyPatch, worktree_root: Path) -> None:
    from app.db.database import init_db

    bad_home = worktree_root / "platform" / "backend" / "nested-learningos-home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(bad_home))
    with pytest.raises(RuntimeError, match="must not be inside the Git worktree"):
        init_db()
    assert not bad_home.exists()
    assert not (worktree_root / "learningos.db").exists()
    assert not (worktree_root / ".learningos").exists()
