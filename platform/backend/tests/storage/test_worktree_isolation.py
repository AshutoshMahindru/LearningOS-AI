from __future__ import annotations

from pathlib import Path

from app.core.artifact_store import ArtifactStore
from app.db.backup import create_backup
from app.db.database import get_connection, init_db
from app.db.ledger import EventLedger


def test_operations_do_not_write_learner_state_into_git_worktree(
    data_home: Path, worktree_root: Path
) -> None:
    root = worktree_root
    init_db()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            ("learner_iso", "iso", "Iso"),
        )
        conn.commit()
        EventLedger(conn).append("learner_iso", "started", {"ok": True})
    finally:
        conn.close()
    ArtifactStore().put(b"worktree-guard")
    create_backup()

    assert data_home.is_relative_to(Path("/"))  # resolved tmp path
    assert not data_home.is_relative_to(root)
    assert not (root / "learningos.db").exists()
    assert not (root / ".learningos").exists()
    assert not (root / "platform" / "learningos.db").exists()
    assert not (root / "platform" / "backend" / "learningos.db").exists()
    assert not (root / "platform" / "backend" / ".learningos").exists()
    assert not (root / "platform" / ".learningos").exists()
    assert data_home.exists()
    assert (data_home / "learningos.db").is_file()
