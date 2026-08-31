from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from app.core.artifact_store import ArtifactStore
from app.db.backup import create_backup, restore_backup
from app.db.database import get_connection, init_db
from app.db.ledger import EventLedger


def test_backup_and_restore_into_clean_home(data_home: Path, tmp_path: Path, monkeypatch) -> None:
    init_db()
    (data_home / "config.json").write_text(
        json.dumps({"theme": "dark"}),
        encoding="utf-8",
    )
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            ("learner_restore", "restorer", "Restorer"),
        )
        conn.commit()
        EventLedger(conn).append("learner_restore", "note", {"text": "durable"})
    finally:
        conn.close()

    digest = ArtifactStore().put(b"blob-to-restore", media_type="text/plain")
    archive = create_backup()
    assert archive.is_file()
    assert archive.parent == data_home / "backups"
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "learningos.db" in names
    assert "config.json" in names
    assert any(name.startswith("artifacts/") for name in names)

    dest = tmp_path / "restored-home"
    restore_backup(archive, dest)
    assert (dest / "learningos.db").is_file()
    assert json.loads((dest / "config.json").read_text(encoding="utf-8"))["theme"] == "dark"

    monkeypatch.setenv("LEARNINGOS_HOME", str(dest))
    conn = get_connection()
    try:
        learner = conn.execute(
            "SELECT username FROM learners WHERE id = ?",
            ("learner_restore",),
        ).fetchone()
        assert learner["username"] == "restorer"
        event = conn.execute(
            "SELECT event_type, payload_json FROM learning_events WHERE learner_id = ?",
            ("learner_restore",),
        ).fetchone()
        assert event["event_type"] == "note"
        assert json.loads(event["payload_json"])["text"] == "durable"
    finally:
        conn.close()

    assert ArtifactStore().get(digest) == b"blob-to-restore"


def test_restore_rejects_non_empty_dest(data_home: Path, tmp_path: Path) -> None:
    init_db()
    archive = create_backup()
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "stale").write_text("nope", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        restore_backup(archive, dest)
