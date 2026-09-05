from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tools.platform import update as update_cli

REPO_ROOT = update_cli.REPO_ROOT
UPDATE_CLI = Path(update_cli.__file__).resolve()


def _seed_home(data_home: Path) -> str:
    from app.core.artifact_store import ArtifactStore
    from app.db.database import get_connection, init_db
    from app.db.ledger import EventLedger

    init_db()
    (data_home / "config.json").write_text(
        json.dumps({"theme": "pre-update"}),
        encoding="utf-8",
    )
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            ("learner_pre", "preuser", "Pre User"),
        )
        conn.commit()
        EventLedger(conn).append("learner_pre", "note", {"text": "before-update"})
    finally:
        conn.close()
    return ArtifactStore().put(b"pre-update-blob", media_type="text/plain")


def _learner_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT id FROM learners").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def test_pre_update_backup_created_before_simulated_update(data_home: Path) -> None:
    digest = _seed_home(data_home)
    result = update_cli.apply_update(home=data_home, simulate=True)

    assert result.backup.is_file()
    assert result.backup.parent == data_home.resolve() / "backups"
    assert result.backup.name.startswith("backup_pre_update_")
    assert result.backup.name.endswith(".tar.gz")
    pointer = data_home / "backups" / update_cli.PRE_UPDATE_POINTER_NAME
    assert pointer.read_text(encoding="utf-8").strip() == result.backup.name

    live_config = json.loads((data_home / "config.json").read_text(encoding="utf-8"))
    assert live_config["theme"] == "pre-update"
    assert live_config[update_cli.PRE_UPDATE_CONFIG_KEY]["backup"] == result.backup.name
    assert live_config[update_cli.PRE_UPDATE_CONFIG_KEY]["mode"] == "simulate"
    assert update_cli.POST_UPDATE_LEARNER_ID in _learner_ids(data_home / "learningos.db")
    assert result.post_update_artifact is not None

    with tarfile.open(result.backup, "r:gz") as tar:
        names = set(tar.getnames())
        assert "learningos.db" in names
        assert "config.json" in names
        assert any(name.startswith("artifacts/") for name in names)
        archived_config = json.loads(tar.extractfile("config.json").read().decode("utf-8"))
        extract_dir = data_home / "backups" / "_inspect"
        extract_dir.mkdir()
        tar.extract("learningos.db", path=extract_dir)

    assert archived_config["theme"] == "pre-update"
    assert update_cli.PRE_UPDATE_CONFIG_KEY not in archived_config
    assert _learner_ids(extract_dir / "learningos.db") == {"learner_pre"}
    assert digest
    from app.core.artifact_store import ArtifactStore

    assert ArtifactStore().get(digest) == b"pre-update-blob"
    assert ArtifactStore().get(result.post_update_artifact) == b"post-update-artifact"


def test_update_cli_writes_backup_then_simulates(data_home: Path) -> None:
    _seed_home(data_home)
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(data_home)
    env["PYTHONPATH"] = str(REPO_ROOT / "platform" / "backend") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(UPDATE_CLI), "--simulate"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Pre-update backup:" in completed.stdout
    archives = list((data_home / "backups").glob("backup_pre_update_*.tar.gz"))
    assert archives
    assert update_cli.POST_UPDATE_LEARNER_ID in _learner_ids(data_home / "learningos.db")


def test_update_refuses_home_inside_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    inside = REPO_ROOT / "platform" / ".g7-update-home-should-not-exist"
    monkeypatch.setenv("LEARNINGOS_HOME", str(inside))
    try:
        with pytest.raises(update_cli.UpdateError, match="Git worktree"):
            update_cli.apply_update(home=inside, simulate=True)
        assert not inside.exists()
    finally:
        if inside.exists():
            import shutil

            shutil.rmtree(inside, ignore_errors=True)
