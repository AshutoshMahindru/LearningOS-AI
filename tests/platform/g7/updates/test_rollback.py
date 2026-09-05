from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tools.platform import rollback as rollback_cli
from tools.platform import state_guard
from tools.platform import update as update_cli

REPO_ROOT = update_cli.REPO_ROOT
ROLLBACK_CLI = Path(rollback_cli.__file__).resolve()
STATE_GUARD_CLI = Path(state_guard.__file__).resolve()


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


def _assert_state_guard_clean() -> None:
    violations = state_guard.find_violations(REPO_ROOT)
    assert violations == [], [path.as_posix() for path in violations]
    completed = subprocess.run(
        [sys.executable, str(STATE_GUARD_CLI), "--repo", str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_rollback_restores_clean_dest_and_refuses_live_home(data_home: Path, dest_home: Path) -> None:
    digest = _seed_home(data_home)
    update = update_cli.apply_update(home=data_home, simulate=True)
    assert update_cli.POST_UPDATE_LEARNER_ID in _learner_ids(data_home / "learningos.db")

    with pytest.raises(rollback_cli.RollbackError, match="live data home"):
        rollback_cli.rollback_to_dest(data_home, live_home=data_home)

    result = rollback_cli.rollback_to_dest(dest_home, live_home=data_home)
    assert result.dest_home == dest_home.resolve()
    assert (dest_home / "learningos.db").is_file()
    assert (dest_home / "artifacts").is_dir()
    restored_config = json.loads((dest_home / "config.json").read_text(encoding="utf-8"))
    assert restored_config["theme"] == "pre-update"
    assert update_cli.PRE_UPDATE_CONFIG_KEY not in restored_config
    assert _learner_ids(dest_home / "learningos.db") == {"learner_pre"}
    assert (dest_home / "artifacts" / "sha256" / digest[:2] / digest[2:]).is_file()
    assert not (
        dest_home / "artifacts" / "sha256" / update.post_update_artifact[:2] / update.post_update_artifact[2:]
    ).exists()
    assert "export LEARNINGOS_HOME=" in result.instructions
    assert str(dest_home.resolve()) in result.instructions
    assert "live data home is not a valid restore target" in result.instructions.lower()
    _assert_state_guard_clean()


def test_rollback_refuses_dest_inside_worktree(data_home: Path) -> None:
    _seed_home(data_home)
    update_cli.apply_update(home=data_home, simulate=True)
    dest = REPO_ROOT / "platform" / ".g7-rollback-dest-should-not-exist"
    try:
        with pytest.raises(rollback_cli.RollbackError, match="Git worktree"):
            rollback_cli.rollback_to_dest(dest, live_home=data_home)
        assert not dest.exists()
    finally:
        if dest.exists():
            import shutil

            shutil.rmtree(dest, ignore_errors=True)
    _assert_state_guard_clean()


def test_rollback_refuses_non_empty_dest(data_home: Path, dest_home: Path) -> None:
    _seed_home(data_home)
    update_cli.apply_update(home=data_home, simulate=True)
    (dest_home / "stale").write_text("nope", encoding="utf-8")
    with pytest.raises(rollback_cli.RollbackError, match="not empty"):
        rollback_cli.rollback_to_dest(dest_home, live_home=data_home)
    assert (dest_home / "stale").read_text(encoding="utf-8") == "nope"
    assert not (dest_home / "learningos.db").exists()


def test_rollback_cli_restores_and_documents_learningos_home(
    data_home: Path, dest_home: Path
) -> None:
    _seed_home(data_home)
    update_cli.apply_update(home=data_home, simulate=True)
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(data_home)
    env["PYTHONPATH"] = str(REPO_ROOT / "platform" / "backend") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(ROLLBACK_CLI), "--dest-home", str(dest_home)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert (dest_home / "learningos.db").is_file()
    assert (dest_home / "artifacts").is_dir()
    assert f"export LEARNINGOS_HOME={dest_home.resolve()}" in completed.stdout
    assert "./start.sh" in completed.stdout
    help_completed = subprocess.run(
        [sys.executable, str(ROLLBACK_CLI), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_completed.returncode == 0, help_completed.stderr
    assert "LEARNINGOS_HOME" in help_completed.stdout
    assert "dest_home" in help_completed.stdout or "dest-home" in help_completed.stdout
    _assert_state_guard_clean()
