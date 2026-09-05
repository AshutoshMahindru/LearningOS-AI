from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

V2_FREEZE_SHA = "35293455ff769855014a49fa56315b22829e74b1"


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools" / "platform").is_dir() and (
            candidate / "platform" / "backend" / "app"
        ).is_dir():
            return candidate
    raise RuntimeError("could not locate LearningOS repository root")


REPO_ROOT = _repo_root()
for _candidate in (
    str(REPO_ROOT / "tools" / "platform"),
    str(REPO_ROOT / "platform" / "backend"),
    str(REPO_ROOT),
):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import state_guard  # noqa: E402
import v2_migrate  # noqa: E402

MIGRATE_PY = REPO_ROOT / "tools" / "platform" / "v2_migrate.py"
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
POPULATED = FIXTURE_ROOT / "populated"
MULTI = FIXTURE_ROOT / "multi"
GARBAGE = FIXTURE_ROOT / "garbage"


def _fingerprint(root: Path) -> list[tuple[str, str, int, str]]:
    if not root.exists():
        return []
    rows: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file():
            payload = path.read_bytes()
            rows.append((rel, "file", len(payload), hashlib.sha256(payload).hexdigest()))
        elif path.is_dir():
            rows.append((rel, "dir", 0, ""))
    return rows


def _dir_mtime(path: Path) -> int | None:
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


def _cli(args: list[str], *, env: dict[str, str] | None = None, home: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if env:
        environment.update(env)
    if home is not None:
        environment["LEARNINGOS_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, str(MIGRATE_PY), *args],
        cwd=str(REPO_ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _seed_v3_learner(home: Path, *, learner_id: str, username: str, display_name: str) -> None:
    os.environ["LEARNINGOS_HOME"] = str(home)
    if str(REPO_ROOT / "platform" / "backend") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "platform" / "backend"))
    from app.db.database import get_connection, init_db

    init_db()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            (learner_id, username, display_name),
        )
        conn.commit()
    finally:
        conn.close()


def _connect(home: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(home / "learningos.db"))
    conn.row_factory = sqlite3.Row
    return conn


def test_dry_run_populated_counts() -> None:
    inspection = v2_migrate.inspect_source(POPULATED)
    assert inspection.learners == 1
    assert inspection.progress == 4
    assert inspection.artifacts == 2
    assert inspection.bundles[0].learner_id == "default"


def test_dry_run_multi_learner_counts() -> None:
    inspection = v2_migrate.inspect_source(MULTI)
    assert inspection.learners == 2
    assert {bundle.learner_id for bundle in inspection.bundles} == {"alice", "bob"}
    assert inspection.progress == 2
    assert inspection.artifacts == 0


def test_dry_run_does_not_mutate_dest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "learningos-home"
    dest.mkdir()
    marker = dest / "keep-me.txt"
    marker.write_text("untouched\n", encoding="utf-8")
    monkeypatch.setenv("LEARNINGOS_HOME", str(dest))
    before = _fingerprint(dest)
    mtime = _dir_mtime(dest)
    marker_mtime = marker.stat().st_mtime_ns

    code = v2_migrate.main(["--dry-run", "--source", str(POPULATED), "--home", str(dest)])

    assert code == 0
    assert dest.is_dir()
    assert not (dest / "learningos.db").exists()
    assert not (dest / "artifacts").exists()
    assert not (dest / "backups").exists()
    assert _fingerprint(dest) == before
    assert _dir_mtime(dest) == mtime
    assert marker.stat().st_mtime_ns == marker_mtime
    assert marker.read_text(encoding="utf-8") == "untouched\n"


def test_dry_run_does_not_create_missing_dest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "missing-home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(dest))
    code = v2_migrate.main(["--dry-run", "--source", str(POPULATED), "--home", str(dest)])
    assert code == 0
    assert not dest.exists()


def test_dry_run_garbage_exits_nonzero(tmp_path: Path) -> None:
    dest = tmp_path / "should-stay-missing"
    code = v2_migrate.main(["--dry-run", "--source", str(GARBAGE), "--home", str(dest)])
    assert code == 1
    assert not dest.exists()

    empty = tmp_path / "empty-dir"
    empty.mkdir()
    code = v2_migrate.main(["--dry-run", "--source", str(empty)])
    assert code == 1


def test_cli_dry_run_populated() -> None:
    completed = _cli(["--dry-run", "--source", str(POPULATED)])
    assert completed.returncode == 0, completed.stderr
    assert "learners: 1" in completed.stdout
    assert "progress: 4" in completed.stdout
    assert "artifacts: 2" in completed.stdout


def test_import_is_additive_existing_learner_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "learningos-home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    _seed_v3_learner(home, learner_id="learner_v3", username="alice", display_name="Alice Original")

    result = v2_migrate.import_source(POPULATED, home, force=False, repo_root=REPO_ROOT)

    assert result.id_map["default"] == "v2:default"
    conn = _connect(home)
    try:
        rows = {row["id"]: row for row in conn.execute("SELECT id, username, display_name FROM learners")}
        assert rows["learner_v3"]["username"] == "alice"
        assert rows["learner_v3"]["display_name"] == "Alice Original"
        assert rows["v2:default"]["display_name"] == "V2 Prototype Learner"
        mastery = conn.execute(
            "SELECT level FROM competency_mastery WHERE learner_id = ? AND competency_id = ?",
            ("v2:default", "comp.ml.system_mapping"),
        ).fetchone()
        assert mastery is not None and int(mastery["level"]) == 3
        events = [
            row["event_type"]
            for row in conn.execute(
                "SELECT event_type FROM learning_events WHERE learner_id = ? ORDER BY rowid",
                ("v2:default",),
            )
        ]
        assert "v2_import" in events
        assert "v2_progress" in events
        assert events.count("v2_session") == 2
        assert events.count("v2_evidence") == 2
    finally:
        conn.close()

    artifact = (POPULATED / "tracking" / "artifacts" / "m01_system_map.txt").read_bytes()
    digest = hashlib.sha256(artifact).hexdigest()
    stored = home / "artifacts" / "sha256" / digest[:2] / digest[2:]
    assert stored.is_file()
    assert stored.read_bytes() == artifact


def test_second_import_without_force_mints_new_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "learningos-home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    first = v2_migrate.import_source(POPULATED, home, force=False, repo_root=REPO_ROOT)
    conn = _connect(home)
    try:
        conn.execute(
            "UPDATE learners SET display_name = ? WHERE id = ?",
            ("Edited In V3", first.id_map["default"]),
        )
        conn.commit()
    finally:
        conn.close()

    second = v2_migrate.import_source(POPULATED, home, force=False, repo_root=REPO_ROOT)
    assert second.id_map["default"] != first.id_map["default"]
    assert second.id_map["default"].startswith("v2:default:")
    assert not second.overwritten_ids

    conn = _connect(home)
    try:
        names = {
            row["id"]: row["display_name"]
            for row in conn.execute("SELECT id, display_name FROM learners")
        }
        assert names[first.id_map["default"]] == "Edited In V3"
        assert names[second.id_map["default"]] == "V2 Prototype Learner"
        assert len(names) == 2
    finally:
        conn.close()


def test_force_overwrites_after_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "learningos-home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    first = v2_migrate.import_source(POPULATED, home, force=False, repo_root=REPO_ROOT)
    conn = _connect(home)
    try:
        conn.execute(
            "UPDATE learners SET display_name = ? WHERE id = ?",
            ("Edited In V3", first.id_map["default"]),
        )
        conn.commit()
    finally:
        conn.close()

    forced = v2_migrate.import_source(POPULATED, home, force=True, repo_root=REPO_ROOT)
    assert forced.id_map["default"] == "v2:default"
    assert forced.overwritten_ids == ["v2:default"]
    assert forced.snapshot is not None
    assert forced.snapshot.is_file()
    assert forced.snapshot.parent == home / "backups"
    assert "pre_v2_import" in forced.snapshot.name

    conn = _connect(home)
    try:
        row = conn.execute(
            "SELECT display_name FROM learners WHERE id = ?",
            ("v2:default",),
        ).fetchone()
        assert row["display_name"] == "V2 Prototype Learner"
    finally:
        conn.close()


def test_import_refuses_worktree_home(monkeypatch: pytest.MonkeyPatch) -> None:
    dest = REPO_ROOT / f".v2-migrate-test-home-{os.getpid()}"
    monkeypatch.setenv("LEARNINGOS_HOME", str(dest))
    try:
        with pytest.raises(v2_migrate.V2MigrateError, match="must not be inside the Git worktree"):
            v2_migrate.import_source(POPULATED, dest, repo_root=REPO_ROOT)
        code = v2_migrate.main(["--source", str(POPULATED), "--home", str(dest)])
        assert code == 1
        assert not dest.exists()
        assert not (REPO_ROOT / "learningos.db").exists()
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_import_leaves_worktree_clean_of_learner_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "learningos-home"
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    markers_before = {
        REPO_ROOT / "learningos.db": (REPO_ROOT / "learningos.db").exists(),
        REPO_ROOT / ".learningos": (REPO_ROOT / ".learningos").exists(),
    }
    before_violations = state_guard.find_violations(REPO_ROOT)

    v2_migrate.import_source(POPULATED, home, repo_root=REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "platform" / "state_guard.py"), "--repo", str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PASSED" in completed.stdout
    assert state_guard.find_violations(REPO_ROOT) == before_violations
    for path, existed in markers_before.items():
        if not existed:
            assert not path.exists()
    with pytest.raises(ValueError):
        home.resolve().relative_to(REPO_ROOT.resolve())
    assert (home / "learningos.db").is_file()


def test_cli_import_multi(tmp_path: Path) -> None:
    home = tmp_path / "learningos-home"
    completed = _cli(["--source", str(MULTI), "--home", str(home)], home=home)
    assert completed.returncode == 0, completed.stderr
    assert "learners: 2" in completed.stdout
    conn = _connect(home)
    try:
        ids = {row["id"] for row in conn.execute("SELECT id FROM learners")}
        assert ids == {"v2:alice", "v2:bob"}
    finally:
        conn.close()


def test_dry_run_against_freeze_commit(tmp_path: Path) -> None:
    if not v2_migrate.freeze_commit_available(REPO_ROOT):
        pytest.skip(f"freeze commit {V2_FREEZE_SHA} is not in this clone")
    dest = tmp_path / "trap-home"
    dest.mkdir()
    before = _fingerprint(dest)
    tracking = v2_migrate.extract_freeze_tracking(tmp_path / "freeze", repo=REPO_ROOT)
    inspection = v2_migrate.inspect_source(tracking)
    assert inspection.learners == 1
    assert inspection.bundles[0].learner_id == "default"
    assert inspection.progress == 0
    assert inspection.artifacts == 0

    completed = _cli(["--dry-run", "--from-freeze", "--home", str(dest)])
    assert completed.returncode == 0, completed.stderr
    assert "learners: 1" in completed.stdout
    assert "progress: 0" in completed.stdout
    assert "artifacts: 0" in completed.stdout
    assert _fingerprint(dest) == before
    assert not (dest / "learningos.db").exists()


def test_freeze_tag_points_at_pinned_sha() -> None:
    completed = subprocess.run(
        ["git", "rev-parse", f"{v2_migrate.V2_FREEZE_TAG}^{{commit}}"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("v2-prototype-freeze tag is not present locally")
    assert completed.stdout.strip() == V2_FREEZE_SHA
