from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from app.core.mission_loader import (
    G3_FIXTURE_PACKAGE_ID,
    G3_FIXTURE_VERSION,
    CurriculumPackageError,
    fixture_package_path,
    load_package,
)
from app.core.registry import CurriculumRegistry

SCHEMA_SQL = """
CREATE TABLE curriculum_packages (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    git_commit_sha TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    installed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE missions (
    id TEXT PRIMARY KEY,
    package_id TEXT NOT NULL,
    title TEXT NOT NULL,
    phase_id TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    schema_version TEXT NOT NULL DEFAULT "v1",
    spec_json TEXT NOT NULL,
    FOREIGN KEY (package_id) REFERENCES curriculum_packages(id) ON DELETE CASCADE
);
"""


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    return home


def test_load_fixture_package():
    package = load_package(fixture_package_path())
    assert package.id == G3_FIXTURE_PACKAGE_ID
    assert package.version == G3_FIXTURE_VERSION
    assert package.identity == (G3_FIXTURE_PACKAGE_ID, G3_FIXTURE_VERSION)
    assert package.digest
    assert len(package.missions) == 1
    mission = package.missions[0]
    assert mission["id"] == "g3.fixture.orientation"
    assert mission["title"]
    assert mission["stages"]
    assert mission["id"] != "M01"
    assert "M01" not in mission["id"]


def test_reject_missing_manifest(tmp_path):
    dest = tmp_path / "empty-pkg"
    dest.mkdir()
    with pytest.raises(CurriculumPackageError) as raised:
        load_package(dest)
    assert raised.value.code == "MISSING_MANIFEST"


def test_reject_truncated_manifest(tmp_path):
    src = fixture_package_path()
    dest = tmp_path / "truncated"
    shutil.copytree(src, dest)
    (dest / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(CurriculumPackageError) as raised:
        load_package(dest)
    assert raised.value.code == "INVALID_MANIFEST"


def test_reject_tampered_digest(tmp_path):
    src = fixture_package_path()
    dest = tmp_path / "tampered-digest"
    shutil.copytree(src, dest)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dest / "SHA256SUMS").unlink()
    with pytest.raises(CurriculumPackageError) as raised:
        load_package(dest)
    assert raised.value.code == "BAD_DIGEST"


def test_reject_tampered_mission_file(tmp_path):
    src = fixture_package_path()
    dest = tmp_path / "tampered-mission"
    shutil.copytree(src, dest)
    mission_path = dest / "missions" / "g3.fixture.orientation.json"
    spec = json.loads(mission_path.read_text(encoding="utf-8"))
    spec["title"] = "tampered"
    mission_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (dest / "SHA256SUMS").unlink()
    with pytest.raises(CurriculumPackageError) as raised:
        load_package(dest)
    assert raised.value.code == "BAD_DIGEST"


def test_registry_records_version_identity_in_memory():
    package = load_package(fixture_package_path())
    registry = CurriculumRegistry()
    returned = registry.register_package(package)
    assert returned is package
    assert registry.identities() == [(G3_FIXTURE_PACKAGE_ID, G3_FIXTURE_VERSION)]
    assert registry.get_package(G3_FIXTURE_PACKAGE_ID, G3_FIXTURE_VERSION) is package


def test_registry_persists_when_connection_available():
    package = load_package(fixture_package_path())
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    registry = CurriculumRegistry(connection_factory=lambda: conn)
    registry.register_package(package)
    row = conn.execute(
        "SELECT id, version, git_commit_sha FROM curriculum_packages WHERE id = ?",
        (G3_FIXTURE_PACKAGE_ID,),
    ).fetchone()
    assert row["id"] == G3_FIXTURE_PACKAGE_ID
    assert row["version"] == G3_FIXTURE_VERSION
    assert row["git_commit_sha"] == package.digest
    assert row["git_commit_sha"] != "HEAD"
    mission = conn.execute(
        "SELECT id, package_id, title FROM missions WHERE id = ?",
        ("g3.fixture.orientation",),
    ).fetchone()
    assert mission["package_id"] == G3_FIXTURE_PACKAGE_ID
    assert mission["title"] == "Orientation"


def test_register_does_not_write_developer_home(isolated_home):
    package = load_package(fixture_package_path())
    registry = CurriculumRegistry()
    real_home_db = Path.home() / ".learningos" / "learningos.db"
    before = real_home_db.stat().st_mtime_ns if real_home_db.exists() else None
    registry.register_package(package)
    # After 11D, probing get_connection() may create a db under LEARNINGOS_HOME.
    # The lane contract is that the developer home must not be used.
    if before is None:
        assert not real_home_db.exists()
    else:
        after = real_home_db.stat().st_mtime_ns
        assert after == before
    assert isolated_home.resolve() != (Path.home() / ".learningos").resolve()
