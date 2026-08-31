"""Portable backup and restore of LEARNINGOS_HOME (db, artifacts, config)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db.database import get_connection, get_data_home, get_db_path


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.getmembers():
        name = member.name.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        target = (dest / member.name).resolve()
        if target != dest and not str(target).startswith(str(dest) + os.sep):
            raise ValueError(f"unsafe archive member: {member.name}")
    archive.extractall(dest, filter="data")


def _verify_artifact_checksums(artifacts_root: Path) -> None:
    sha_root = artifacts_root / "sha256"
    if not sha_root.is_dir():
        return
    for path in sha_root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        expected = path.parent.name + path.name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"restored artifact checksum mismatch at {path}")


def create_backup(dest_dir: Path | None = None, *, label: str | None = None) -> Path:
    """Checkpoint WAL and write a tar.gz of learningos.db, artifacts/, and config.json."""
    home = get_data_home()
    dest_dir = Path(dest_dir) if dest_dir is not None else home / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = _utc_stamp()
    ident = uuid.uuid4().hex[:8]
    prefix = f"backup_{label}_" if label else "backup_"
    archive_path = dest_dir / f"{prefix}{stamp}_{ident}.tar.gz"

    conn = get_connection()
    tmp_db: Path | None = None
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        fd, tmp_name = tempfile.mkstemp(prefix="learningos-backup-", suffix=".db", dir=dest_dir)
        os.close(fd)
        tmp_db = Path(tmp_name)
        snapshot = sqlite3.connect(str(tmp_db))
        try:
            conn.backup(snapshot)
        finally:
            snapshot.close()
    finally:
        conn.close()

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            if tmp_db is not None and tmp_db.exists():
                tar.add(tmp_db, arcname="learningos.db")
            artifacts = home / "artifacts"
            if artifacts.is_dir():
                tar.add(artifacts, arcname="artifacts")
            config = home / "config.json"
            if config.is_file():
                tar.add(config, arcname="config.json")
    finally:
        if tmp_db is not None:
            tmp_db.unlink(missing_ok=True)
            Path(str(tmp_db) + "-wal").unlink(missing_ok=True)
            Path(str(tmp_db) + "-shm").unlink(missing_ok=True)

    return archive_path


def restore_backup(archive: Path, dest_home: Path) -> None:
    """Unpack a backup into a clean dest_home (must not exist or must be empty)."""
    archive = Path(archive)
    dest_home = Path(dest_home).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"backup archive not found: {archive}")
    if dest_home.exists():
        if not dest_home.is_dir():
            raise NotADirectoryError(str(dest_home))
        if any(dest_home.iterdir()):
            raise FileExistsError(f"restore destination is not empty: {dest_home}")
    else:
        dest_home.mkdir(parents=True)

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        if "learningos.db" not in names:
            raise ValueError("backup archive is missing learningos.db")
        _safe_extract(tar, dest_home)

    artifacts = dest_home / "artifacts"
    if artifacts.exists():
        _verify_artifact_checksums(artifacts)
