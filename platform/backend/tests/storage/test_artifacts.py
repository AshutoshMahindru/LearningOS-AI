from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.core.artifact_store import ArtifactStore
from app.db.database import init_db


def test_put_get_roundtrip_and_layout(data_home: Path) -> None:
    init_db()
    store = ArtifactStore()
    payload = b"artifact-bytes-\x00\xff"
    digest = store.put(payload, media_type="application/octet-stream")
    assert digest == hashlib.sha256(payload).hexdigest()
    assert store.get(digest) == payload
    path = store.path_for(digest)
    assert path == data_home / "artifacts" / "sha256" / digest[:2] / digest[2:]
    assert path.is_file()
    assert store.put(payload) == digest


def test_checksum_mismatch_is_detected(data_home: Path) -> None:
    init_db()
    store = ArtifactStore()
    digest = store.put(b"original")
    store.path_for(digest).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.get(digest)


def test_get_missing_artifact(data_home: Path) -> None:
    init_db()
    store = ArtifactStore()
    missing = hashlib.sha256(b"absent").hexdigest()
    with pytest.raises(FileNotFoundError):
        store.get(missing)


def test_explicit_root(tmp_path: Path) -> None:
    root = tmp_path / "custom-artifacts"
    store = ArtifactStore(root)
    digest = store.put(b"rooted")
    assert (root / "sha256" / digest[:2] / digest[2:]).is_file()
    assert store.get(digest) == b"rooted"
