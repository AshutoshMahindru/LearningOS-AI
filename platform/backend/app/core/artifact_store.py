"""Content-addressed SHA-256 artifact store under LEARNINGOS_HOME/artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from app.db.database import get_data_home


def _validate_sha256_hex(value: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"invalid sha256 hex digest: {value!r}")
    return digest


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=dest.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dest)
        dir_fd = os.open(dest.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class ArtifactStore:
    def __init__(self, root: Path | str | None = None):
        if root is None:
            self.root = get_data_home() / "artifacts"
        else:
            self.root = Path(root)

    def path_for(self, hash: str) -> Path:
        digest = _validate_sha256_hex(hash)
        return self.root / "sha256" / digest[:2] / digest[2:]

    def put(self, data: bytes, *, media_type: str | None = None) -> str:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        payload = bytes(data)
        digest = hashlib.sha256(payload).hexdigest()
        dest = self.path_for(digest)
        if dest.exists():
            existing = dest.read_bytes()
            if hashlib.sha256(existing).hexdigest() != digest:
                _atomic_write(dest, payload)
            return digest
        _atomic_write(dest, payload)
        _ = media_type
        return digest

    def get(self, hash: str) -> bytes:
        digest = _validate_sha256_hex(hash)
        path = self.path_for(digest)
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found: {digest}")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ValueError(f"checksum mismatch for artifact {digest}")
        return data
