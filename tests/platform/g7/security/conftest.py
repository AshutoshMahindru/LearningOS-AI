from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
WORKER_ROOT = REPO_ROOT / "platform" / "worker"
for _path in (REPO_ROOT, BACKEND_ROOT, WORKER_ROOT):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    home = Path(tempfile.mkdtemp(prefix="learningos-g7-sec-", dir="/tmp"))
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.setenv("LEARNINGOS_USE_KEYCHAIN", "0")
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


@pytest.fixture
def worker_env(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    sock = Path(f"/tmp/los-g7-{uuid.uuid4().hex}.sock")
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(isolated_home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    env["LEARNINGOS_USE_KEYCHAIN"] = "0"
    try:
        yield {"home": isolated_home, "sock": sock, "env": env}
    finally:
        sock.unlink(missing_ok=True)
