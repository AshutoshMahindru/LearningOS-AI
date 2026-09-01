from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PLATFORM_ROOT / "backend"
WORKER_ROOT = Path(__file__).resolve().parents[1]
DAEMON_PATH = WORKER_ROOT / "daemon.py"
WATCHDOG_PATH = WORKER_ROOT / "watchdog.py"
for _path in (BACKEND_ROOT, WORKER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def start_daemon(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, str(DAEMON_PATH)], env=env)


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    # macOS AF_UNIX sun_path is ~104 bytes; pytest tmp paths are too long.
    sock = Path(f"/tmp/los-g3-{uuid.uuid4().hex}.sock")
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        yield {"home": home, "sock": sock, "env": env}
    finally:
        sock.unlink(missing_ok=True)
