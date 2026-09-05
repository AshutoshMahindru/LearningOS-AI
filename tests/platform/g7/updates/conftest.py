from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"

for candidate in (REPO_ROOT, BACKEND_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)


@pytest.fixture
def data_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    home = Path(tempfile.mkdtemp(prefix="learningos-g7-updates-", dir="/tmp"))
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    yield home
    import shutil

    shutil.rmtree(home, ignore_errors=True)


@pytest.fixture
def dest_home() -> Path:
    dest = Path(tempfile.mkdtemp(prefix="learningos-g7-restore-", dir="/tmp"))
    yield dest
    import shutil

    shutil.rmtree(dest, ignore_errors=True)
