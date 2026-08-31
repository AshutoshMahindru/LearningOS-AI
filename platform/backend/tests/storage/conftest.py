from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "platform" / "backend" / "app").is_dir() and (
            (candidate / ".git").exists() or (candidate / "architecture" / "learningos-v3").is_dir()
        ):
            return candidate
    raise RuntimeError("could not locate Git worktree root")


@pytest.fixture
def worktree_root() -> Path:
    return _repo_root()


@pytest.fixture
def data_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "learningos-home"
    home.mkdir()
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _isolate_learningos_home(data_home: Path) -> Path:
    return data_home
