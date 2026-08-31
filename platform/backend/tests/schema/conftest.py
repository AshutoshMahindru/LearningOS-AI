from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "learningos-home"
    home.mkdir()
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.delenv("LEARNINGOS_WORKER_SOCKET", raising=False)
    return home
