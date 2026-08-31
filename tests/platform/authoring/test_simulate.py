from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.authoring.errors import AuthoringError
from tools.authoring.simulate import simulate_package

REPO_ROOT = Path(__file__).resolve().parents[3]
F01_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "f01"
CLI_PATH = REPO_ROOT / "tools" / "authoring" / "cli.py"


def _repo_learningos_markers() -> list[Path]:
    return [
        REPO_ROOT / ".learningos",
        REPO_ROOT / "platform" / ".learningos",
        REPO_ROOT / "platform" / "fixtures" / "f01" / ".learningos",
        REPO_ROOT / "tools" / "authoring" / ".learningos",
        REPO_ROOT / "tests" / "platform" / "authoring" / ".learningos",
        REPO_ROOT / "learningos.db",
    ]


def test_simulate_writes_only_under_tmp_home(tmp_path: Path) -> None:
    home = tmp_path / "learningos-home"
    markers = _repo_learningos_markers()
    before = {path: path.exists() for path in markers}
    real_home = Path.home() / ".learningos"
    real_before = real_home.stat().st_mtime_ns if real_home.exists() else None

    result = simulate_package(F01_PACKAGE, home=home, repo_root=REPO_ROOT)

    assert result.ok
    assert result.home == home.resolve()
    assert result.trace_path.is_file()
    assert result.trace_path.is_relative_to(home.resolve())
    payload = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert payload["mission_id"] == "M00"
    assert payload["worktree_writes"] is False
    assert payload["mission"] == "COMPLETED"
    types = [stage["type"] for stage in payload["stages"]]
    assert types[0] == "orientation"
    assert "experiment" in types
    experiment = next(stage for stage in payload["stages"] if stage["type"] == "experiment")
    assert experiment["actions"] == ["predict", "execute", "submit"]
    assert experiment["worker_invoked"] is False
    assert payload["gate"]["status"] == "PASSED"
    assert "predict" in result.text
    with pytest.raises(ValueError):
        home.resolve().relative_to(REPO_ROOT.resolve())

    for path in markers:
        if not before[path]:
            assert not path.exists(), f"simulate created {path}"
    if real_before is None:
        # Must not create the developer home as a side effect.
        if real_home.exists():
            pytest.fail("simulate created ~/.learningos")
    else:
        assert real_home.stat().st_mtime_ns == real_before


def test_simulate_refuses_repo_local_home() -> None:
    target = REPO_ROOT / ".learningos"
    existed = target.exists()
    with pytest.raises(AuthoringError) as raised:
        simulate_package(F01_PACKAGE, home=target, repo_root=REPO_ROOT)
    assert raised.value.code == "DATA_HOME"
    if not existed:
        assert not target.exists()


def test_cli_simulate_does_not_create_repo_local_learningos(tmp_path: Path) -> None:
    home = tmp_path / "cli-home"
    markers = _repo_learningos_markers()
    before = {path: path.exists() for path in markers}
    env = os.environ.copy()
    env.pop("LEARNINGOS_HOME", None)
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "simulate", "--home", str(home)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "COMPLETED" in completed.stdout
    assert "Git worktree untouched" in completed.stdout
    assert (home / "authoring" / "simulate-trace.json").is_file()
    assert not (REPO_ROOT / ".learningos").exists() or before[REPO_ROOT / ".learningos"]
    for path in markers:
        if not before[path]:
            assert not path.exists()


def test_cli_simulate_default_tmp_home_is_outside_repo() -> None:
    markers = _repo_learningos_markers()
    before = {path: path.exists() for path in markers}
    env = os.environ.copy()
    env.pop("LEARNINGOS_HOME", None)
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "simulate"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "LEARNINGOS_HOME" in completed.stdout
    for path in markers:
        if not before[path]:
            assert not path.exists(), f"simulate created {path}"
    # Default home is a temp path printed on the first line.
    first = completed.stdout.splitlines()[0]
    assert first.startswith("LEARNINGOS_HOME")
    printed_home = Path(first.split(maxsplit=1)[1])
    with pytest.raises(ValueError):
        printed_home.resolve().relative_to(REPO_ROOT.resolve())
