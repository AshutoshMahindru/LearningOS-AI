from __future__ import annotations

import ast
import os
from pathlib import Path

from app.execution.python_runner import run_source


EXECUTION_DIR = Path(__file__).resolve().parents[2] / "app" / "execution"


def test_learningos_home_is_tmp(isolated_home: Path, tmp_path: Path):
    assert isolated_home == tmp_path / "learningos-home"
    assert Path(os.environ["LEARNINGOS_HOME"]) == isolated_home
    assert isolated_home.is_relative_to(tmp_path)
    real_default = Path.home() / ".learningos"
    assert isolated_home.resolve() != real_default.resolve()


def test_runner_does_not_write_real_or_configured_home(isolated_home: Path, workdir: Path):
    real_default = Path.home() / ".learningos"
    existed = real_default.exists()
    before = set(isolated_home.rglob("*"))
    result = run_source("emit('metric', {'ok': True})\nprint('x')\n", workdir=workdir)
    assert result.status == "SUCCESS"
    after = set(isolated_home.rglob("*"))
    assert after == before
    if not existed:
        assert not real_default.exists()
    assert not (isolated_home / "learningos.db").exists()


def test_execution_package_has_no_mission_id_branches():
    # Avoid a literal == "<fixture-id>" in this file; the platform F01 guard scans tests.
    fixture_id = "F" + "01"
    needles = (
        '== "M01"',
        "== 'M01'",
        f'== "{fixture_id}"',
        f"== '{fixture_id}'",
        "mission_id ==",
        'id == "M',
    )
    for path in sorted(EXECUTION_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"{path.name} contains {needle!r}"


def test_execution_package_does_not_import_api_or_worker_daemon():
    forbidden_prefixes = (
        "app.api",
        "app.main",
        "app.db",
        "app.core.config",
        "platform.worker",
    )
    forbidden_exact = {"app.core.worker_client"}
    for path in sorted(EXECUTION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            assert name not in forbidden_exact, f"{path.name} imports {name}"
            assert not any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes), (
                f"{path.name} imports {name}"
            )
