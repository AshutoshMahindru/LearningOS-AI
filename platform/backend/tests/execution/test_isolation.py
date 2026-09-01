from __future__ import annotations

import ast
import os
from pathlib import Path

from app.execution.python_runner import run_source
from app.execution.result_schema import validate_structured_result


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


def test_io_open_cannot_write_learningos_home_db(isolated_home: Path, workdir: Path):
    db_path = isolated_home / "learningos.db"
    wal_path = isolated_home / "learningos.db-wal"
    shm_path = isolated_home / "learningos.db-shm"
    db_path.write_bytes(b"safe-bytes")
    wal_path.write_bytes(b"wal-bytes")
    shm_path.write_bytes(b"shm-bytes")
    source = (
        "import io\n"
        f"io.open({str(db_path)!r}, 'w').write('corrupt')\n"
    )
    result = run_source(source, workdir=workdir, data_home=isolated_home)
    validate_structured_result(result)
    assert result.status == "FAILED"
    assert db_path.read_bytes() == b"safe-bytes"
    assert wal_path.read_bytes() == b"wal-bytes"
    assert shm_path.read_bytes() == b"shm-bytes"

    wal_source = (
        "import io\n"
        f"io.open({str(wal_path)!r}, 'w').write('corrupt-wal')\n"
    )
    wal_result = run_source(wal_source, workdir=workdir, data_home=isolated_home)
    assert wal_result.status == "FAILED"
    assert wal_path.read_bytes() == b"wal-bytes"

    shm_source = (
        "import io\n"
        f"io.open({str(shm_path)!r}, 'w').write('corrupt-shm')\n"
    )
    shm_result = run_source(shm_source, workdir=workdir, data_home=isolated_home)
    assert shm_result.status == "FAILED"
    assert shm_path.read_bytes() == b"shm-bytes"


def test_io_open_cannot_write_fake_repo_worktree(tmp_path: Path, workdir: Path, isolated_home: Path):
    fake_repo = tmp_path / "fake-repo"
    (fake_repo / ".git").mkdir(parents=True)
    (fake_repo / "platform" / "backend" / "app").mkdir(parents=True)
    (fake_repo / "architecture" / "learningos-v3").mkdir(parents=True)
    marker = fake_repo / "pwned.txt"
    source = (
        "import io\n"
        f"io.open({str(marker)!r}, 'w').write('pwned')\n"
    )
    result = run_source(
        source,
        workdir=workdir,
        repo_root=fake_repo,
        data_home=isolated_home,
    )
    validate_structured_result(result)
    assert result.status == "FAILED"
    assert not marker.exists()


def test_io_open_inside_workdir_still_allowed(workdir: Path):
    result = run_source(
        "import io\nio.open('ok.txt', 'w', encoding='utf-8').write('ok')\n",
        workdir=workdir,
    )
    validate_structured_result(result)
    assert result.status == "SUCCESS"
    assert (workdir / "ok.txt").read_text(encoding="utf-8") == "ok"


def test_banned_imports_are_blocked(workdir: Path):
    for snippet in (
        "import os",
        "import pathlib",
        "import sqlite3",
        "import subprocess",
        "import socket",
    ):
        result = run_source(snippet, workdir=workdir)
        assert result.status == "FAILED", snippet


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
