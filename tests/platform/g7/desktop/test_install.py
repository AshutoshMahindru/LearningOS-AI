from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import install

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_installer_creates_managed_runtime_outside_worktree(fake_repo: Path, data_home: Path, Runner) -> None:
    runner = Runner()
    result = install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=runner)

    assert result.python.is_file()
    assert result.python.is_relative_to(data_home.resolve())
    assert not result.python.is_relative_to(fake_repo.resolve())
    assert (data_home / "runtime" / "python" / "pyvenv.cfg").is_file()
    assert (data_home / "runtime" / "frontend" / "node_modules" / ".bin" / "vite").is_file()
    assert (data_home / "runtime" / "bootstrap.json").is_file()
    assert not (fake_repo / ".venv").exists()
    assert not (fake_repo / "venv").exists()
    assert not (fake_repo / "learningos.db").exists()
    assert not (fake_repo / ".learningos").exists()

    link = fake_repo / "platform" / "frontend" / "node_modules"
    assert link.is_symlink()
    assert link.resolve() == (data_home / "runtime" / "frontend" / "node_modules").resolve()
    assert not link.resolve().is_relative_to(fake_repo.resolve())
    assert runner.venv_creates == 1
    assert runner.pip_installs == 1
    assert runner.npm_ci == 1
    assert any(call[:3] == [str(result.python), "-m", "pip"] and "install" in call for call in runner.calls)
    assert any("ci" in call and "--prefix" in call for call in runner.calls)


def test_second_bootstrap_is_idempotent(fake_repo: Path, data_home: Path, Runner) -> None:
    first = install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=Runner())
    assert not first.skipped_backend
    assert not first.skipped_frontend

    second_runner = Runner()
    second = install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=second_runner)
    assert second.skipped_backend
    assert second.skipped_frontend
    assert second_runner.venv_creates == 0
    assert second_runner.pip_installs == 0
    assert second_runner.npm_ci == 0
    assert second.python == first.python


def test_requirements_change_reinstalls_backend(fake_repo: Path, data_home: Path, Runner) -> None:
    install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=Runner())
    requirements = fake_repo / "platform" / "backend" / "requirements.txt"
    requirements.write_text(requirements.read_text(encoding="utf-8") + "jsonschema>=4.19.0\n", encoding="utf-8")
    runner = Runner()
    result = install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=runner)
    assert not result.skipped_backend
    assert result.skipped_frontend
    assert runner.pip_installs == 1
    assert runner.npm_ci == 0
    assert runner.venv_creates == 0


def test_rejects_learningos_home_inside_worktree(fake_repo: Path, Runner) -> None:
    with pytest.raises(install.PreflightError, match="inside the Git worktree"):
        install.bootstrap(repo_root=fake_repo, data_home=fake_repo / ".learningos", runner=Runner())


def test_preflight_fails_without_python_311(fake_repo: Path, data_home: Path, Runner) -> None:
    runner = Runner(python_version="Python 3.10.14")
    with pytest.raises(install.PreflightError, match="Python 3.11"):
        install.bootstrap(
            repo_root=fake_repo,
            data_home=data_home,
            runner=runner,
            python_candidates=["python3"],
        )
    assert runner.venv_creates == 0
    assert runner.pip_installs == 0


def test_preflight_fails_without_node(fake_repo: Path, data_home: Path, Runner) -> None:
    runner = Runner(node_missing=True)
    with pytest.raises(install.PreflightError, match="Node.js 20"):
        install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=runner)


def test_check_does_not_install(
    fake_repo: Path, data_home: Path, monkeypatch: pytest.MonkeyPatch, Runner
) -> None:
    runner = Runner()
    monkeypatch.setenv("LEARNINGOS_REPO_ROOT", str(fake_repo))
    monkeypatch.setenv("LEARNINGOS_HOME", str(data_home))
    code = install.main(["--check"], runner=runner, executor=lambda *_args, **_kwargs: None)
    assert code == 0
    assert runner.venv_creates == 0
    assert runner.pip_installs == 0
    assert runner.npm_ci == 0
    assert not (data_home / "runtime").exists()


def test_stamp_records_hashes_outside_worktree(fake_repo: Path, data_home: Path, Runner) -> None:
    install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=Runner())
    stamp = json.loads((data_home / "runtime" / "bootstrap.json").read_text(encoding="utf-8"))
    requirements = fake_repo / "platform" / "backend" / "requirements.txt"
    lockfile = fake_repo / "platform" / "frontend" / "package-lock.json"
    assert stamp["version"] == install.STAMP_VERSION
    assert stamp["requirements_sha256"] == install.file_digest(requirements)
    assert stamp["package_lock_sha256"] == install.file_digest(lockfile)


def test_existing_worktree_node_modules_are_left_in_place(fake_repo: Path, data_home: Path, Runner) -> None:
    existing = fake_repo / "platform" / "frontend" / "node_modules"
    existing.mkdir()
    (existing / "keep").write_text("local\n", encoding="utf-8")
    install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=Runner())
    assert existing.is_dir()
    assert not existing.is_symlink()
    assert (existing / "keep").read_text(encoding="utf-8") == "local\n"
    assert (data_home / "runtime" / "frontend" / "node_modules" / ".bin" / "vite").is_file()


def test_managed_runtime_is_not_inside_this_checkout(data_home: Path) -> None:
    paths = install.RuntimePaths.from_home(data_home)
    assert not paths.runtime.is_relative_to(REPO_ROOT)
    assert not paths.venv.is_relative_to(REPO_ROOT)
    assert os.environ.get("LEARNINGOS_HOME") != str(REPO_ROOT)
