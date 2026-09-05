from __future__ import annotations

import os
from pathlib import Path

import pytest

import install
import launch


def test_prepare_launch_execs_start_sh_with_managed_python(fake_repo: Path, data_home: Path, Runner) -> None:
    plan = install.prepare_launch(
        repo_root=fake_repo,
        data_home=data_home,
        start_args=("--smoke",),
        runner=Runner(),
    )
    assert Path(plan.command[1]).resolve() == (fake_repo / "start.sh").resolve()
    assert plan.command[-1] == "--smoke"
    assert plan.cwd == fake_repo.resolve()
    assert Path(plan.env["LEARNINGOS_HOME"]) == data_home.resolve()
    assert plan.env["LEARNINGOS_PYTHON"] == str(plan.result.python)
    assert plan.env["LEARNINGOS_FRONTEND_HOME"] == str(plan.result.frontend_home)
    assert str(plan.result.paths.bin_dir) in plan.env["PATH"].split(os.pathsep)
    assert Path(plan.env["LEARNINGOS_PYTHON"]).is_relative_to(data_home.resolve())
    assert not Path(plan.env["LEARNINGOS_PYTHON"]).is_relative_to(fake_repo.resolve())


def test_second_launch_skips_package_install(fake_repo: Path, data_home: Path, Runner) -> None:
    install.prepare_launch(repo_root=fake_repo, data_home=data_home, runner=Runner())
    runner = Runner()
    plan = install.prepare_launch(repo_root=fake_repo, data_home=data_home, runner=runner)
    assert plan.result.skipped_backend
    assert plan.result.skipped_frontend
    assert runner.pip_installs == 0
    assert runner.npm_ci == 0
    assert runner.venv_creates == 0


def test_launch_cli_forwards_to_start_sh(
    fake_repo: Path, data_home: Path, monkeypatch: pytest.MonkeyPatch, Runner
) -> None:
    recorded: dict[str, object] = {}

    def fake_exec(program: str, command, env) -> None:
        recorded["program"] = program
        recorded["command"] = list(command)
        recorded["env"] = dict(env)

    monkeypatch.setenv("LEARNINGOS_REPO_ROOT", str(fake_repo))
    monkeypatch.setenv("LEARNINGOS_HOME", str(data_home))
    code = launch.main(["--", "--check"], runner=Runner(), executor=fake_exec)
    assert code == 0
    assert recorded["command"][1].endswith("start.sh")
    assert recorded["command"][-1] == "--check"
    env = recorded["env"]
    assert isinstance(env, dict)
    assert env["LEARNINGOS_PYTHON"].startswith(str(data_home.resolve()))
    assert env["LEARNINGOS_HOME"] == str(data_home.resolve())
