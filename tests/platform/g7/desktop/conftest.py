from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BACKEND_ROOT = _REPO_ROOT / "platform" / "backend"
_PLATFORM_TOOLS = _REPO_ROOT / "tools" / "platform"
_DESKTOP_TOOLS = _REPO_ROOT / "tools" / "desktop"

for path in (_BACKEND_ROOT, _PLATFORM_TOOLS, _DESKTOP_TOOLS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


class RecordingRunner:
    """In-process stand-in for venv/pip/npm so tests stay offline."""

    def __init__(
        self,
        *,
        python_version: str = "Python 3.12.2",
        node_version: str = "v20.11.1",
        npm_version: str = "10.8.2",
        node_missing: bool = False,
        npm_missing: bool = False,
        python_missing: bool = False,
        fail_pip: bool = False,
    ) -> None:
        self.python_version = python_version
        self.node_version = node_version
        self.npm_version = npm_version
        self.node_missing = node_missing
        self.npm_missing = npm_missing
        self.python_missing = python_missing
        self.fail_pip = fail_pip
        self.calls: list[list[str]] = []
        self.pip_installs = 0
        self.npm_ci = 0
        self.venv_creates = 0

    def run(self, command: Sequence[str], **kwargs) -> subprocess.CompletedProcess[str]:
        argv = [str(part) for part in command]
        self.calls.append(argv)
        name = Path(argv[0]).name.lower()

        if self.python_missing and "python" in name and "--version" in argv:
            raise FileNotFoundError("python3")
        if self.node_missing and name == "node":
            raise FileNotFoundError("node")
        if self.npm_missing and name.startswith("npm"):
            raise FileNotFoundError("npm")

        if argv[-1] == "--version" and "-m" not in argv:
            if "python" in name:
                return subprocess.CompletedProcess(argv, 0, self.python_version + "\n", "")
            if name == "node":
                return subprocess.CompletedProcess(argv, 0, self.node_version + "\n", "")
            if name.startswith("npm"):
                return subprocess.CompletedProcess(argv, 0, self.npm_version + "\n", "")

        if "-m" in argv and "venv" in argv:
            dest = Path(argv[argv.index("venv") + 1])
            self._write_venv(dest)
            self.venv_creates += 1
            return subprocess.CompletedProcess(argv, 0, "", "")

        if "-m" in argv and "pip" in argv:
            if "--version" in argv:
                return subprocess.CompletedProcess(argv, 0, "pip 24.0\n", "")
            if "install" in argv:
                if self.fail_pip:
                    return subprocess.CompletedProcess(argv, 1, "", "pip failed")
                self.pip_installs += 1
                return subprocess.CompletedProcess(argv, 0, "", "")

        if "-m" in argv and "ensurepip" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")

        if "-c" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")

        if name.startswith("npm") and "ci" in argv:
            prefix = self._npm_prefix(argv, kwargs)
            self._write_vite(prefix)
            self.npm_ci += 1
            return subprocess.CompletedProcess(argv, 0, "", "")

        return subprocess.CompletedProcess(argv, 0, "", "")

    def _npm_prefix(self, argv: list[str], kwargs: dict) -> Path:
        if "--prefix" in argv:
            return Path(argv[argv.index("--prefix") + 1])
        cwd = kwargs.get("cwd")
        if cwd:
            return Path(cwd)
        raise AssertionError(f"npm ci without prefix: {argv}")

    def _write_venv(self, dest: Path) -> None:
        bindir = dest / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        python = bindir / "python"
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        (dest / "pyvenv.cfg").write_text("home = /tmp\ninclude-system-site-packages = false\n", encoding="utf-8")

    def _write_vite(self, prefix: Path) -> None:
        vite = prefix / "node_modules" / ".bin" / "vite"
        vite.parent.mkdir(parents=True, exist_ok=True)
        vite.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        vite.chmod(0o755)
        (prefix / "node_modules" / ".package-lock.json").write_text("{}\n", encoding="utf-8")


def write_fake_repo(root: Path) -> None:
    backend = root / "platform" / "backend"
    frontend = root / "platform" / "frontend"
    worker = root / "platform" / "worker"
    tools_platform = root / "tools" / "platform"
    (backend / "app").mkdir(parents=True)
    frontend.mkdir(parents=True)
    worker.mkdir(parents=True)
    tools_platform.mkdir(parents=True)
    (root / "start.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (root / "start.sh").chmod(0o755)
    (backend / "requirements.txt").write_text("fastapi>=0.100.0\nuvicorn>=0.23.2\n", encoding="utf-8")
    (frontend / "package.json").write_text('{"name":"frontend","private":true}\n', encoding="utf-8")
    (frontend / "package-lock.json").write_text('{"name":"frontend","lockfileVersion":3}\n', encoding="utf-8")
    (backend / "app" / "main.py").write_text("# fake\n", encoding="utf-8")
    (worker / "daemon.py").write_text("# fake\n", encoding="utf-8")
    (tools_platform / "dev.py").write_text("# fake\n", encoding="utf-8")


@pytest.fixture
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture
def fake_repo() -> Path:
    root = Path(tempfile.mkdtemp(prefix="learningos-g7-desktop-repo-", dir="/tmp"))
    write_fake_repo(root)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def data_home() -> Path:
    root = Path(tempfile.mkdtemp(prefix="learningos-g7-desktop-home-", dir="/tmp"))
    home = root / "home"
    home.mkdir()
    yield home
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def Runner() -> type[RecordingRunner]:
    return RecordingRunner
