#!/usr/bin/env python3
"""Bootstrap a managed LearningOS runtime outside the Git worktree.

Learners run ``python3 tools/desktop/launch.py``. This module creates
``$LEARNINGOS_HOME/runtime/`` (Python env + Node UI install) and can exec
``./start.sh`` with ``LEARNINGOS_PYTHON`` set. Host Python 3.11+ is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_PLATFORM_DIR = Path(__file__).resolve().parent
if str(_PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_DIR))

from dev import Check, is_within, print_checks, resolve_data_home, validate_data_home  # noqa: E402

STAMP_VERSION = 1
MIN_PYTHON = (3, 11)
MIN_NODE_MAJOR = 20
BACKEND_IMPORT_PROBE = "import fastapi, uvicorn, pydantic, jsonschema"
VITE_RELATIVE = Path("node_modules") / ".bin" / "vite"


class InstallError(Exception):
    """Learner-visible bootstrap failure."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class PreflightError(InstallError):
    """Host or path check failed before any install work."""


class SubprocessRunner:
    """Thin subprocess wrapper so tests can record commands without a network."""

    def run(self, command: Sequence[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(command), **kwargs)


@dataclass(frozen=True)
class RuntimePaths:
    data_home: Path
    runtime: Path
    venv: Path
    python: Path
    bin_dir: Path
    frontend: Path
    stamp: Path
    pip_cache: Path
    npm_cache: Path

    @classmethod
    def from_home(cls, data_home: Path) -> RuntimePaths:
        home = data_home.expanduser().resolve(strict=False)
        runtime = home / "runtime"
        venv = runtime / "python"
        if os.name == "nt":
            bin_dir = venv / "Scripts"
            python = bin_dir / "python.exe"
        else:
            bin_dir = venv / "bin"
            python = bin_dir / "python"
        return cls(
            data_home=home,
            runtime=runtime,
            venv=venv,
            python=python,
            bin_dir=bin_dir,
            frontend=runtime / "frontend",
            stamp=runtime / "bootstrap.json",
            pip_cache=runtime / "pip-cache",
            npm_cache=runtime / "npm-cache",
        )


@dataclass(frozen=True)
class BootstrapResult:
    paths: RuntimePaths
    python: Path
    frontend_home: Path
    skipped_backend: bool
    skipped_frontend: bool


@dataclass(frozen=True)
class LaunchPlan:
    command: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    result: BootstrapResult


def default_repo_root() -> Path:
    raw = os.environ.get("LEARNINGOS_REPO_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_python_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"Python\s+(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_node_major(text: str) -> int | None:
    match = re.search(r"v?(\d+)\.", text.strip())
    if not match:
        return None
    return int(match.group(1))


def _command_output(runner: SubprocessRunner, command: Sequence[str]) -> str:
    try:
        completed = runner.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FileNotFoundError(str(exc)) from exc
    if completed.returncode != 0:
        raise FileNotFoundError(
            (completed.stdout or "") + (completed.stderr or "") or f"{command[0]} exited {completed.returncode}"
        )
    return ((completed.stdout or "") + (completed.stderr or "")).strip()


def unique_python_candidates(explicit: Sequence[str] | None = None) -> list[str]:
    if explicit is not None:
        source: Sequence[str] = explicit
    else:
        source = (sys.executable, "python3.13", "python3.12", "python3.11", "python3")
    names: list[str] = []
    for candidate in source:
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def find_host_python(
    runner: SubprocessRunner,
    *,
    candidates: Sequence[str] | None = None,
) -> tuple[str, str]:
    errors: list[str] = []
    for candidate in unique_python_candidates(candidates):
        try:
            detail = _command_output(runner, (candidate, "--version"))
        except FileNotFoundError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        version = parse_python_version(detail)
        if version is not None and version >= MIN_PYTHON:
            return candidate, detail.splitlines()[0] if detail else candidate
        errors.append(f"{candidate}: {detail or 'unparsed version'}")
    tried = "; ".join(errors) if errors else "no Python interpreter found"
    raise PreflightError(
        "LearningOS requires Python 3.11 or newer on this machine. "
        "Install Python 3.11+ and re-run python3 tools/desktop/launch.py. "
        f"Tried: {tried}"
    )


def find_node(runner: SubprocessRunner) -> tuple[str, str]:
    try:
        detail = _command_output(runner, ("node", "--version"))
    except FileNotFoundError as exc:
        raise PreflightError(
            "LearningOS needs Node.js 20 or newer to run the UI. "
            f"Install Node.js 20+ and re-run python3 tools/desktop/launch.py. ({exc})"
        ) from exc
    major = parse_node_major(detail)
    if major is None or major < MIN_NODE_MAJOR:
        raise PreflightError(
            "LearningOS needs Node.js 20 or newer to run the UI. "
            f"Found {detail or 'an unknown Node.js version'}."
        )
    return "node", detail.splitlines()[0] if detail else "node"


def find_npm(runner: SubprocessRunner) -> tuple[str, str]:
    npm = shutil.which("npm") or "npm"
    try:
        detail = _command_output(runner, (npm, "--version"))
    except FileNotFoundError as exc:
        raise PreflightError(
            "LearningOS needs npm (bundled with Node.js 20+) to prepare the UI runtime. "
            f"Install Node.js 20+ and re-run python3 tools/desktop/launch.py. ({exc})"
        ) from exc
    return npm, detail.splitlines()[0] if detail else npm


def required_repo_files(repo_root: Path) -> list[Path]:
    return [
        repo_root / "start.sh",
        repo_root / "platform" / "backend" / "requirements.txt",
        repo_root / "platform" / "frontend" / "package.json",
        repo_root / "platform" / "frontend" / "package-lock.json",
        repo_root / "platform" / "backend" / "app" / "main.py",
        repo_root / "platform" / "worker" / "daemon.py",
        repo_root / "tools" / "platform" / "dev.py",
    ]


def host_preflight(
    *,
    repo_root: Path,
    data_home: Path,
    runner: SubprocessRunner,
    python_candidates: Sequence[str] | None = None,
) -> tuple[list[Check], str, str]:
    checks: list[Check] = []
    try:
        python_bin, python_detail = find_host_python(runner, candidates=python_candidates)
        checks.append(Check("Python 3.11+", True, python_detail))
    except PreflightError as exc:
        checks.append(Check("Python 3.11+", False, str(exc)))
        python_bin, python_detail = "", str(exc)

    try:
        _node, node_detail = find_node(runner)
        checks.append(Check("Node.js 20+", True, node_detail))
    except PreflightError as exc:
        checks.append(Check("Node.js 20+", False, str(exc)))

    try:
        _npm, npm_detail = find_npm(runner)
        checks.append(Check("npm", True, npm_detail))
    except PreflightError as exc:
        checks.append(Check("npm", False, str(exc)))

    missing = [str(path.relative_to(repo_root)) for path in required_repo_files(repo_root) if not path.is_file()]
    checks.append(
        Check(
            "repository files",
            not missing,
            "present" if not missing else f"missing {', '.join(missing)}",
        )
    )
    checks.append(validate_data_home(data_home, repo_root))
    return checks, python_bin, python_detail


def assert_host_ready(checks: Sequence[Check]) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        details = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise PreflightError(details)


def load_stamp(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_stamp(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def backend_ready(python: Path, runner: SubprocessRunner) -> bool:
    if not python.is_file():
        return False
    try:
        completed = runner.run(
            (str(python), "-c", BACKEND_IMPORT_PROBE),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def frontend_ready(frontend_home: Path) -> bool:
    return (frontend_home / VITE_RELATIVE).is_file()


def _run_checked(
    runner: SubprocessRunner,
    command: Sequence[str],
    *,
    action: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int = 600,
) -> None:
    try:
        completed = runner.run(
            command,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise InstallError(f"{action} timed out after {timeout}s.") from exc
    except OSError as exc:
        raise InstallError(f"{action} failed: {exc}") from exc
    if completed.returncode != 0:
        raise InstallError(f"{action} failed (exit {completed.returncode}).")


def create_venv(host_python: str, paths: RuntimePaths, runner: SubprocessRunner) -> None:
    paths.venv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating managed Python runtime at {paths.venv}")
    _run_checked(
        runner,
        (host_python, "-m", "venv", str(paths.venv)),
        action="Creating the managed Python runtime",
        timeout=120,
    )
    if not paths.python.is_file():
        raise InstallError(f"Managed Python interpreter missing after venv create: {paths.python}")


def install_backend(paths: RuntimePaths, requirements: Path, runner: SubprocessRunner) -> None:
    print(f"Installing API dependencies into {paths.venv}")
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(paths.pip_cache)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    probe = runner.run(
        (str(paths.python), "-m", "pip", "--version"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    if probe.returncode != 0:
        _run_checked(
            runner,
            (str(paths.python), "-m", "ensurepip", "--upgrade"),
            action="Bootstrapping pip in the managed Python runtime",
            env=env,
            timeout=120,
        )
    _run_checked(
        runner,
        (
            str(paths.python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "-r",
            str(requirements),
        ),
        action="Installing API dependencies",
        env=env,
        timeout=600,
    )
    if not backend_ready(paths.python, runner):
        raise InstallError("Managed Python runtime is missing API packages after install.")


def install_frontend(
    paths: RuntimePaths,
    frontend_src: Path,
    npm: str,
    runner: SubprocessRunner,
) -> None:
    print(f"Installing UI runtime at {paths.frontend}")
    paths.frontend.mkdir(parents=True, exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        source = frontend_src / name
        shutil.copy2(source, paths.frontend / name)
    env = os.environ.copy()
    env["npm_config_cache"] = str(paths.npm_cache)
    env["NODE_ENV"] = "development"
    env.pop("npm_config_production", None)
    _run_checked(
        runner,
        (npm, "ci", "--no-fund", "--no-audit", "--prefix", str(paths.frontend)),
        action="Installing the managed UI runtime",
        cwd=paths.frontend,
        env=env,
        timeout=600,
    )
    if not frontend_ready(paths.frontend):
        raise InstallError(f"UI runtime missing Vite after npm ci: {paths.frontend / VITE_RELATIVE}")


def ensure_frontend_link(frontend_src: Path, managed_frontend: Path) -> None:
    """Point the checkout at the managed node_modules so ./start.sh finds Vite."""
    managed_modules = (managed_frontend / "node_modules").resolve()
    if not managed_modules.is_dir():
        raise InstallError(f"Managed UI node_modules missing: {managed_modules}")
    link = frontend_src / "node_modules"
    if link.is_symlink():
        try:
            current = link.resolve()
        except OSError:
            current = None
        if current == managed_modules:
            return
        link.unlink()
    elif link.exists():
        # Developer checkout already has a local install; leave it in place.
        return
    link.symlink_to(managed_modules, target_is_directory=True)


def bootstrap(
    *,
    repo_root: Path | None = None,
    data_home: Path | None = None,
    runner: SubprocessRunner | None = None,
    python_candidates: Sequence[str] | None = None,
) -> BootstrapResult:
    repo = (repo_root or default_repo_root()).expanduser().resolve()
    home = (data_home or resolve_data_home(None)).expanduser().resolve(strict=False)
    active_runner = runner or SubprocessRunner()
    if is_within(home, repo):
        raise PreflightError(
            f"{home} is inside the Git worktree; choose an external LEARNINGOS_HOME"
        )

    checks, host_python, _python_detail = host_preflight(
        repo_root=repo,
        data_home=home,
        runner=active_runner,
        python_candidates=python_candidates,
    )
    print_checks(checks)
    assert_host_ready(checks)
    if not host_python:
        raise PreflightError("LearningOS requires Python 3.11 or newer on this machine.")

    npm, _npm_detail = find_npm(active_runner)
    paths = RuntimePaths.from_home(home)
    if is_within(paths.runtime, repo):
        raise PreflightError(
            f"Managed runtime {paths.runtime} would land inside the Git worktree; "
            "set LEARNINGOS_HOME to an external directory."
        )
    paths.runtime.mkdir(parents=True, exist_ok=True)

    requirements = repo / "platform" / "backend" / "requirements.txt"
    frontend_src = repo / "platform" / "frontend"
    req_hash = file_digest(requirements)
    lock_hash = file_digest(frontend_src / "package-lock.json")
    stamp = load_stamp(paths.stamp)
    stamp_ok = stamp.get("version") == STAMP_VERSION
    skip_backend = bool(
        stamp_ok
        and stamp.get("requirements_sha256") == req_hash
        and paths.python.is_file()
        and backend_ready(paths.python, active_runner)
    )
    skip_frontend = bool(
        stamp_ok
        and stamp.get("package_lock_sha256") == lock_hash
        and frontend_ready(paths.frontend)
    )

    if not skip_backend:
        if not paths.python.is_file():
            create_venv(host_python, paths, active_runner)
        install_backend(paths, requirements, active_runner)
    else:
        print(f"Managed Python runtime is current: {paths.python}")

    if not skip_frontend:
        install_frontend(paths, frontend_src, npm, active_runner)
    else:
        print(f"Managed UI runtime is current: {paths.frontend}")

    ensure_frontend_link(frontend_src, paths.frontend)
    write_stamp(
        paths.stamp,
        {
            "version": STAMP_VERSION,
            "requirements_sha256": req_hash,
            "package_lock_sha256": lock_hash,
            "python": str(paths.python),
            "frontend": str(paths.frontend),
        },
    )
    print(f"Managed runtime ready under {paths.runtime}")
    return BootstrapResult(
        paths=paths,
        python=paths.python,
        frontend_home=paths.frontend,
        skipped_backend=skip_backend,
        skipped_frontend=skip_frontend,
    )


def launch_environment(result: BootstrapResult, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    bin_dir = str(result.paths.bin_dir)
    existing_path = env.get("PATH", "")
    env["PATH"] = bin_dir + (os.pathsep + existing_path if existing_path else "")
    env["LEARNINGOS_HOME"] = str(result.paths.data_home)
    env["LEARNINGOS_PYTHON"] = str(result.python)
    env["LEARNINGOS_FRONTEND_HOME"] = str(result.frontend_home)
    env["VIRTUAL_ENV"] = str(result.paths.venv)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def prepare_launch(
    *,
    repo_root: Path | None = None,
    data_home: Path | None = None,
    start_args: Sequence[str] = (),
    runner: SubprocessRunner | None = None,
    python_candidates: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> LaunchPlan:
    repo = (repo_root or default_repo_root()).expanduser().resolve()
    result = bootstrap(
        repo_root=repo,
        data_home=data_home,
        runner=runner,
        python_candidates=python_candidates,
    )
    start_script = repo / "start.sh"
    if not start_script.is_file():
        raise InstallError(f"Missing launcher {start_script}")
    bash = shutil.which("bash") or "bash"
    command = (bash, str(start_script), *start_args)
    return LaunchPlan(
        command=command,
        env=launch_environment(result, environ),
        cwd=repo,
        result=result,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the managed LearningOS runtime under LEARNINGOS_HOME "
            "(default ~/.learningos) and optionally launch the app."
        ),
        epilog="One-click launch: python3 tools/desktop/launch.py",
    )
    parser.add_argument("--check", action="store_true", help="run host diagnostics without installing or starting")
    parser.add_argument("--launch", action="store_true", help="after bootstrap, exec ./start.sh")
    parser.add_argument("--data-home", help="external mutable data directory (or set LEARNINGOS_HOME)")
    parser.add_argument(
        "start_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to ./start.sh after -- (for example -- --smoke)",
    )
    return parser.parse_args(argv)


def _forwarded_start_args(raw: Sequence[str]) -> tuple[str, ...]:
    args = list(raw)
    if args and args[0] == "--":
        args = args[1:]
    return tuple(args)


def main(
    argv: list[str] | None = None,
    *,
    runner: SubprocessRunner | None = None,
    executor=None,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    repo_root = default_repo_root()
    data_home = resolve_data_home(args.data_home)
    active_runner = runner or SubprocessRunner()
    try:
        if args.check:
            checks, _python, _detail = host_preflight(
                repo_root=repo_root,
                data_home=data_home,
                runner=active_runner,
            )
            print_checks(checks)
            assert_host_ready(checks)
            print("Preflight PASSED.")
            return 0
        if args.launch:
            plan = prepare_launch(
                repo_root=repo_root,
                data_home=data_home,
                start_args=_forwarded_start_args(args.start_args),
                runner=active_runner,
            )
            print("Launching LearningOS (API, worker, UI)...")
            launch = executor or os.execvpe
            launch(plan.command[0], plan.command, plan.env)
            return 0
        bootstrap(repo_root=repo_root, data_home=data_home, runner=active_runner)
        return 0
    except PreflightError as exc:
        print(f"Preflight FAILED: {exc}", file=sys.stderr)
        return 1
    except InstallError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
