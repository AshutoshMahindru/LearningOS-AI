#!/usr/bin/env python3
"""Preflight, start, supervise, smoke-test, and stop LearningOS V3 services."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Sequence

try:
    from .state_guard import find_violations
except ImportError:  # Direct script execution from start.sh.
    from state_guard import find_violations


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
FRONTEND_ROOT = REPO_ROOT / "platform" / "frontend"
DEFAULT_WORKER_SOCKET = Path("/tmp/learningos_worker.sock")
REQUIRED_BACKEND_MODULES = ("fastapi", "uvicorn", "pydantic", "jsonschema")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: tuple[str, ...]
    cwd: Path


@dataclass
class ManagedProcess:
    spec: ServiceSpec
    process: subprocess.Popen[bytes]


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_data_home(raw_path: str | None) -> Path:
    configured = raw_path or os.environ.get("LEARNINGOS_HOME") or "~/.learningos"
    return Path(configured).expanduser().resolve(strict=False)


def validate_data_home(data_home: Path, repo_root: Path = REPO_ROOT) -> Check:
    if is_within(data_home, repo_root.resolve()):
        return Check(
            "external data home",
            False,
            f"{data_home} is inside the Git worktree; choose an external LEARNINGOS_HOME",
        )

    existing_parent = data_home
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    writable = existing_parent.is_dir() and os.access(existing_parent, os.W_OK | os.X_OK)
    return Check(
        "external data home",
        writable,
        f"{data_home} (writable parent: {existing_parent})" if writable else f"{existing_parent} is not writable",
    )


def _version(command: Sequence[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "no version output"
    return completed.returncode == 0, detail


def _major_version(detail: str) -> int | None:
    for token in detail.replace("Python", "").split():
        candidate = token.removeprefix("v").split(".", maxsplit=1)[0]
        if candidate.isdigit():
            return int(candidate)
    return None


def _port_available(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in probe_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((probe_host, port))
        except OSError:
            return False
    return True


def run_preflight(
    *,
    data_home: Path,
    host: str,
    backend_port: int,
    frontend_port: int,
    python_bin: str,
) -> list[Check]:
    checks: list[Check] = []
    python_ok, python_detail = _version((python_bin, "--version"))
    checks.append(Check("Python 3.11+", python_ok and sys.version_info >= (3, 11), python_detail))

    node_ok, node_detail = _version(("node", "--version"))
    npm_ok, npm_detail = _version(("npm", "--version"))
    node_major = _major_version(node_detail)
    checks.append(
        Check(
            "Node.js 20+",
            node_ok and node_major is not None and node_major >= 20,
            node_detail,
        )
    )
    checks.append(Check("npm", npm_ok, npm_detail))

    missing_modules = [name for name in REQUIRED_BACKEND_MODULES if importlib.util.find_spec(name) is None]
    checks.append(
        Check(
            "backend dependencies",
            not missing_modules,
            "installed" if not missing_modules else f"missing {', '.join(missing_modules)}; run python -m pip install -r platform/backend/requirements.txt",
        )
    )
    frontend_ready = (FRONTEND_ROOT / "node_modules" / ".bin" / "vite").exists()
    checks.append(
        Check(
            "frontend dependencies",
            frontend_ready,
            "installed" if frontend_ready else "missing; run npm ci --prefix platform/frontend",
        )
    )

    required_paths = (
        BACKEND_ROOT / "app" / "main.py",
        FRONTEND_ROOT / "package-lock.json",
        REPO_ROOT / "platform" / "worker" / "daemon.py",
    )
    missing_paths = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.is_file()]
    checks.append(Check("service entrypoints", not missing_paths, "present" if not missing_paths else f"missing {', '.join(missing_paths)}"))

    checks.append(validate_data_home(data_home))
    violations = find_violations(REPO_ROOT)
    checks.append(
        Check(
            "learner-state worktree guard",
            not violations,
            "clean" if not violations else ", ".join(path.as_posix() for path in violations),
        )
    )
    checks.append(Check(f"backend port {backend_port}", _port_available(host, backend_port), "available"))
    checks.append(Check(f"frontend port {frontend_port}", _port_available(host, frontend_port), "available"))
    return checks


def print_checks(checks: Sequence[Check]) -> None:
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")


def command_override(variable: str, default: Sequence[str]) -> tuple[str, ...]:
    raw = os.environ.get(variable)
    if not raw:
        return tuple(default)
    parsed = tuple(shlex.split(raw))
    if not parsed:
        raise ValueError(f"{variable} must not be empty")
    return parsed


def service_specs(args: argparse.Namespace, python_bin: str) -> list[ServiceSpec]:
    backend_command = [
        python_bin,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
    ]
    if args.reload:
        backend_command.append("--reload")

    return [
        ServiceSpec(
            "worker",
            command_override(
                "LEARNINGOS_WORKER_COMMAND",
                (python_bin, str(REPO_ROOT / "platform" / "worker" / "daemon.py")),
            ),
            REPO_ROOT,
        ),
        ServiceSpec(
            "backend",
            command_override("LEARNINGOS_BACKEND_COMMAND", backend_command),
            BACKEND_ROOT,
        ),
        ServiceSpec(
            "frontend",
            command_override(
                "LEARNINGOS_FRONTEND_COMMAND",
                (
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--host",
                    args.host,
                    "--port",
                    str(args.frontend_port),
                    "--strictPort",
                ),
            ),
            FRONTEND_ROOT,
        ),
    ]


class Supervisor:
    def __init__(self, *, environment: dict[str, str], grace_seconds: float = 8.0) -> None:
        self.environment = environment
        self.grace_seconds = grace_seconds
        self.processes: list[ManagedProcess] = []

    def start(self, spec: ServiceSpec) -> ManagedProcess:
        print(f"Starting {spec.name}: {shlex.join(spec.command)}")
        kwargs: dict[str, object] = {"cwd": spec.cwd, "env": self.environment}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(spec.command, **kwargs)
        managed = ManagedProcess(spec, process)
        self.processes.append(managed)
        return managed

    def _signal_group(self, managed: ManagedProcess, requested_signal: signal.Signals) -> None:
        if managed.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                if requested_signal == signal.SIGTERM:
                    managed.process.terminate()
                else:
                    managed.process.kill()
            else:
                os.killpg(os.getpgid(managed.process.pid), requested_signal)
        except (OSError, ProcessLookupError):
            return

    def shutdown(self) -> None:
        if not self.processes:
            return
        print("Stopping LearningOS services...")
        for managed in reversed(self.processes):
            self._signal_group(managed, signal.SIGTERM)

        deadline = time.monotonic() + self.grace_seconds
        for managed in reversed(self.processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                managed.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self._signal_group(managed, signal.SIGKILL)
        for managed in reversed(self.processes):
            try:
                managed.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print(f"WARNING: {managed.spec.name} did not stop cleanly", file=sys.stderr)


def _http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return 200 <= response.status < 400
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _worker_ready(socket_path: Path) -> bool:
    if os.name == "nt" or not socket_path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            client.connect(str(socket_path))
        return True
    except OSError:
        return False


def wait_for_smoke(
    supervisor: Supervisor,
    *,
    host: str,
    backend_port: int,
    frontend_port: int,
    worker_socket: Path,
    timeout: float,
) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    probes: dict[str, Callable[[], bool]] = {
        "backend health": lambda: _http_ready(f"http://{probe_host}:{backend_port}/api/v1/system/health"),
        "frontend": lambda: _http_ready(f"http://{probe_host}:{frontend_port}/"),
        "worker IPC": lambda: _worker_ready(worker_socket),
    }
    ready: set[str] = set()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for managed in supervisor.processes:
            return_code = managed.process.poll()
            if return_code is not None:
                print(f"{managed.spec.name} exited before readiness (status {return_code})", file=sys.stderr)
                return False
        for name, probe in probes.items():
            if name not in ready and probe():
                ready.add(name)
                print(f"[READY] {name}")
        if ready == set(probes):
            return True
        time.sleep(0.25)
    missing = sorted(set(probes) - ready)
    print(f"Smoke timeout waiting for: {', '.join(missing)}", file=sys.stderr)
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run diagnostics without starting services")
    parser.add_argument("--smoke", action="store_true", help="start all services, prove readiness, then stop")
    parser.add_argument("--timeout", type=float, default=45.0, help="smoke readiness timeout in seconds")
    parser.add_argument("--reload", action="store_true", help="enable backend auto-reload")
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind host")
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--data-home", help="external mutable data directory (or set LEARNINGOS_HOME)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2
    if args.backend_port == args.frontend_port:
        print("Backend and frontend ports must differ", file=sys.stderr)
        return 2

    python_bin = os.environ.get("LEARNINGOS_PYTHON") or sys.executable or shutil.which("python3")
    if not python_bin:
        print("No Python interpreter found", file=sys.stderr)
        return 2
    data_home = resolve_data_home(args.data_home)
    checks = run_preflight(
        data_home=data_home,
        host=args.host,
        backend_port=args.backend_port,
        frontend_port=args.frontend_port,
        python_bin=python_bin,
    )
    print_checks(checks)
    if not all(check.passed for check in checks):
        print("Preflight FAILED; repair the checks above before starting.", file=sys.stderr)
        return 1
    if args.check:
        print("Preflight PASSED.")
        return 0

    data_home.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["LEARNINGOS_HOME"] = str(data_home)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    supervisor = Supervisor(environment=environment)
    stop_requested = Event()

    def request_stop(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}; requesting clean shutdown.")
        stop_requested.set()

    previous_handlers: dict[signal.Signals, object] = {}
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, request_stop)

    try:
        for spec in service_specs(args, python_bin):
            supervisor.start(spec)

        print(f"Frontend: http://{args.host}:{args.frontend_port}")
        print(f"Backend:  http://{args.host}:{args.backend_port}/api/v1")
        print(f"Data:     {data_home}")
        if args.smoke:
            socket_path = Path(os.environ.get("LEARNINGOS_WORKER_SOCKET", DEFAULT_WORKER_SOCKET))
            passed = wait_for_smoke(
                supervisor,
                host=args.host,
                backend_port=args.backend_port,
                frontend_port=args.frontend_port,
                worker_socket=socket_path,
                timeout=args.timeout,
            )
            print("Platform smoke PASSED." if passed else "Platform smoke FAILED.")
            return 0 if passed else 1

        print("LearningOS is running. Press Ctrl+C to stop all services.")
        while not stop_requested.wait(0.25):
            for managed in supervisor.processes:
                return_code = managed.process.poll()
                if return_code is not None:
                    print(f"{managed.spec.name} exited unexpectedly (status {return_code})", file=sys.stderr)
                    return return_code or 1
        return 0
    except (OSError, ValueError) as exc:
        print(f"Unable to start LearningOS: {exc}", file=sys.stderr)
        return 1
    finally:
        supervisor.shutdown()
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
