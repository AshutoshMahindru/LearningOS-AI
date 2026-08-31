#!/usr/bin/env python3
"""Supervise platform/worker/daemon.py. Not the 11A entrypoint."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DAEMON_PATH = Path(__file__).resolve().parent / "daemon.py"
RESTART_DELAY_SEC = 0.5
MAX_CRASH_RESTARTS = 8


def run_worker_with_watchdog(timeout_sec: int | None = None) -> int:
    """Launch daemon.py, forward stop signals, and restart it after crashes.

    timeout_sec, when set and positive, is a wall-clock cap on the supervisor
    itself (legacy). The default is no cap so the worker can stay up for health.
    """
    stop = {"requested": False}
    process: subprocess.Popen[bytes] | None = None
    deadline = time.monotonic() + timeout_sec if timeout_sec and timeout_sec > 0 else None

    def _forward(signum: int, _frame: object) -> None:
        stop["requested"] = True
        child = process
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signal.SIGTERM)
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    restarts = 0
    while not stop["requested"]:
        process = subprocess.Popen([sys.executable, str(DAEMON_PATH)])
        while process.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                process.send_signal(signal.SIGKILL)
                process.wait()
                return 1
            time.sleep(0.2)
        if stop["requested"] or process.returncode == 0:
            return 0
        restarts += 1
        if restarts > MAX_CRASH_RESTARTS:
            print(
                f"Watchdog: daemon crashed {restarts} times; giving up (status {process.returncode}).",
                file=sys.stderr,
            )
            return process.returncode or 1
        print(
            f"Watchdog: daemon exited {process.returncode}; restarting in {RESTART_DELAY_SEC}s.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(RESTART_DELAY_SEC)
    return 0


if __name__ == "__main__":
    raw = os.environ.get("LEARNINGOS_WATCHDOG_TIMEOUT_SEC")
    timeout = int(raw) if raw else None
    raise SystemExit(run_worker_with_watchdog(timeout_sec=timeout))
