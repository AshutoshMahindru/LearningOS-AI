#!/usr/bin/env python3
"""One-click LearningOS launch: bootstrap the managed runtime, then exec ./start.sh."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PLATFORM_DIR = Path(__file__).resolve().parents[1] / "platform"
if str(_PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_DIR))

import install  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the managed LearningOS runtime if needed and start the API, worker, and UI. "
            "Run this from a clone; do not create an environment or install packages yourself."
        ),
        epilog="Example: python3 tools/desktop/launch.py",
    )
    parser.add_argument("--check", action="store_true", help="run host diagnostics without installing or starting")
    parser.add_argument("--data-home", help="external mutable data directory (or set LEARNINGOS_HOME)")
    parser.add_argument(
        "start_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to ./start.sh after -- (for example -- --smoke)",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    runner=None,
    executor=None,
) -> int:
    args = parse_args(argv)
    forwarded = list(args.start_args)
    install_argv: list[str] = []
    if args.check:
        install_argv.append("--check")
    else:
        install_argv.append("--launch")
    if args.data_home:
        install_argv.extend(["--data-home", args.data_home])
    if forwarded:
        if forwarded[0] != "--":
            install_argv.append("--")
        install_argv.extend(forwarded)
    return install.main(install_argv, runner=runner, executor=executor or os.execvpe)


if __name__ == "__main__":
    raise SystemExit(main())
