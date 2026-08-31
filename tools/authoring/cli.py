#!/usr/bin/env python3
"""Validate, preview, or dry-run a LearningOS fixture package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    if __name__ != "__main__":
        raise ImportError("import tools.authoring.cli, or run python -m tools.authoring")
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.append(repo)
    from tools.authoring.cli import main as _packaged_main

    raise SystemExit(_packaged_main())

from .errors import AuthoringError
from .package import default_package_path
from .preview import preview_package
from .simulate import simulate_package
from .validate import format_validate_result, validate_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help="fixture package directory (default: platform/fixtures/f01)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="validate package integrity and WP-136 mission documents")
    sub.add_parser("preview", help="print stage list and assistance policies")

    sim = sub.add_parser("simulate", help="dry-run stage sequence under a temporary LEARNINGOS_HOME")
    sim.add_argument(
        "--home",
        type=Path,
        default=None,
        help="external LEARNINGOS_HOME (default: a fresh temp directory outside the Git worktree)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    package = args.package if args.package is not None else default_package_path()
    try:
        if args.command == "validate":
            result = validate_package(package)
            stream = sys.stdout if result.ok else sys.stderr
            print(format_validate_result(result), file=stream)
            return 0 if result.ok else 1
        if args.command == "preview":
            print(preview_package(package), end="")
            return 0
        if args.command == "simulate":
            outcome = simulate_package(package, home=args.home)
            print(outcome.text, end="")
            print(f"Trace             {outcome.trace_path}")
            return 0
    except AuthoringError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
