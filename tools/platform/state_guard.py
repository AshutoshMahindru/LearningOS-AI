#!/usr/bin/env python3
"""Reject mutable LearningOS V3 learner state inside the Git worktree."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SKIP_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__"}
PLATFORM_STATE_DIRECTORIES = {".learningos", "artifacts", "backups", "sessions", "tracking"}


def _is_database_file(name: str) -> bool:
    return name == "learningos.db" or name.startswith("learningos.db-")


def is_learner_state_path(relative_path: Path) -> bool:
    """Return True for a path reserved for mutable V3 learner state."""
    parts = relative_path.parts
    if ".learningos" in parts or _is_database_file(relative_path.name):
        return True
    if parts and parts[0] == "platform":
        return bool(set(parts[1:-1]) & PLATFORM_STATE_DIRECTORIES)
    return False


def tracked_paths(repo_root: Path) -> Iterable[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    for raw_path in completed.stdout.split(b"\0"):
        if raw_path:
            yield Path(os.fsdecode(raw_path))


def tracked_violations(repo_root: Path) -> list[Path]:
    return sorted(path for path in tracked_paths(repo_root) if is_learner_state_path(path))


def filesystem_violations(repo_root: Path) -> list[Path]:
    violations: set[Path] = set()
    for current, directories, files in os.walk(repo_root):
        current_path = Path(current)
        relative_directory = current_path.relative_to(repo_root)
        directories[:] = [name for name in directories if name not in SKIP_DIRECTORIES]

        for directory in directories:
            relative = relative_directory / directory
            if is_learner_state_path(relative / ".state"):
                violations.add(relative)
        for filename in files:
            relative = relative_directory / filename
            if is_learner_state_path(relative):
                violations.add(relative)
    return sorted(violations)


def find_violations(repo_root: Path) -> list[Path]:
    return sorted(set(tracked_violations(repo_root)) | set(filesystem_violations(repo_root)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the current LearningOS checkout)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo.expanduser().resolve()
    try:
        violations = find_violations(repo_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Learner-state guard could not inspect {repo_root}: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("Mutable learner state is not allowed inside the Git worktree:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation.as_posix()}", file=sys.stderr)
        print("Set LEARNINGOS_HOME to a directory outside the repository.", file=sys.stderr)
        return 1

    print("Learner-state guard PASSED: no mutable V3 state is present in the worktree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
