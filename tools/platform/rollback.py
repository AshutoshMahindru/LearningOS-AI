#!/usr/bin/env python3
"""Restore a pre-update backup into a clean dest_home (never the live home).

dest_home must not exist or must be empty, must not be the live
LEARNINGOS_HOME, and must not sit inside the Git worktree.

After a successful rollback, point LEARNINGOS_HOME at the restored dest:

  export LEARNINGOS_HOME=/path/to/restored-home
  ./start.sh

The live data home is not a valid restore target because it already
contains the database. Leave it in place and start the app against dest_home.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    _REPO = Path(__file__).resolve().parents[2]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

from tools.platform.update import (
    REPO_ROOT,
    ensure_backend_on_path,
    is_inside_worktree,
    latest_pre_update_archive,
    resolve_data_home,
)


class RollbackError(RuntimeError):
    """Failed to restore a pre-update backup into dest_home."""


@dataclass(frozen=True)
class RollbackResult:
    archive: Path
    dest_home: Path
    live_home: Path
    instructions: str


def post_rollback_instructions(dest_home: Path, live_home: Path) -> str:
    dest = Path(dest_home).expanduser().resolve()
    live = Path(live_home).expanduser().resolve()
    return (
        f"Restored pre-update backup into dest_home: {dest}\n"
        f"Live data home was not modified: {live}\n"
        "\n"
        "Point LEARNINGOS_HOME at the restored dest_home before starting LearningOS:\n"
        "\n"
        f"  export LEARNINGOS_HOME={dest}\n"
        "  ./start.sh\n"
        "\n"
        "The live data home is not a valid restore target.\n"
    )


def resolve_archive(archive: Path | str | None, live_home: Path) -> Path:
    if archive is not None:
        path = Path(archive).expanduser()
        if not path.is_absolute():
            nested = live_home / "backups" / path
            path = nested if nested.is_file() else path.resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise RollbackError(f"backup archive not found: {path}")
        return path
    found = latest_pre_update_archive(live_home)
    if found is None:
        raise RollbackError(f"no pre-update backup under {live_home / 'backups'}")
    return found


def validate_dest_home(
    dest_home: Path | str,
    *,
    live_home: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    dest = Path(dest_home).expanduser().resolve()
    live = Path(live_home).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    if dest == live:
        raise RollbackError(f"live data home is not a valid restore target: {dest}")
    if is_inside_worktree(dest, root):
        raise RollbackError(
            f"dest_home must not be inside the Git worktree ({dest} is under {root})"
        )
    if dest.exists():
        if not dest.is_dir():
            raise RollbackError(f"dest_home is not a directory: {dest}")
        if any(dest.iterdir()):
            raise RollbackError(f"restore destination is not empty: {dest}")
    return dest


def rollback_to_dest(
    dest_home: Path | str,
    *,
    archive: Path | str | None = None,
    live_home: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
) -> RollbackResult:
    """Restore the pre-update backup into a clean dest_home outside the worktree."""
    ensure_backend_on_path()
    live = resolve_data_home(live_home)
    if is_inside_worktree(live, repo_root):
        raise RollbackError(
            f"LEARNINGOS_HOME must not be inside the Git worktree "
            f"({live} is under {Path(repo_root).resolve()})"
        )
    dest = validate_dest_home(dest_home, live_home=live, repo_root=repo_root)
    archive_path = resolve_archive(archive, live)
    from app.db.backup import restore_backup

    try:
        restore_backup(archive_path, dest)
    except FileExistsError as exc:
        raise RollbackError(str(exc) or f"restore destination is not empty: {dest}") from exc
    except FileNotFoundError as exc:
        raise RollbackError(str(exc)) from exc
    except (NotADirectoryError, ValueError) as exc:
        raise RollbackError(str(exc)) from exc
    return RollbackResult(
        archive=archive_path,
        dest_home=dest,
        live_home=live,
        instructions=post_rollback_instructions(dest, live),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "After rollback, point LEARNINGOS_HOME at dest_home:\n"
            "  export LEARNINGOS_HOME=/path/to/restored-home\n"
            "  ./start.sh\n"
        ),
    )
    parser.add_argument(
        "--dest-home",
        dest="dest_home",
        type=Path,
        required=True,
        help="clean restore directory (must not be the live LEARNINGOS_HOME or the Git worktree)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="pre-update backup archive (default: latest backup_pre_update_*.tar.gz under $LEARNINGOS_HOME/backups)",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="live LEARNINGOS_HOME that holds the pre-update backup (default: $LEARNINGOS_HOME or ~/.learningos)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = rollback_to_dest(args.dest_home, archive=args.archive, live_home=args.home)
    except RollbackError as exc:
        print(f"rollback failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"rollback failed: {exc}", file=sys.stderr)
        return 1
    print(result.instructions, end="")
    print(f"Archive: {result.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
