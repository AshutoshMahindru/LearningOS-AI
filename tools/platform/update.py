#!/usr/bin/env python3
"""Apply a platform/curriculum update after writing a pre-update backup.

A pre-update archive is always written first under
``$LEARNINGOS_HOME/backups/backup_pre_update_{timestamp}_*.tar.gz``.
The live data home is never the restore target; use
``tools/platform/rollback.py --dest-home`` into a clean directory, then
point ``LEARNINGOS_HOME`` at that dest_home.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"

PRE_UPDATE_LABEL = "pre_update"
PRE_UPDATE_ARCHIVE_PREFIX = "backup_pre_update_"
PRE_UPDATE_POINTER_NAME = "pre_update_latest"
PRE_UPDATE_CONFIG_KEY = "last_platform_update"
POST_UPDATE_LEARNER_ID = "post_update_probe"
POST_UPDATE_ARTIFACT_BYTES = b"post-update-artifact"

_PROBE_USERNAME = "post-update-probe"
_PROBE_DISPLAY = "Post Update Probe"


class UpdateError(RuntimeError):
    """Failed to write a pre-update backup or apply an update."""


@dataclass(frozen=True)
class UpdateResult:
    home: Path
    backup: Path
    simulated: bool
    package_id: str | None
    migrations: tuple[int, ...]
    post_update_artifact: str | None


def ensure_backend_on_path() -> Path:
    text = str(BACKEND_ROOT)
    if text not in sys.path:
        sys.path.insert(0, text)
    return BACKEND_ROOT


def is_inside_worktree(path: Path, repo_root: Path = REPO_ROOT) -> bool:
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    return resolved == root or resolved.is_relative_to(root)


def resolve_data_home(raw_path: str | Path | None = None) -> Path:
    configured = raw_path if raw_path is not None else os.environ.get("LEARNINGOS_HOME") or "~/.learningos"
    return Path(configured).expanduser().resolve()


def reject_inside_worktree(path: Path, *, what: str, repo_root: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    if is_inside_worktree(resolved, repo_root):
        raise UpdateError(
            f"{what} must not be inside the Git worktree "
            f"({resolved} is under {repo_root.resolve()})"
        )
    return resolved


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backups_dir(home: Path) -> Path:
    return Path(home) / "backups"


def pre_update_pointer_path(home: Path) -> Path:
    return backups_dir(home) / PRE_UPDATE_POINTER_NAME


def list_pre_update_archives(home: Path) -> list[Path]:
    directory = backups_dir(home)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(PRE_UPDATE_ARCHIVE_PREFIX) and path.name.endswith(".tar.gz")
    )


def write_pre_update_pointer(home: Path, archive: Path) -> Path:
    pointer = pre_update_pointer_path(home)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(archive.name + "\n", encoding="utf-8")
    return pointer


def read_pre_update_pointer(home: Path) -> Path | None:
    pointer = pre_update_pointer_path(home)
    if not pointer.is_file():
        return None
    name = pointer.read_text(encoding="utf-8").strip()
    if not name:
        return None
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = backups_dir(home) / name
    if candidate.is_file():
        return candidate
    return None


def latest_pre_update_archive(home: Path) -> Path | None:
    pointed = read_pre_update_pointer(home)
    if pointed is not None:
        return pointed
    archives = list_pre_update_archives(home)
    if not archives:
        return None
    return max(archives, key=lambda path: path.stat().st_mtime)


def bind_data_home(home: Path) -> Path:
    resolved = reject_inside_worktree(resolve_data_home(home), what="LEARNINGOS_HOME")
    os.environ["LEARNINGOS_HOME"] = str(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def create_pre_update_backup(home: Path | None = None) -> Path:
    """Checkpoint the live home and write ``backup_pre_update_{timestamp}_*.tar.gz``."""
    ensure_backend_on_path()
    resolved = bind_data_home(resolve_data_home(home))
    from app.db.backup import create_backup

    dest = backups_dir(resolved)
    dest.mkdir(parents=True, exist_ok=True)
    archive = create_backup(dest, label=PRE_UPDATE_LABEL)
    if PRE_UPDATE_ARCHIVE_PREFIX not in archive.name:
        raise UpdateError(f"pre-update backup label missing from {archive.name}")
    write_pre_update_pointer(resolved, archive)
    return archive


def _read_config(home: Path) -> dict[str, Any]:
    path = home / "config.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UpdateError(f"config.json must be an object: {path}")
    return payload


def _write_config(home: Path, payload: dict[str, Any]) -> None:
    path = home / "config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_update_metadata(
    home: Path,
    *,
    backup: Path,
    simulated: bool,
    package_id: str | None,
    migrations: tuple[int, ...],
) -> None:
    config = _read_config(home)
    config[PRE_UPDATE_CONFIG_KEY] = {
        "at": utc_stamp(),
        "backup": backup.name,
        "mode": "simulate" if simulated else "apply",
        "package_id": package_id,
        "migrations": list(migrations),
    }
    _write_config(home, config)


def _insert_probe_learner() -> None:
    from app.db.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            (POST_UPDATE_LEARNER_ID, _PROBE_USERNAME, _PROBE_DISPLAY),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
    finally:
        conn.close()


def _put_post_update_artifact() -> str:
    from app.core.artifact_store import ArtifactStore

    return ArtifactStore().put(POST_UPDATE_ARTIFACT_BYTES, media_type="text/plain")


def _load_curriculum_package(package_dir: Path) -> str:
    from app.core.mission_loader import load_package
    from app.core.registry import register_package

    package = load_package(package_dir)
    register_package(package)
    return package.id


def apply_update(
    *,
    home: Path | str | None = None,
    simulate: bool = False,
    package: Path | str | None = None,
) -> UpdateResult:
    """Write a pre-update backup, then apply a platform/curriculum update."""
    ensure_backend_on_path()
    resolved = bind_data_home(resolve_data_home(home))
    db_path = resolved / "learningos.db"
    if not db_path.is_file():
        raise UpdateError(f"no LearningOS database to update at {db_path}")

    backup = create_pre_update_backup(resolved)

    from app.db.migrations import run_migrations

    applied = tuple(run_migrations())
    package_id: str | None = None
    post_digest: str | None = None
    if package is not None:
        package_id = _load_curriculum_package(Path(package).expanduser().resolve())
    if simulate:
        _insert_probe_learner()
        post_digest = _put_post_update_artifact()
    _record_update_metadata(
        resolved,
        backup=backup,
        simulated=simulate,
        package_id=package_id,
        migrations=applied,
    )
    return UpdateResult(
        home=resolved,
        backup=backup,
        simulated=simulate,
        package_id=package_id,
        migrations=applied,
        post_update_artifact=post_digest,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="live LEARNINGOS_HOME (default: $LEARNINGOS_HOME or ~/.learningos)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="after the backup, apply a simulated update (probe learner + artifact + config marker)",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help="curriculum package directory to load after the pre-update backup",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = apply_update(home=args.home, simulate=args.simulate, package=args.package)
    except UpdateError as exc:
        print(f"update failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"update failed: {exc}", file=sys.stderr)
        return 1
    mode = "simulated" if result.simulated else "applied"
    print(f"Pre-update backup: {result.backup}")
    print(f"Update {mode} against {result.home}")
    if result.package_id:
        print(f"Curriculum package: {result.package_id}")
    if result.migrations:
        print(f"Migrations applied: {', '.join(str(item) for item in result.migrations)}")
    print("Roll back with: python3 tools/platform/rollback.py --dest-home <clean-dir>")
    print("Then: export LEARNINGOS_HOME=<clean-dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
