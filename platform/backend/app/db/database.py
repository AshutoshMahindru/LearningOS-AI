"""SQLite persistence rooted at LEARNINGOS_HOME (never the Git worktree)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

BUSY_TIMEOUT_MS = 5000


def _detect_repo_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        has_platform = (candidate / "platform" / "backend" / "app").is_dir()
        has_git = (candidate / ".git").exists()
        has_arch = (candidate / "architecture" / "learningos-v3").is_dir()
        if has_platform and (has_git or has_arch):
            return candidate
    return None


def get_data_home() -> Path:
    raw = os.environ.get("LEARNINGOS_HOME") or "~/.learningos"
    return Path(raw).expanduser().resolve()


def get_db_path() -> Path:
    return get_data_home() / "learningos.db"


def _reject_home_inside_repo(home: Path) -> None:
    repo_root = _detect_repo_root()
    if repo_root is None:
        return
    home = home.resolve()
    repo_root = repo_root.resolve()
    if home == repo_root or home.is_relative_to(repo_root):
        raise RuntimeError(
            f"LEARNINGOS_HOME must not be inside the Git worktree "
            f"({home} is under {repo_root})"
        )


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with WAL, FK, and busy timeout applied."""
    home = get_data_home()
    _reject_home_inside_repo(home)
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(get_db_path()), timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def init_db() -> Path:
    """Create LEARNINGOS_HOME, open the database, and apply pending migrations."""
    home = get_data_home()
    _reject_home_inside_repo(home)
    home.mkdir(parents=True, exist_ok=True)
    (home / "artifacts").mkdir(exist_ok=True)
    (home / "backups").mkdir(exist_ok=True)
    from app.db.migrations import run_migrations

    run_migrations()
    return home
