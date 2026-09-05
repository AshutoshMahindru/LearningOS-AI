from __future__ import annotations

import sys
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools" / "platform").is_dir() and (
            candidate / "platform" / "backend" / "app"
        ).is_dir():
            return candidate
    raise RuntimeError("could not locate LearningOS repository root")


REPO_ROOT = _repo_root()
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
PLATFORM_TOOLS = REPO_ROOT / "tools" / "platform"
MIGRATE_PY = PLATFORM_TOOLS / "v2_migrate.py"
POPULATED = FIXTURE_ROOT / "populated"
MULTI = FIXTURE_ROOT / "multi"
GARBAGE = FIXTURE_ROOT / "garbage"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
